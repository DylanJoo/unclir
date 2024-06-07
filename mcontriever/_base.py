from pyserini.encode import JsonlCollectionIterator

from datasets import load_from_disk

# dataset = load_from_disk('/{HOME}/datasets/neuclir-csl')
# dataset = load_from_disk('/{HOME}/datasets/neuclir1')

class HgfCollectionIterator(JsonlCollectionIterator):
    def __init__(self, collection_path: str, fields=None, docid_field=None, delimiter="\n"):
        super().__init__(self, collection_path, fields, docid_field, delimiter)
        # Assume multimodal input files are located in the same directory as the collection file
        if os.path.isdir(collection_path):
            self.collection_dir = collection_path
        else:
            self.collection_dir = os.path.dirname(collection_path)
        if fields:
            self.fields = fields
        else:
            self.fields = ['text']
        self.docid_field = docid_field
        self.delimiter = delimiter
        self.all_info = self._load(collection_path)
        self.size = len(self.all_info['id'])
        self.batch_size = 1
        self.shard_id = 0
        self.shard_num = 1

    def _parse_fields_from_info(self, info):
        """
        :params info: dict, containing all fields as speicifed in self.fields either under 
        the key of the field name or under the key of 'contents'.  If under `contents`, this 
        function will parse the input contents into each fields based the self.delimiter
        return: List, each corresponds to the value of self.fields
        """
        n_fields = len(self.fields)

        # if all fields are under the key of info, read these rather than 'contents' 
        if all([field in info for field in self.fields]):
            return [info[field].strip() for field in self.fields]

        assert "contents" in info, f"contents not found in info: {info}"
        contents = info['contents']
        # whether to remove the final self.delimiter (especially \n)
        # in CACM, a \n is always there at the end of contents, which we want to remove;
        # but in SciFact, Fiqa, and more, there are documents that only have title but not text (e.g. "This is title\n")
        # where the trailing \n indicates empty fields
        if contents.count(self.delimiter) == n_fields:
            # the user appends one more delimiter to the end, we remove it
            if contents.endswith(self.delimiter):
                # not using .rstrip() as there might be more than one delimiters at the end
                contents = contents[:-len(self.delimiter)]
        return [field.strip(" ") for field in contents.split(self.delimiter)]

    def _load(self, collection_path):
        filenames = []
        if os.path.isfile(collection_path):
            filenames.append(collection_path)
        else:
            for filename in os.listdir(collection_path):
                filenames.append(os.path.join(collection_path, filename))
        all_info = {field: [] for field in self.fields}
        all_info['id'] = []
        for filename in filenames:
            with open(filename) as f:
                for line_i, line in tqdm(enumerate(f)):
                    info = json.loads(line)
                    if self.docid_field:
                        _id = info.get(self.docid_field, None)
                    else:
                        _id = info.get('id', info.get('_id', info.get('docid', None)))
                    if _id is None:
                        raise ValueError(f"Cannot find f'`{self.docid_field if self.docid_field else '`id` or `_id` or `docid'}`' from {filename}.")
                    all_info['id'].append(str(_id))
                    fields_info = self._parse_fields_from_info(info)
                    if len(fields_info) != len(self.fields):
                        raise ValueError(
                            f"{len(fields_info)} fields are found at Line#{line_i} in file {filename}." \
                            f"{len(self.fields)} fields expected." \
                            f"Line content: {info['contents']}"
                        )

                    for i in range(len(fields_info)):
                        if 'path' in self.fields[i]:
                            _info = fields_info[i]
                            if not _info.startswith(("http://", "https://")):
                                fields_info[i] = os.path.join(self.collection_dir, fields_info[i])
                        all_info[self.fields[i]].append(fields_info[i])
        return all_info


