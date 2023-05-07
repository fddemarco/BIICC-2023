class ReadFile:
    def __init__(self, filename, chunk_size=int(1e8)):
        self.chunk_size = chunk_size
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        data = self.file.read(self.chunk_size)
        if not data:
            raise StopIteration
        return data
