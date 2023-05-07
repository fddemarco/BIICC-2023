from read_file import ReadFile


class AppendFiles:
    def __init__(self, input_filenames, output_filename, chunk_size=int(1e8)):
        self.chunk_size = chunk_size
        self.input_filenames = input_filenames
        self.output_filename = output_filename
        self.file = None

    def __enter__(self):
        self.file = open(self.output_filename, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def run(self):
        for file in self.input_filenames:
            self.append_file(file)

    def append_file(self, input_file):
        with ReadFile(input_file) as reader:
            for data in reader:
                self.file.write(data)
            self.file.write('\n')
