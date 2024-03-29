import os

from src.Rag.PdfReader import PdfReader


class MultiPdfReader:
    def __init__(self, path):
        self.path = path
        files = os.listdir(path)
        self.pdf_files = [file for file in files if file.endswith('.pdf')]
        self.data = []

    def read(self):
        for file in self.pdf_files:
            self.data.append(PdfReader(os.path.join(self.path, file)).open_pdf(0))
        return self.data
