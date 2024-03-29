import fitz
from tqdm.asyncio import tqdm


class PdfReader:
    def __init__(self, path):
        self.data_path = path

    def open_pdf(self, page_number_offset):
        doc = fitz.open(self.data_path)
        pages_and_text = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text("text")
            text = self.text_formatter(text)
            pages_and_text.append({"page_number": page_number-page_number_offset,
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_setence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,
                                    "text": text})
        return pages_and_text

    def text_formatter(self, text):
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text
