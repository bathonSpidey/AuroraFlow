from spacy.lang.en import English
import re
from tqdm.auto import tqdm


class PagesProcessor:
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.chunk_size = 10
        
        

    def add_senteces(self, pages_and_text):
        for item in tqdm(pages_and_text):
            item["sentences"] = list(self.nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])
        return pages_and_text
    
    def chunkify(self, pages_and_text):
        for item in tqdm(pages_and_text):
            item["sentence_chunks"] = [item["sentences"][i:i+self.chunk_size] for i in range(0, len(item["sentences"]), self.chunk_size)]
        return pages_and_text
    
    def add_chunk_metadata(self, pages_and_text):
        pages_and_chunks = []
        for item in tqdm(pages_and_text):
            for chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]
                joined_sentence_chunk = "".join(chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
                chunk_dict["sentence_chunk"] = joined_sentence_chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
                pages_and_chunks.append(chunk_dict)
        return pages_and_chunks
    
    
        
        