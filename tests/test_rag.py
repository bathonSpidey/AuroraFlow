import unittest
import os

from src.AuroraRag import AuroraRag

class TestRag(unittest.TestCase):
    
    def setUp(self) -> None:
        root_path = os.path.join(os.getcwd(), "tests/data/test.pdf")
        self.rag = AuroraRag(root_path, "all-mpnet-base-v2")
    def test_reading_pdf(self):
        pages_and_text = self.rag.encoder.recieve_data()
        print(len(pages_and_text))
        self.assertEqual(len(pages_and_text), 1208)
        
    def test_processing_pdf(self):
        pages_and_text = self.rag.encoder.process_data()
        self.assertEqual(len(pages_and_text[0]["sentences"]), 1)
        
    def test_chunkify(self):
        pages_and_text = self.rag.encoder.make_chunks()
        self.assertEqual(len(pages_and_text[0]["sentence_chunks"]), 1)
        
    def test_make_chunks(self):
        pages_and_chunks = self.rag.encoder.process_chunks()
        self.assertEqual(pages_and_chunks[0]["chunk_token_count"], 7.25)
        
    def test_makeing_embedings_should_provide_right_data_frame(self):
        embeddings = self.rag.pdf_to_dataframe()
        self.assertEqual(embeddings.shape[0], 1680)
        
    def test_matching_query_with_text(self):
        query = "foods high in fiber"
        results = self.rag.search(query)
        print(results)
        self.assertEqual(len(results), 5)
        
        
        
        