import unittest
import os
from src.Rag.AuroraRag import AuroraRag

from src.MultiRag.MultiAuroraRag import MultiAuroraRag


class TestRag(unittest.TestCase):

    def setUp(self) -> None:
        print(os.getcwd())
        root_path = os.path.join(os.getcwd(), "/data/test.pdf")
        self.rag = AuroraRag("data/test.pdf", "all-mpnet-base-v2")

    def test_reading_pdf(self):
        pages_and_text = self.rag.encoder.recieve_data()
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
        query = "what is potien"
        results = self.rag.search(query)
        self.rag.display_results(results)
        self.assertEqual(len(results), 5)

    def test_multi_aurora_rag(self):
        rag = MultiAuroraRag(os.getcwd() + "/data/rice", "all-mpnet-base-v2")
        result = rag.search("what nutrients are important for rice")
        rag.display_results(result)
        self.assertEqual(len(result), 5)
