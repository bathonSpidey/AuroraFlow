import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm.asyncio import tqdm

from src.Rag.PdfReader import PdfReader
from src.Rag.PagesProcessor import PagesProcessor


class RagEncoder:
    def __init__(self, path, model, page_offset, device):
        self.device = device
        self.pdf_reader = PdfReader(path)
        self.processor = PagesProcessor()
        self.embedding_model = SentenceTransformer(model, device="cpu")
        self.min_token_length = 30
        self.page_offset = page_offset

    def recieve_data(self):
        return self.pdf_reader.open_pdf(self.page_offset)

    def process_data(self):
        pages_and_text = self.recieve_data()
        pages_and_text = self.processor.add_senteces(pages_and_text)
        return pages_and_text

    def make_chunks(self):
        pages_and_text = self.process_data()
        return self.processor.chunkify(pages_and_text)

    def process_chunks(self):
        pages_and_text = self.make_chunks()
        return self.processor.add_chunk_metadata(pages_and_text)

    def make_embeddings(self, clean_cache=False):
        pages_and_chunks = self.process_chunks()
        return self.add_embeddings(pages_and_chunks, clean_cache)

    def add_embeddings(self, pages_and_chunks, clean_cache):
        root = os.path.join(os.getcwd(), ".cache")
        if os.path.exists(f"{root}/rag_embeddings.csv") and not clean_cache:
            text_chunks_and_embedding_df = pd.read_csv(f"{root}/rag_embeddings.csv")
            text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" "))
            return text_chunks_and_embedding_df
        df = pd.DataFrame(pages_and_chunks)
        filtered_df = df[df["chunk_token_count"] > self.min_token_length].to_dict(orient="records")
        self.embedding_model.to(self.device)
        for item in tqdm(filtered_df):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        text_chunks_and_embedding_df = pd.DataFrame(filtered_df)
        if not os.path.exists(root):
            os.makedirs(root)
        text_chunks_and_embedding_df.to_csv(f"{root}/rag_embeddings.csv", index=False)
        return text_chunks_and_embedding_df

    def embed_query(self, query, device):
        return self.embedding_model.encode(query, convert_to_tensor=True).to(device)

    def match(self, query_embedding, database_embeddings):
        return util.pytorch_cos_sim(query_embedding, database_embeddings)[0]
