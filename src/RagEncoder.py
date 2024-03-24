from src.PagesProcessor import PagesProcessor
from src.PdfReader import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
import os
import numpy as np

print(os.path.join(os.getcwd()))
class RagEncoder:
    def __init__(self, path, model):
        self.pdf_reader = PdfReader(path)
        self.processor = PagesProcessor()
        self.embedding_model = SentenceTransformer(model, device="cpu")
        self.min_token_length = 30

    def recieve_data(self):
        return self.pdf_reader.open_pdf()

    def process_data(self):
        pages_and_text = self.pdf_reader.open_pdf()
        pages_and_text = self.processor.add_senteces(pages_and_text)
        return pages_and_text

    def make_chunks(self):
        pages_and_text = self.process_data()
        return self.processor.chunkify(pages_and_text)

    def process_chunks(self):
        pages_and_text = self.make_chunks()
        return self.processor.add_chunk_metadata(pages_and_text)

    def make_embeddings(self):
        pages_and_chunks = self.process_chunks()
        return self.add_embeddings(pages_and_chunks)
    
    def add_embeddings(self, pages_and_chunks, device="cuda"):
        root = os.path.join(os.getcwd(), "src/.cache")
        print("Directory created: ", root)
        if os.path.exists(f"{root}/rag_embeddings.csv"):
            text_chunks_and_embedding_df = pd.read_csv(f"{root}/rag_embeddings.csv")
            text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            return text_chunks_and_embedding_df
        df = pd.DataFrame(pages_and_chunks)
        filtered_df = df[df["chunk_token_count"] > self.min_token_length].to_dict(orient="records")
        self.embedding_model.to(device)
        for item in tqdm(filtered_df):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        text_chunks_and_embedding_df = pd.DataFrame(filtered_df)
        text_chunks_and_embedding_df.to_csv(f"{root}/rag_embeddings.csv", index=False)
        return text_chunks_and_embedding_df
    
    def embed_query(self, query, device):
        return self.embedding_model.encode(query, convert_to_tensor=True).to(device)
    
    def match(self, query_embedding, database_embeddings):
        return util.pytorch_cos_sim(query_embedding, database_embeddings)[0]
    