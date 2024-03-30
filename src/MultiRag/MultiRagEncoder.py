import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm.asyncio import tqdm

from src.Rag.PagesProcessor import PagesProcessor


class MultiRagEncoder:
    def __init__(self, data, model, device):
        self.device = device
        self.data = data
        self.processor = PagesProcessor()
        self.embedding_model = SentenceTransformer(model, device="cpu")
        self.min_token_length = 10

    def make_embeddings(self,folder_path, clean_cache, ):
        root = os.path.join(os.getcwd(), ".cache")
        if os.path.exists(f"{root}/rag_embeddings_{folder_path}.csv") and not clean_cache:
            text_chunks_and_embedding_df = pd.read_csv(f"{root}/rag_embeddings_{folder_path}.csv")
            text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" "))
            return text_chunks_and_embedding_df
        processed_df = []
        for file in self.data:
            pages_and_text = self.processor.add_senteces(file)
            chunks = self.processor.chunkify(pages_and_text)
            processed = self.processor.add_chunk_metadata(chunks)
            processed_df.append(self.add_embeddings(processed))
        combined_df = pd.concat(processed_df, ignore_index=True)
        if not os.path.exists(root):
            os.makedirs(root)
        combined_df.to_csv(f"{root}/rag_embeddings_{folder_path}.csv", index=False)
        return combined_df

    def add_embeddings(self, processed):
        df = pd.DataFrame(processed)
        filtered_df = df[df["chunk_token_count"] > self.min_token_length].to_dict(orient="records")
        self.embedding_model.to(self.device)
        for item in tqdm(filtered_df):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        return pd.DataFrame(filtered_df)

    def embed_query(self, query, device):
        return self.embedding_model.encode(query, convert_to_tensor=True).to(device)

    def match(self, query_embedding, database_embeddings):
        return util.pytorch_cos_sim(query_embedding, database_embeddings)[0]
