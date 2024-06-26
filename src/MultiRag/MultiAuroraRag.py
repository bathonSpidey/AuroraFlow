import os
import textwrap

import numpy as np
import torch

from src.MultiRag.MultiPdfReader import MultiPdfReader
from src.MultiRag.MultiRagEncoder import MultiRagEncoder


class MultiAuroraRag:
    def __init__(self, path, model, page_offset=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.files = os.listdir(path)
        last_slash_index = path.rfind("/")
        self.folder_path = "Multi_Aurora_"+path[last_slash_index + 1:]
        self.data = MultiPdfReader(path).read()
        self.multirag_encoder = MultiRagEncoder(self.data, model, self.device)

    def pdf_to_dataframe(self, folder_path,clean_cache):
        return self.multirag_encoder.make_embeddings(folder_path, clean_cache)

    def search(self, query, clean_cache=False):
        df = self.pdf_to_dataframe(self.folder_path, clean_cache)
        database_embeddings = torch.tensor(np.stack(df["embedding"].tolist(), axis=0), dtype=torch.float32).to(
            self.device)
        pages_and_chunks = df.to_dict(orient="records")
        query_embedding = self.multirag_encoder.embed_query(query, self.device)
        scores = self.multirag_encoder.match(query_embedding, database_embeddings)
        top_answers = self.retrieve(pages_and_chunks, scores)
        return top_answers

    def retrieve(self, pages_and_chunks, scores):
        top_five = torch.topk(scores, 5)
        top_answers = []
        for score, index in zip(top_five[0], top_five[1]):
            answer = {}
            answer["score"] = score
            answer["response"] = self.wrapped(pages_and_chunks[index]["sentence_chunk"])
            top_answers.append(answer)
        return top_answers

    def wrapped(self, text, wrap_length=80):
        return textwrap.fill(text, wrap_length)

    def display_results(self, results):
        display = ""
        for result in results:
            display += result["response"] + "\n \n \n"
        print(display)
        return display
