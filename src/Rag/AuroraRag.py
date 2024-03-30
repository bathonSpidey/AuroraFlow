import textwrap

import numpy as np
import torch

from src.Rag.RagEncoder import RagEncoder


class AuroraRag:
    def __init__(self, path, model, page_offset=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = RagEncoder(path, model, page_offset, self.device)



    def pdf_to_dataframe(self, clean_cache):
        return self.encoder.make_embeddings(clean_cache)

    def search(self, query, clean_cache=False):
        df = self.pdf_to_dataframe(clean_cache)
        database_embeddings = torch.tensor(np.stack(df["embedding"].tolist(), axis=0), dtype=torch.float32).to(
            self.device)
        pages_and_chunks = df.to_dict(orient="records")
        query_embedding = self.encoder.embed_query(query, self.device)
        scores = self.encoder.match(query_embedding, database_embeddings)
        top_answers = self.retrieve(pages_and_chunks, scores)
        return top_answers

    def retrieve(self, pages_and_chunks, scores):
        top_five = torch.topk(scores, 5)
        top_answers = []
        for score, index in zip(top_five[0], top_five[1]):
            answer = {}
            answer["score"] = score
            answer["response"] = self.wrapped(pages_and_chunks[index]["sentence_chunk"])
            answer["file_name"] = pages_and_chunks[index]["file_name"]
            answer["page_number"] = pages_and_chunks[index]["page_number"]
            top_answers.append(answer)
        return top_answers

    def wrapped(self, text, wrap_length=80):
        return textwrap.fill(text, wrap_length)

    def display_results(self, results):
        display = ""
        for result in results:
            display += result["response"] + "\n"
            display += f"For more information check: {result['file_name']} at index {result['page_number'] + 1}\n"
            display += "Please note index is the total number of pages in the pdf regardless the page number \n \n"

        print(display)
        return display
