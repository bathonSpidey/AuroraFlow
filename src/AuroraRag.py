
from src.RagEncoder import RagEncoder
import torch
import numpy as np
import textwrap



class AuroraRag:
    def __init__(self, path, model):
        self.encoder = RagEncoder(path, model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def pdf_to_dataframe(self):
        return self.encoder.make_embeddings()
    
    def search(self, query):
        df = self.pdf_to_dataframe()
        database_embeddings = torch.tensor(np.stack(df["embedding"].tolist(), axis=0), dtype=torch.float32).to(self.device)
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
            top_answers.append(answer)
        return top_answers
    
    def wrapped(self, text, wrap_length=80):
        return textwrap.fill(text, wrap_length)
    
    
    
        

    