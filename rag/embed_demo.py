from sentence_transformers import SentenceTransformer 
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ['i love icecream', 'my favourite icecream falvor is chocolate', 'i hate my shoes' ]
embeddings = model.encode(sentences)
print(embeddings.shape)

scores = util.cos_sim(embeddings, embeddings)
print(scores)