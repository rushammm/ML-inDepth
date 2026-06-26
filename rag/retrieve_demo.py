from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

text = """Why RAG is Used. Up-to-Date Knowledge: LLMs have knowledge cut-off
dates and cannot inherently know private or newly published data unless it is
provided to them. Cost Efficiency: It is much cheaper and faster to update a
database of documents than to repeatedly fine-tune or re-train an entire
language model. Security: Organizations can restrict an AI chatbot to only
pull answers from approved, verified internal documents."""

# INDEXING: chunk the document with overlap, then embed each chunk
words = text.split()
chunk_size = 35
overlap = 10
step = chunk_size - overlap
chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
chunk_embeddings = embedder.encode(chunks)

# RETRIEVAL: embed the question, take the top-k closest chunks
query = "why is RAG cheaper than fine-tuning?"
query_embedding = embedder.encode(query)
scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

k = 2
top_values, top_indices = scores.topk(k)
retrieved = [chunks[i] for i in top_indices]

# AUGMENT: stuff the retrieved chunks into the prompt
context = "\n".join(retrieved)
prompt = f"""Answer the question using ONLY the context below.
Context: {context}
Question: {query}"""

# GENERATE: let the LLM write a grounded answer
answer = generator(prompt, max_new_tokens=100)[0]["generated_text"]

print("QUESTION:", query)
print("ANSWER:", answer)
