from sentence_transformers import SentenceTransformer 

model = SentenceTransformer("all-MiniLM-L6-v2")
text = """Why RAG is UsedUp-to-Date Knowledge: LLMs have knowledge cut-off dates and cannot inherently know private or newly published data unless it is provided to them.Cost Efficiency: It is much cheaper and faster to update a database of documents than to repeatedly fine-tune or re-train an entire language model.Security: Organizations can restrict an AI chatbot to only pull answers from approved, verified internal documents."""
words = text.split()
chunk_size = 20
chunks = []
for i in range(0, len(words), chunk_size):
     chunk = " ".join(words[i : i + chunk_size])
     chunks.append(chunk)

print(len(chunks))

for i, chunk in enumerate(chunks):
    print(i, chunk)
