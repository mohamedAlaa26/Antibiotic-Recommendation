import chromadb
from chromadb.utils import embedding_functions
import Prepare_documents_metadata as prep

# Load prepared data
documents = prep.documents
metadatas = prep.metadatas
ids = prep.ids

print(f"✅ Prepared {len(documents)} documents for indexing")

# Initialize Chroma client
client = chromadb.Client()

# Use SentenceTransformers for embeddings
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create collection
collection = client.get_or_create_collection("antibiotic_cases", embedding_function=embedding_fn)

# ✅ Add in chunks
batch_size = 5000
for i in range(0, len(documents), batch_size):
    collection.add(
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )
    print(f"✅ Indexed {min(i+batch_size, len(documents))} documents")

print("🎯 All documents successfully indexed!")
