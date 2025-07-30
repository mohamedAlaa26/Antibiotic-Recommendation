import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama

# Step 1: Connect to Chroma
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client_chroma = chromadb.Client()
collection = client_chroma.get_or_create_collection("antibiotic_cases", embedding_function=embedding_fn)

# Step 2: Load a local LLaMA 
# Example: Download a small GGUF model from HuggingFace (e.g., llama-2-7b.gguf)
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",  # Path to your local model
    n_ctx=4096,  # Context size
    n_threads=6   # Adjust to your CPU
)

def answer_query(query, top_k=3):
    # Step 3: Retrieve top matches from Chroma
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_docs = "\n".join(results['documents'][0])

    # Step 4: Create the prompt for LLaMA
    prompt = f"""
    You are a clinical assistant. Answer based ONLY on the retrieved cases.
    If you are unsure, say "I don't know".

    Retrieved Cases:
    {retrieved_docs}

    Question: {query}
    Answer:
    """

    # Step 5: Generate response locally
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["Question:"]
    )

    return response['choices'][0]['text']

# Test the function
if __name__ == "__main__":
    print("✅ RAG system ready!")
    user_question = "What is the best antibiotic class for Klebsiella in urine infection?"
    answer = answer_query(user_question)
    print(f"Answer: {answer}")
