from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persist_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model='embeddinggemma:300m')

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# def retriever_func(query):
#     retriever = db.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 5,
#             "score_threshold": 0.3
#         }
#     )

def retriever_func(query, db):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs = {
            "k": 5,
            "fetch_k": 16,
            "lambda_mult": 0.5 
        }
    )

    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    # Display results
    print("--- Context ---")

    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
    
    return relevant_docs

def main():

    queries = ["What was NVIDIA's first graphics accelerator called?",
    "Which company did NVIDIA acquire to enter the mobile processor market?",
    "What was Microsoft's first hardware product release?",
    "How much did Microsoft pay to acquire GitHub?",
    "In what year did Tesla begin production of the Roadster?",
    "Who succeeded Ze'ev Drori as CEO in October 2008?",
    "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?",
    "What was the original name of Microsoft before it became Microsoft?"]

    for query in queries:
        retriever_func(query)

if __name__ == "__main__":
    main()