from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ingestion_pipeline import load_documents

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

embedding_model = OllamaEmbeddings(model="embeddinggemma:300m")
llm = OllamaLLM(model="gemma3:4b", temperature=0)

documents = load_documents()

# Sematic chunking
def semantic_chunking(documents=None):

    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )

    chunks = semantic_splitter.split_documents(documents)

    print("SEMANTIC CHUNKING RESULTS:")
    print("=" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: ({len(chunk.page_content)} chars)")
        print(f'"{chunk.page_content}"')
        print()

    return chunks

# Create the prompt
def agentic_chunking():
    prompt = f"""
    You are a text chunking expert. Split this text into logical chunks.

    Rules:
    - Each chunk should be around 200 characters or less
    - Split at natural topic boundaries
    - Keep related information together
    - Put "<<<SPLIT>>>" between chunks

    Text:
    {documents}

    Return the text with <<<SPLIT>>> markers where you want to split:
    """

    # Get AI response
    print("🤖 Asking AI to chunk the text...")
    response = llm.invoke(prompt)
    marked_text = response

    # Split the text at the markers
    chunks = marked_text.split("<<<SPLIT>>>")

    # Clean up the chunks (remove extra whitespace)
    clean_chunks = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if cleaned:  # Only keep non-empty chunks
            clean_chunks.append(cleaned)

    # Show results
    print("\n🎯 AGENTIC CHUNKING RESULTS:")
    print("=" * 50)

    for i, chunk in enumerate(clean_chunks, 1):
        print(f"Chunk {i}: ({len(chunk)} chars)")
        print(f'"{chunk}"')
        print()

def main():
    while True:
        user = input("Enter your choice of chunking strategy: (eg: semantic and agentic), quit for exit : ")

        if user.lower() == "quit":
            break
        elif user.lower() == "semantic":
            semantic_chunking()
        elif user.lower() == "agentic":
            agentic_chunking()
        else:
            print("Invalid input")

if __name__ == "__main__":
    main()