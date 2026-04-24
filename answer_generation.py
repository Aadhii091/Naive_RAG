from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from retreival_pipeline import retriever_func


# chat_history = []
llm = OllamaLLM(model='gemma3:4b')

def ask_question(user_question, chat_history, vector_db):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        history_copy = chat_history.copy()
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + history_copy + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = llm.invoke(messages)
        search_question = result.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    relevant_docs = retriever_func(search_question, vector_db)

    combined_input = f"""
    You are a retrieval-based QA assistant. Your task is to answer the user's question strictly using the provided documents.

    ---------------------
    USER QUESTION:
    {search_question}
    ---------------------

    DOCUMENTS:
    {chr(10).join([f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in relevant_docs])}
    ---------------------

    INSTRUCTIONS:

    1. ONLY use information from the provided documents.
    2. DO NOT use prior knowledge.
    3. If the answer is not explicitly present, respond:
    "I don't have enough information to answer that question based on the provided documents."
    4. Keep the answer concise, precise, and directly relevant to the question.
    5. Do NOT include irrelevant details.

    CITATION RULES:

    - You MUST cite the page number(s) used.
    - If multiple pages are used, list them.

    OUTPUT FORMAT (strictly follow):

    Answer:
    <your answer here>

    Source:
    <Page number(s)>
    """

    messages = [
            SystemMessage(content='You are a helpful assistant'),
            HumanMessage(content=combined_input)     ]

    result = llm.invoke(messages)

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=result))

    print(result)
    return result

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\n Your question: ")

        if question.lower() == "quit":
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()
