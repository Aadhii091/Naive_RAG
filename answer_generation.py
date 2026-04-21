from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from retreival_pipeline import retriever_func


chat_history = []
model = OllamaLLM(model='gemma3:12b')

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    relevant_docs = retriever_func(search_question)

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {chr(10).join([f" Document {i} : {doc.metadata}, - {doc.page_content}" for i, doc in enumerate(relevant_docs, 1)])}
    Please provide a clear, precise answer and you are constrained to answer exactly what was asked using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    you are constrained to answer in this structure : 
    Query : the query user had asked.
    Answer : answer of you. 
    Document No : no of the document you are preferring.
    source : the document id you prefer for answering.                
    """

    messages = [
            SystemMessage(content='You are a helpful assistant'),
            HumanMessage(content=combined_input)     ]

    result = model.invoke(messages)

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=result))

    print(result)

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\n Your question: ")

        if question.lower() == "quit":
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()
