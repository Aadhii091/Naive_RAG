from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from retreival_pipeline import retriever_func


queries = ["What was NVIDIA's first graphics accelerator called?",
    "Which company did NVIDIA acquire to enter the mobile processor market?",
    "What was Microsoft's first hardware product release?",
    "How much did Microsoft pay to acquire GitHub?",
    "In what year did Tesla begin production of the Roadster?",
    "Who succeeded Ze'ev Drori as CEO in October 2008?",
    "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?",
    "What was the original name of Microsoft before it became Microsoft?"]

# for query in queries:
#     relevant_docs = retriever_func(query)

#     combined_input = f"""Based on the following documents, please answer this question: {query}

#     Documents:
#     {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

#     Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
#     """
#     model = OllamaLLM(model='gemma3:12b')

#     messages = [
#         SystemMessage(content='You are a helpful assistant'),
#         HumanMessage(content=combined_input)
#     ]

#     result = model.invoke(messages)

#     # Display the full result and content only
#     print("\n--- Generated Response ---")
#     # print("Full result:")
#     # print(result)
#     print("Content only:")
#     print(result)
query = "When did Apple first release the iPhone?"
relevant_docs = retriever_func(query)

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
model = OllamaLLM(model='gemma3:12b')

messages = [
        SystemMessage(content='You are a helpful assistant'),
        HumanMessage(content=combined_input)     ]

result = model.invoke(messages)

print(result)