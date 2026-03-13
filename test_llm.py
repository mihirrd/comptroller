from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")

print(llm.invoke("Hello"))
