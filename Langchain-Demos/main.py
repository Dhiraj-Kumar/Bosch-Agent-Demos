from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="gpt-oss:120b-cloud", base_url="https://ollama.com")

for chunk in llm.stream("What is the meaning of life?"):
    print(chunk.content, end="", flush=True)
