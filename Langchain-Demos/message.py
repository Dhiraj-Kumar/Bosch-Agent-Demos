from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="gpt-oss:120b-cloud", base_url="https://ollama.com")

messages = [
    SystemMessage(
        content="You are a helpful AI assistant who gives answer with humour. Use emojis while drafting your responses.")
]

while (True):
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    print(f"AI: {response.content}")
    messages.append(AIMessage(content=response.content))

print(messages)
