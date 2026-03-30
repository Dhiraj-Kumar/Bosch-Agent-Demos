import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.title("Chatbot Example")

# user_input = st.chat_input("Type your message here...")
# if user_input:
#     model = ChatOllama(
#         model="gpt-oss:120b-cloud",
#         base_url="https://ollama.com"
#     )
#     response = model.invoke(user_input)
#     st.write(response.content)

messages = [
    SystemMessage(content="You are a funny and helpful assistant."),
]

user_input = st.text_input("Type your message here...")
if st.button("Send"):
    model = ChatOllama(
        model="gpt-oss:120b-cloud",
        base_url="https://ollama.com"
    )
    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    st.write(response.content)
