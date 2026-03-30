from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOllama(model="gpt-oss:120b-cloud", base_url="https://ollama.com")


def Chat(state: ChatState) -> ChatState:
    response = llm.invoke(state['messages'])
    return {'messages': [response]}


graph = StateGraph(ChatState)
graph.add_node('Chat', Chat)
graph.add_edge(START, 'Chat')
graph.add_edge('Chat', END)

workflow = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    result = workflow.invoke(
        {'messages': [HumanMessage(content=user_input)]}, config=config)
    print(f"AI: {result['messages'][-1].content}")
