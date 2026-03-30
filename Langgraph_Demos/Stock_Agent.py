from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from langgraph.types import interrupt, Command
import requests
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOllama(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com"
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def call_stock_api(symbol: str):
    """
    Gets the stock data from online API based on stock symbol
    """
    result = requests.get(
        f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=0OO7615N52SWYC0W')
    return result.json()


@tool
def make_payment():
    """
    Fake payment simulation for development environment. Makes payment for the required stock and quantity.
    """
    decision = interrupt("Do you want to proceed with the payment?")
    if decision.lower() == "yes":
        return "Payment successful!"
    else:
        return "Payment cancelled."


tools = [call_stock_api, make_payment]
model_with_tools = model.bind_tools(tools=tools)


def agent_node(state: AgentState):
    response = model_with_tools.invoke(state['messages'])
    return {'messages': [response]}


tool_node = ToolNode(tools)

graph = StateGraph(AgentState)

graph.add_node('agent_node', agent_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'agent_node')
graph.add_conditional_edges('agent_node', tools_condition)
graph.add_edge('tools', 'agent_node')

workflow = graph.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}

with open('stock_agent.png', 'wb') as f:
    f.write(workflow.get_graph().draw_mermaid_png())

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    result = workflow.invoke({'messages': [HumanMessage(
        content=user_input)]}, config=config)

    interrupt_response = result.get("__interrupt__", [])
    if interrupt_response:
        print("Interrupt message:", interrupt_response[0].value)
        user_decision = input("Enter your decision (yes/no): ").lower()
        result = workflow.invoke(Command(resume=user_decision), config=config)
    print(result['messages'][-1].content)
