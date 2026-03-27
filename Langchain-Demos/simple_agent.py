from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.tools import tool
import requests

load_dotenv()

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com"
)

DB_URI = "postgresql://postgres:niit1234@localhost:5432/conversationdb?sslmode=disable"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    api_key = "0466cbde7c464dd7f56717dc5a926737"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Could not retrieve weather data for {city}."


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=llm,
        system_prompt="You are a helpful AI assistant who gives answer with humour. Use emojis while drafting your responses.",
        checkpointer=checkpointer,
        tools=[get_weather]
    )

    while (True):
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        response = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        }, {"configurable": {"thread_id": "1"}})
        print(f"AI: {response['messages'][-1].content}")
