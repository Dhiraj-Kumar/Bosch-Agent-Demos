import operator
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


class PostState(TypedDict):
    # Standard message history
    messages: Annotated[List[BaseMessage], operator.add]
    # Keep track of how many times we've polished
    revision_count: int


llm = ChatOllama(
    model="gemini-3-flash-preview:cloud",
    base_url="https://ollama.com",
)


def writer_node(state: PostState):
    """Generates the LinkedIn post or revises it based on feedback."""
    instructions = (
        "You are a viral LinkedIn content creator. Write a post based on the topic provided. "
        "Use a strong hook, plenty of white space, and an engaging CTA. "
        "If you see a critique, rewrite the post to address the specific feedback."
    )
    count = state.get("revision_count", 0)
    messages = [HumanMessage(content=instructions)] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response], "revision_count": count + 1}


def critic_node(state: PostState):
    """Critiques the post for virality and engagement."""
    last_post = state["messages"][-1].content

    # We ask the critic to decide if it's 'GOOD' or needs work
    prompt = f"""Review this LinkedIn post:
    1. Is there a 'hook' in the first 2 lines?
    2. Is it easy to read on mobile (short sentences)?
    3. Is there a question at the end to drive comments?
    4. Is it creating curiousity among readers?
    5. Is there a slight humour in the post?
    
    If the post is perfect, respond ONLY with 'READY'.
    Otherwise, provide a bulleted list of what to fix.
    
    Post: {last_post}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


def route_post(state: PostState) -> Literal["writer", "end"]:
    """The Conditional Edge."""
    last_feedback = state["messages"][-1].content

    # Stop if the critic is happy or if we've tried 2 times already
    if "READY" in last_feedback.upper() or state["revision_count"] >= 5:
        return "end"

    return "writer"


builder = StateGraph(PostState)

builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)


# builder.set_entry_point("writer")
builder.add_edge(START, "writer")
builder.add_edge("writer", "critic")

builder.add_conditional_edges(
    "critic",
    route_post,
    {
        "writer": "writer",
        "end": END
    }
)

graph = builder.compile(checkpointer=InMemorySaver())
image_data = graph.get_graph().draw_mermaid_png()
with open("reflection_graph.png", mode="wb") as f:
    f.write(image_data)

topic = {"messages": [HumanMessage(
    content="Write a post about why AI agents are the future of software.")]}

for message_chunk, metadata in graph.stream(topic, stream_mode="messages", config={"configurable": {"thread_id": "1"}}):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
