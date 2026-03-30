from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="""
        Write a beginner friendly article on {topic}
    """,
    input_variables=["topic"],
    validate_template=True
)

prompt2 = PromptTemplate(
    template="""
    Generate a professional LinkedIn post with a strong hook on below mentioned topic:
    \n topic: {topic}
    """,
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="""
    Merge the provided article and LinkedIn Post into a single document
    \n {article} and LinkedIn Post {post}
    """,
    input_variables=["article", "post"]
)

model = ChatOllama(
    model="gemini-3-flash-preview:cloud",
    base_url="https://ollama.com",
)

parallel_chain = RunnableParallel(
    {
        'article': prompt1 | model | StrOutputParser(),
        'post': prompt2 | model | StrOutputParser()
    }
)

merge_chain = prompt3 | model | StrOutputParser()

chain = parallel_chain | merge_chain

# print(chain.get_graph().draw_ascii())

for chunk in chain.stream({'topic': 'AI disruption in IT'}):
    print(chunk, end='', flush=True)
