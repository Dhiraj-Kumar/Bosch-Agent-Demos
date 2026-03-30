from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="gpt-oss:120b-cloud", base_url="https://ollama.com")

prompt = PromptTemplate(
    template="Write an article on the below topic: \n{topic}", input_variables=["topic"])

prompt2 = PromptTemplate(
    template="Create 5 MCQ type questions based on the following article: \n{article}", input_variables=["article"])

chain = prompt | llm | prompt2 | llm | StrOutputParser()

# print(chain.get_graph().draw_ascii())

# result = chain.invoke({"topic": "Artificial Intelligence"})
# print(result)

for chunk in chain.stream({"topic": "Artificial Intelligence"}):
    print(chunk, end="", flush=True)
