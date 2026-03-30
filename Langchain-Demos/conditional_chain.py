from pydantic import BaseModel, Field
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="gpt-oss:120b-cloud", base_url="https://ollama.com")


class Feedback(BaseModel):
    participant_name: str = Field(description="Name of the participant")
    summary: str = Field(description="Brief Summary of the overall feedback")
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(
        description="Sentitment of the feedback like positive, negative or neutral")
    highlights: list[str] = Field(
        description="List of positive hightlights of the program describe by the participant")
    lowlights: list[str] = Field(
        description="List of negative highlights of the program described by the participant")


pyparser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(template="Analyze the following feedback \n {feedback} \n {format_instructions}", input_variables=[
                        "feedback"], partial_variables={"format_instructions": pyparser.get_format_instructions()})


chain = prompt | llm | pyparser

positive_email_prompt = PromptTemplate(
    template="Write a positive email to the participant {participant_name} appreciating their feedback and thanking them for their time.", input_variables=["participant_name"])

negative_email_prompt = PromptTemplate(
    template="Write a negative email to the participant {participant_name} apologizing for the inconvenience and assuring them that their feedback will be taken into consideration.", input_variables=["participant_name"])

conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive",
     positive_email_prompt | llm | StrOutputParser()),
    (lambda x: x.sentiment == "Negative",
     negative_email_prompt | llm | StrOutputParser()),
    (lambda x: "Not able to analyze the sentiment")
)

chain_result = chain | conditional_chain

# result = chain_result.invoke({'feedback': "The Java Fullstack training program was poorly structured and failed to adequately cover essential modules like Core Java, Spring Boot, Hibernate, and Angular. The hands-on projects and live coding sessions were confusing and did not help in applying concepts to real-world scenarios at all. The trainer lacked the necessary knowledge and was unsupportive throughout the course, while the sessions on Git and deployment were incomplete and left me without a clear understanding of end-to-end development. The pace during the Spring Boot section was impossibly fast, and the total lack of practice time made it even more difficult to follow. Furthermore, the complete absence of sessions on debugging and code optimization severely hindered the learning experience. The front-end sessions on Angular were extremely rushed and lacked any helpful real-time examples to clarify the material. Out of 5 I would give a 1 rating for this program. Feedback given by Dhiraj Kumar.. Return only a valid JSON object with node additonal text or explanation. The JSON object should have the following structure: { participant_name: string, summary: string, sentiment: 'Positive' | 'Negative' | 'Neutral', highlights: string[], lowlights: string[] }"})

# print(result)
