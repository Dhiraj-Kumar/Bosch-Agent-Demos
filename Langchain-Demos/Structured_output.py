from pydantic import BaseModel, Field
from typing import Literal
from langchain_ollama import ChatOllama
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


structured_model = llm.with_structured_output(Feedback)

feedback = structured_model.invoke(
    "The Java Fullstack training program was well-structured and covered essential modules like Core Java, Spring Boot, Hibernate, and Angular. The hands-on projects and live coding sessions made it easier to apply concepts in real-world scenarios. The trainer was knowledgeable and supportive, and the sessions on Git and deployment provided a complete view of end-to-end development. However, the pace during the Spring Boot section felt a bit fast, and more time for practice would have been helpful. Additionally, a dedicated session on debugging and code optimization could enhance the learning experience. Some front-end sessions, especially on Angular, felt rushed, and could benefit from more real-time examples. Out of 5 I would give 4 rating for this program. Feedback given by Dhiraj Kumar. Return only a valid JSON object with node additonal text or explanation. The JSON object should have the following structure: { participant_name: string, summary: string, sentiment: 'Positive' | 'Negative' | 'Neutral', highlights: string[], lowlights: string[] }")

print(feedback.sentiment)
