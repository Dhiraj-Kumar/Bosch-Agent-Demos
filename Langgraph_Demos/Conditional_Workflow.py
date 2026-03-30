from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOllama(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com"
)


class ScreeningModel(BaseModel):  # This is the model for Structured output
    candidate_name: str = Field("Name of the Candidate")
    job_title: str = Field("Job title mentioned in job description")
    candidate_experience: float = Field(
        "Working experience of the candidate as per the resume")
    experience_required: float = Field(
        "Working experience required for the job as per the job description")
    skill_match: float = Field(
        "Skill match score of the candidate. Value must be between 0 and 1", ge=0, le=1)
    email: str = Field("Email address of the candidate")


class ScreeningState(TypedDict):  # State for the graph
    candidate_name: str = Field("Name of the Candidate")
    job_title: str = Field("Job title mentioned in job description")
    candidate_experience: float = Field(
        "Working experience of the candidate as per the resume")
    experience_required: float = Field(
        "Working experience required for the job as per the job description")
    skill_match: float = Field(
        "Skill match score of the candidate. Value must be between 0 and 1", ge=0, le=1)
    resume_text: str = Field("Resume text passed as an input by the user")
    job_description: str = Field(
        "Job description text passed as an input by the user")
    email: str = Field("Email address of the candidate")


structured_model = model.with_structured_output(ScreeningModel)


def AnalyzeResumeWithJD(state: ScreeningState) -> ScreeningState:
    prompt = f"""Analyze the provided resume text and job description to extract the candidate name and total years of experience from resume, and extract the job title and required years of experience from the job description. Compare the candidate's skills with the job requirements and compute a skill_match score as a float value between 0.0 and 1.0, where 0.0 indicates no relevant skills matches the job description and 1.0 indicates strong alignment with most required skills, prioritizing skill relevance over job title. Return only a valid JSON object with no additional text or explanation, using exactly this format: {{"candidate_name": <string>, "job_title": <string>, "skill_match": <float>, "candidate_experience": <float>, "experience_required": <float>}}
    Resume text:
    {state['resume_text']}
    \n
    Job Description:
    {state['job_description']}
    """
    output = structured_model.invoke(prompt)
    print(output)
    return {'candidate_name': output.candidate_name, "skill_match": output.skill_match, "candidate_experience": output.candidate_experience, "experience_required": output.experience_required, "job_title": output.job_title}


def CheckCriteria(state: ScreeningState) -> Literal["ShortlistMail", "RejectionMail"]:
    if state['skill_match'] >= 0.6 and state['candidate_experience'] >= state['experience_required']:
        return "ShortlistMail"
    else:
        return "RejectionMail"


def ShortlistMail(state: ScreeningState) -> ScreeningState:
    prompt = f"Draft a mail to {state['candidate_name']} stating that his/her resume is shortlisted for the post of {state['job_title']}. Maintain a professional tone."
    result = model.invoke(prompt)
    return {'email': result.content}


def RejectionMail(state: ScreeningState) -> ScreeningState:
    prompt = f"Draft a mail to {state['candidate_name']} stating that his/her resume is rejected for the post of {state['job_title']}. Maintain a polite and professional tone."
    result = model.invoke(prompt)
    return {'email': result.content}


graph = StateGraph(ScreeningState)
graph.add_node('AnalyzeResumeWithJD', AnalyzeResumeWithJD)
graph.add_node('ShortlistMail', ShortlistMail)
graph.add_node('RejectionMail', RejectionMail)

graph.add_edge(START, 'AnalyzeResumeWithJD')
graph.add_conditional_edges('AnalyzeResumeWithJD', CheckCriteria)
graph.add_edge('ShortlistMail', END)
graph.add_edge('RejectionMail', END)

workflow = graph.compile()

with open('conditional.png', 'wb') as f:
    f.write(workflow.get_graph().draw_mermaid_png())

result = workflow.invoke({'resume_text': """
Candidate Name: Ankit Verma
Professional Summary

Customer Support and Operations professional with 4 years of experience in handling client communications, issue tracking, and process coordination. Skilled in customer relationship management, documentation, and basic data entry tasks. Seeking opportunities to transition into a technical role.

Work Experience
Customer Support Executive
BrightConnect Services, New Delhi

Responded to customer inquiries via email and chat
Logged and tracked issues using internal ticketing systems
Coordinated with technical teams to relay customer feedback
Prepared weekly support reports using spreadsheets
Assisted in onboarding new support staff

Operations Assistant
QuickServe Solutions

Managed daily operational checklists and documentation
Coordinated schedules and internal communications
Maintained records and generated basic reports
Supported process improvement initiatives

Skills
Customer communication and support
Issue tracking and ticket management
Microsoft Excel and Google Sheets
Documentation and reporting
Basic computer operations

Education
Bachelor of Arts in Business Administration
XYZ College
2016 – 2019   
""", 'job_description': """
Job Title: Software Engineer – Backend

Job Summary
We are looking for a Backend Software Engineer to design, develop, and maintain scalable server-side applications. The ideal candidate will work closely with frontend developers, product managers, and DevOps teams to deliver reliable and high-performance systems.

Key Responsibilities
Design and develop backend services and APIs
Write clean, maintainable, and efficient code
Optimize applications for performance and scalability
Integrate databases, third-party services, and APIs
Participate in code reviews and system design discussions
Troubleshoot and debug production issues

Required Skills
Strong proficiency in Python or Java
Experience with backend frameworks such as Django, FastAPI, or Spring Boot
Solid understanding of REST APIs and microservices architecture
Experience with SQL databases (PostgreSQL, MySQL)
Basic knowledge of Docker and containerization
Familiarity with Git and CI/CD pipelines

Experience & Qualifications
2–5 years of professional backend development experience
Bachelor’s degree in Computer Science or a related field (or equivalent practical experience)
Experience working in an Agile/Scrum environment
Understanding of system design and scalability concepts

Nice-to-Have Skills
Experience with cloud platforms (AWS, GCP, or Azure)
Knowledge of NoSQL databases (MongoDB, Redis)
Exposure to message queues (Kafka, RabbitMQ)                          
"""})

print(result)
