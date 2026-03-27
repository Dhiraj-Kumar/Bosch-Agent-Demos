from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

# Loading the documents
loader = PyPDFLoader("./Documents/HR-Policy.pdf")
documents = loader.load()

# Splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Creating the embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vectoreStore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# Creating the retriever
retriever = vectoreStore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)

# Augmentation prompt template
prompt = PromptTemplate(
    template="""you are a helpful assistant. Answer the question using only the context below
    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)

llm = ChatOllama(model="gpt-oss:120b-cloud", base_url="https://ollama.com")

# Create a RAG pipeline Chain
rag_chain = (RunnableParallel(context=retriever, question=RunnablePassthrough(
)) | prompt | llm | StrOutputParser())

while (True):
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    response = rag_chain.invoke(user_input)
    print(f"AI: {response}")
