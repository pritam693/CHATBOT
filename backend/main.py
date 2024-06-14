import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from app.udf.data_preprocess import read_files_from_folder
from app.udf.vector_database import create_database
from app.udf.contextualize_prompt import contextualize_prompt
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI()

# Set CORS policy
origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://localhost:5173"  # Assuming your Angular app runs on port 4200
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    #allow_origins = ["http://localhost:5173/"]  # Allows requests from specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ['api_key'],
    api_type=os.environ['api_type'],
    api_version=os.environ['api_version'],
    model_name=os.environ["model_name"]
)

# Get current working directory
cwd = os.getcwd()
# Read files from a folder
folder_path = os.path.join(cwd, "app/resume")
vector_db_path = os.path.join(cwd, "app/db")
file_data = read_files_from_folder(folder_path)
client = chromadb.PersistentClient(path=vector_db_path)

if len(os.listdir(vector_db_path)) == 0:
    resume_collection = create_database(file_data=file_data, emb_model=openai_ef, vector_db_path=vector_db_path)
else:
    resume_collection = client.get_collection("resume_collection")

# Initialize the model
model = AzureChatOpenAI(
    openai_api_version=os.environ['api_version'],
    azure_deployment=os.environ["azure_chat_deployment"],
)

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    # Initialize Langchain messages
    messages = [
        SystemMessage(content="Please act like a helpful assistant."),
        HumanMessage(content="Hi, can you help me out with a task?"),
        AIMessage(content="Sure, how can I help you?")
    ]

    query_embed = openai_ef([query.message])
    prompt = HumanMessage(content=contextualize_prompt(resume_collection=resume_collection, query_embed=query_embed, query=query.message))
    messages.append(prompt)
    response = model(messages)

    if not response:
        raise HTTPException(status_code=500, detail="Failed to generate response")

    return {"response": response.content}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
