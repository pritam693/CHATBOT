import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.udf.data_preprocess import read_files_from_folder
from app.udf.vector_database import create_database
from app.udf.contextualize_prompt import contextualize_prompt
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ['api_key'],
    api_type=os.environ['api_type'],
    api_version=os.environ['api_version'],
    model_name=os.environ["model_name"]
)

# Get current working directory
cwd = os.getcwd()
folder_path = os.path.join(cwd, "app/resume")
vector_db_path = os.path.join(cwd, "app/db")
file_data = read_files_from_folder(folder_path)
st.write(f"Number of resumes read: {len(file_data)}")
client = chromadb.PersistentClient(path=vector_db_path)

if len(os.listdir(vector_db_path)) == 0:
    resume_collection = create_database(file_data=file_data, emb_model=openai_ef, vector_db_path=vector_db_path)
    st.write(f"Number of resumes in the database: {len(file_data)}")
else:
    resume_collection = client.get_collection("resume_collection")
    

# Initialize the model
model = AzureChatOpenAI(
    openai_api_version=os.environ['api_version'],
    azure_deployment=os.environ["azure_chat_deployment"],
)

# Streamlit UI
st.title("Chat with AI Assistant")

user_input = st.text_area("Your message:")

if st.button("Send"):
    if user_input:
        # Process the message
        query_embed = openai_ef([user_input])
        prompt = HumanMessage(content=contextualize_prompt(resume_collection=resume_collection, query_embed=query_embed, query=user_input))
        messages = [
            SystemMessage(content="Please act like a helpful assistant."),
            HumanMessage(content=user_input),
            AIMessage(content="Sure, how can I help you?")
        ]
        messages.append(prompt)
        response = model(messages)

        if response:
            st.write(f"Response: {response.content}")
        else:
            st.write("Failed to generate response")
    else:
        st.write("Please enter a message")
