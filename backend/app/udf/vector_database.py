# import chromadb and create a client
import chromadb

def create_database(file_data,emb_model,vector_db_path):
    #client = chromadb.Client()
    client = chromadb.PersistentClient(path=vector_db_path)

    # try:
    #     client.delete_collection("resume_collection")
    # except:
        # get the data from file_data and create chromadb collection
    documents = []
    metadatas = []
    ids = []

    for index, data in enumerate(file_data):
        documents.append(data['content'])
        metadatas.append({'source': data['file_name']})
        ids.append(str(index + 1))
    
    #Create a collection in ChromaDB
    resume_collection =  client.get_or_create_collection(name="resume_collection", embedding_function=emb_model)

    # Add files to the ChromaDB collection
    resume_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    
    )
    
    # augmented_prompt=contextualize_prompt(resume_collection)
    return resume_collection
