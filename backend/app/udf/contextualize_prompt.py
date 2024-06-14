
def contextualize_prompt(resume_collection,query_embed,query):
  """ function to augment the prompt with the new knowledge retrieved from the vector database """
  
  # here is where you retrieve the info from the vector database
  results = resume_collection.query(query_embeddings=query_embed, n_results=10) #<----- here we are querying the vector db
                                                                                                                                           # (the n_results just because the default is 10 but we only have 3)
  # # get the text from the results
  source_knowledge = "\n\n".join(results['documents'][0])

  # #feed into an augmented prompt
  augmented_prompt = f"""Using the contexts below, answer the query.

  Context: {source_knowledge}

  Query: {query}
  """
  return augmented_prompt