from chroma_db_vectorstore import * 



def main():

  for filename in os.listdir('./input'):
    if filename.endswith('.pdf'):
        # Convert PDF to text
        text = pdf_to_text(os.path.join('./input', filename))

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Convert chunks to vector representations and store in Chroma DB
        documents_list = []
        embeddings_list = []
        ids_list = []
        
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            
            documents_list.append(chunk)
            embeddings_list.append(vector)
            ids_list.append(f"{filename}_{i}")
        
        
        collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list)


chroma_vectorstore = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function= embeddings)

tokenizer = AutoTokenizer.from_pretrained(model_path)
jais_llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

