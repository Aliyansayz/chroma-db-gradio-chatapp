from chroma_db_vectorstore import * 
import os
from langchain.lms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.agents.toolkits import Chroma(
    create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo)



def main():

  text_splitter, embeddings = text_split_embeddings(model_name = 'inception-mbzuai/jais-13b' )
  client , collection  = get_pdf_embeddings( text_splitter, embeddings) 
  chroma_vectorstore = Chroma(
      client=client,
      collection_name="my_collection",
      embedding_function= embeddings)




tokenizer = AutoTokenizer.from_pretrained(model_path)
jais_llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

