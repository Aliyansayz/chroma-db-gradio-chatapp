import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


# Function to convert PDF to text
def pdf_to_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# create the open-source embedding function 'embeddings'
embeddings = HuggingFaceEmbeddings(model_name="inception-mbzuai/jais-13b")

# Initialize Chroma DB client
client = chromadb.PersistentClient(path="./db")
collection = client.create_collection(name="my_collection")

# Process each PDF in the ./input directory
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

