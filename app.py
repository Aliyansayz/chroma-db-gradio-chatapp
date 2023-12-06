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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "inception-mbzuai/jais-13b"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)


def get_response(text,tokenizer=tokenizer,model=model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=200-input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return response

# text= "عاصمة دولة الإمارات العربية المتحدة ه"
# print(get_response(text))

# text = "The capital of UAE is"
# print(get_response(text))



