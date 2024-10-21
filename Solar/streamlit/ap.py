import streamlit as st
import tempfile as tf
import json
import os
import PyPDF2
import time
import torch
import transformers
from process import *  # Asegurando la correcta importación
from huggingface_hub import login
from langchain_groq import ChatGroq
#from grobid_client.grobid_client import GrobiClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_context(context):
    res = []
    for item in context:
        res.append(item.page_content)
    return res

def clean_gen(gen):
    res = {}
    for line in gen.split("\n"):
        if ":" in line:
            try:
                s, e = line.split(":")
                res[s.strip()] = e
            except:
                pass
    return res

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class SolarQA:
    def __init__(self, use_platform, user_key, llm_id, hf_key, llm_platform="LOCAL", temperature=0.1, sim_model_id="Salesforce/SFR-Embedding-Mistral", input_file_path=str(), context_file_path=""):
        self.use_platform = use_platform
        self.llm_id = llm_id
        self.user_key = user_key
        self.hf_key = hf_key
        self.llm_platform = llm_platform.lower()
        self.temperature = temperature
        self.sim_model_id = sim_model_id
        self.sys_prompt = """
        You are an assistant for extract information from context and selection the possible answer from the selection provided.
        You are given the extracted parts of a paper about solar chemistry and a question. Provide the extracted information and nothing else.
        """
        self.context_file_path = context_file_path
        self.input_file_path = input_file_path
        #self.prompt_file_pdf = prompt_file_pdf
        self.context_result = {
            "generation_model": self.llm_id,
            "similarity_model": self.sim_model_id,
            "similarity_metric": "Cosine_Similarity",
            "result": []
        }
        login(self.hf_key)
        self.get_text()
        self.get_vector()
        print("¡¡¡Vector Store Database is prepared!!!")
        self.get_llm()

    def get_text(self):
        title_list = ["Abstract", "Experimental", "Results and discussion"]
        if self.input_file_path[-3:] == "pdf":
            data = process_paper(self.input_file_path)
        else:
            with open(self.input_file_path, "rb") as f:
                data = json.load(f)
        self.context = ""
        for section in data:
            if section["title"] in title_list:
                self.context += section["title"]
                self.context += "\n"
                self.context += section["content"]
                self.context += "\n"           


    def get_llm(self):
        if self.use_platform:
            if self.llm_platform == "groq":
                os.environ["GROQ_API_KEY"] = self.user_key
                self.llm = ChatGroq(temperature=self.temperature, model_name=self.llm_id)
            else:
                raise ValueError('Unsupported Platform')
        else:
            try:
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.llm_id)
                self.llm = transformers.AutoModelForCausalLM.from_pretrained(
                    self.llm_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=bnb_config
                )
                self.terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            except:
                raise ValueError('Unsupported Platform')

    def get_vector(self):
        model_kwargs = {"device": "cpu"}
        self.sim_model = HuggingFaceEmbeddings(model_name=self.sim_model_id, model_kwargs=model_kwargs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
            length_function=len
        )
        chunks = text_splitter.split_text(text=self.context)
        #embeddings = self.sim_model.embed_text(chunks)
        embeddings = self.sim_model.embed_query(chunks)
        if len(embeddings) > 0 and len(embeddings[0]) > 0:
            self.vector_store = FAISS.from_texts(chunks, embedding=self.sim_model, normalize_L2=True, distance_strategy="COSINE")
        else:
            raise ValueError("La lista 'embeddings' está vacía o no contiene suficientes datos.")

    def search(self, query, k):
        embed_q = self.sim_model.embed_query(query)
        self.context = self.vector_store.similarity_search_with_score_by_vector(embed_q, k)

    def format_prompt(self, query, k):
        self.search(query, k)
        prompt = self.sys_prompt + "\n" + "Question:"
        prompt += query
        prompt += "\n"
        prompt += "Context:"
        for i in range(k):
            prompt += f"{self.context[i]}\n"
        return prompt

    def generation(self, query_data):
        res = ""
        if self.use_platform:
            for key, query in query_data.items():
                new_prompt = self.format_prompt(query, 5)
                messages = [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": new_prompt}]
                outputs = self.llm.invoke(messages)
                response = outputs.content
                temp_res = {
                    "question_category": key,
                    "query": query,
                    "generation": clean_gen(response),
                    "evidence": []
                }
                for i in range(len(self.context)):
                    context = self.context[i][0].page_content
                    sim_score = float(self.context[i][1])
                    temp_res["evidence"].append({"pdf_reference": context, "similarity_score": sim_score})
                self.context_result["result"].append(temp_res)
                res += response
                res += "\n"
            self.result = clean_gen(res)
        else:
            for key, query in query_data.items():
                new_prompt = self.format_prompt(query, 5)
                messages = [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": new_prompt}]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                outputs = self.llm.generate(
                    input_ids,
                    max_new_tokens=1024,
                    eos_token_id=self.terminators,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                )
                response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                temp_res = {
                    "question_category": key,
                    "query": query,
                    "generation": clean_gen(response),
                    "evidence": []
                }
                for i in range(len(self.context)):
                    context = self.context[i][0].page_content
                    sim_score = float(self.context[i][1])
                    temp_res["evidence"].append({"pdf_reference": context, "similarity_score": sim_score})
                self.context_result["result"].append(temp_res)
                res += response
                res += "\n"
            self.result = clean_gen(res)

    def save_context(self):
        with open(self.context_file_path, "w") as f:
            json.dump(self.context_result, f)
        print(f"RAG context is saved at: {self.context_file_path}")

# Process the uploaded PDF file
def process_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_json_automatically(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            query_data = json.load(f)
        st.write("Archivo JSON cargado automáticamente.")
        return query_data
    else:
        st.error(f"El archivo JSON no se encontró en la ruta: {json_path}")
        return None


# Main page (Upload and process PDF)
def main_page():
    st.image("/Users/alexandrafaje/Desktop/Solar/solar_chem/logo_pg.png", width=600)

    json_path = "/Users/alexandrafaje/Desktop/Solar/solar_chem/prompts.json"
    query_data = load_json_automatically(json_path)
    print(query_data)

    uploaded_pdf = st.file_uploader("", type="pdf")
    
    if uploaded_pdf is not None:
        print(1)
        if st.button("Submit"):
            print(2)
            with st.spinner('Analyzing your paper... Please be patient'):
                print(uploaded_pdf.name)
                temp_dir = tf.mkdtemp()
                pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
                #print(path)
                
                # Guardamos el archivo PDF temporalmente
                
                #file_path = f"./{uploaded_pdf.name}"  
                #with open(pdf_path, "wb") as f:
                #    f.write(uploaded_pdf.getvalue())
                #print(f.name)
                
                args_dict = {
                    "use_platform": str2bool("False"),
                    "user_key": "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",
                    "llm_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "hf_key": "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",
                    "llm_platform": "groq",
                    "sim_model_id": "Salesforce/SFR-Embedding-Mistral",
                    "input_file_path": uploaded_pdf.name,  # Ruta del archivo PDF subido
                   # "prompt_file_pdf": "/Users/alexandrafaje/Desktop/Solar/solar_chem/prompts.json",
                    "context_file_path": "./context_result.json"  # Ruta completa para guardar el archivo de salida
                }
                
                #prompt_file_pdf = "/Users/alexandrafaje/Desktop/Solar/solar_chem/prompts.json"
                start_time = time.time()
                solar = SolarQA(**args_dict)
                st.write(f"--- {time.time() - start_time} seconds for Data Preparation and Model Loading ---")
                temp_time = time.time()

                #with open(prompt_file_pdf, "rb") as f:
                #    query_data = json.load(f)
                
                solar.generation(query_data=query_data)
                st.write("--- Model generation time consumption: %s seconds ---" % (time.time() - temp_time))
                solar.save_context()
                st.write("Analysis completed. Here are the results:")

                # Display results in expandable boxes
                with st.expander("Catalyst: TiO2"):
                    st.write("Paragraph 1: This is the extracted text about the catalyst TiO2.")
                    st.write("Paragraph 2: Additional information about the catalyst...")
                
                # Otros bloques...
def main():
    query_params = st.experimental_get_query_params()  # Replaces the experimental function
    main_page()



if __name__ == "__main__":
    main()
