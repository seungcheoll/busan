# components/qa_utils.py
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from components.llm import GroqLlamaChat

# 🔑 API Key 불러오기
def load_api_key():
    return st.secrets["general"]["API_KEY"]

# 🧩 사용자 유형별 템플릿 불러오기
def load_all_templates():
    templates = {
        "대학생": open("template/template_un.txt", "r", encoding="utf-8").read(),
        "첫 취업 준비": open("template/template_first.txt", "r", encoding="utf-8").read(),
        "이직 준비": open("template/template_move.txt", "r", encoding="utf-8").read(),
    }
    return templates

# 🧠 벡터 DB 및 QA 체인 초기화 함수
@st.cache_resource
def init_qa_chain(api_key):
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = GroqLlamaChat(groq_api_key=api_key)
    
    company_df = pd.read_excel("map_busan.xlsx")
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()

    return llm, retriever, company_df, map_html_content
