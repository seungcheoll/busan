import streamlit as st
st.set_page_config(page_title="부산 기업 RAG", layout="wide")

import os
import pandas as pd
import folium
from streamlit.components.v1 import html

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document, ChatResult
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from groq import Groq

# ✅ 커스텀 ChatModel 클래스
class GroqLlamaChat(BaseChatModel):
    groq_api_key: str
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    _client: Groq = None

    def __init__(self, **data):
        super().__init__(**data)
        self._client = Groq(api_key=self.groq_api_key)

    def _call(self, messages, **kwargs):
        formatted = []
        for m in messages:
            if isinstance(m, HumanMessage):
                formatted.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                formatted.append({"role": "assistant", "content": m.content})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted,
        )
        return response.choices[0].message.content

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        content = self._call(messages, **kwargs)
        return ChatResult(
            generations=[{"text": content, "message": AIMessage(content=content)}]
        )

    @property
    def _llm_type(self):
        return "groq-llama-4"

    @property
    def _identifying_params(self):
        return {"model": self.model}

# ✅ 텍스트 파일 로딩 함수
def load_api_key():
    return st.secrets["general"]["API_KEY"]

def load_template():
    with open("template.txt", "r", encoding="utf-8") as file:
        return file.read()

@st.cache_resource
def init_qa_chain():
    api_key = load_api_key()
    template = load_template()

    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = GroqLlamaChat(groq_api_key=api_key)
    prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # ✅ 기업 위치정보 로딩
    company_df = pd.read_excel("부산기업정보_위도경도포함.xlsx")

    # ✅ 전체 지도 HTML 파일 미리 읽어오기
    with open("전체기업_지도.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()

    return qa_chain, company_df, map_html_content

# ✅ 세션 상태에 QA 체인과 위치정보 저장
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.company_df, st.session_state.map_html = init_qa_chain()

# ✅ UI 구성
st.title("🚢 부산 취업 상담 챗봇(JOB MAN)")

query = st.text_input("🎯 질문을 입력하세요:", placeholder="예) 신입 사원이 처음 받는 연봉 3000만원 이상 되는 선박 제조업 회사를 추천해줘")

# ✅ 버튼 클릭 시, 체인 실행
if st.button("💬 질문 실행") and query:
    with st.spinner("🤖 JOB MAN이 부산 기업 정보를 검색 중입니다..."):
        result = st.session_state.qa_chain.invoke(query)
        # ✅ 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "✅ JOB MAN의 답변",
            "📚 참고 문서",
            "🗺 관련 기업 위치",
            "📍 부산 기업 분포"
        ])
        
        # ✅ 탭 1: GPT 답변
        with tab1:
            st.write(result["result"])
        
        # ✅ 탭 2: 참고 문서
        with tab2:
            for i, doc in enumerate(result["source_documents"]):
                with st.expander(f"문서 {i+1}"):
                    st.write(doc.page_content)
        
        # ✅ 탭 3: 기업 위치
        with tab3:
            source_docs = result["source_documents"]
            company_names = [doc.metadata.get("company") for doc in source_docs if "company" in doc.metadata]
            matched_df = st.session_state.company_df[st.session_state.company_df['회사명'].isin(company_names)]
        
            if not matched_df.empty:
                m = folium.Map(
                    location=[matched_df["위도"].mean(), matched_df["경도"].mean()],
                    zoom_start=12
                )
                for _, row in matched_df.iterrows():
                    folium.Marker(
                        [row["위도"], row["경도"]],
                        tooltip=row["회사명"],
                        popup=row["회사명"]
                    ).add_to(m)
                html(m._repr_html_(), height=500)
            else:
                st.info("해당 기업 위치 정보가 없습니다.")
        
        # ✅ 탭 4: 부산 기업 분포 (바로 내장 렌더링)
        with tab4:
            st.markdown("### 🗺 부산 전체 기업 분포 지도")
        
            # 이미 세션에 저장된 HTML 내용 바로 렌더링
            html(st.session_state.map_html, height=600)
        
            st.caption("※ 지도는 전체 기업 위치를 시각화한 결과입니다.")
