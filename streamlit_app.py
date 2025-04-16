import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import ChatResult
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

    company_df = pd.read_excel("map_busan.xlsx")
    with open("전체기업_지도.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()

    return qa_chain, company_df, map_html_content

# ✅ 앱 시작
st.set_page_config(page_title="부산 기업 RAG", layout="wide")
st.title("🚢 부산 취업 상담 챗봇(JOB MAN)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.company_df, st.session_state.map_html = init_qa_chain()

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input("🎯 질문을 입력하세요:", value=st.session_state.query, key="main_query", placeholder="예) 신입 사원이 처음 받는 연봉 3000만원 이상 되는 선박 제조업 회사를 추천해줘")

if st.button("💬 질문 실행"):
    with st.spinner("🤖 JOB MAN이 부산 기업 정보를 검색 중입니다..."):
        result = st.session_state.qa_chain.invoke(query)
        st.session_state.gpt_result = result["result"]
        st.session_state.source_docs = result["source_documents"]
        st.session_state.query = ""
        st.rerun()tate["search_input"] = ""
        st.experimental_rerun()

    search_input = st.text_input(
        "🔍 회사명으로 검색 (예: 현대, 시스템, 조선 등)",
        value=st.session_state.search_keyword,
        key="search_input",
        placeholder="검색어 입력 후 엔터"
    )
    st.session_state.search_keyword = search_input

    if st.session_state.search_keyword:
        st.button("검색 초기화", on_click=reset_search)

    if st.session_state.search_keyword.strip():
        matched_df = st.session_state.company_df[
            st.session_state.company_df["회사명"].str.contains(st.session_state.search_keyword, case=False, na=False)
        ]
        if matched_df.empty:
            st.warning(f"'{st.session_state.search_keyword}'를 포함하는 기업이 없습니다.")
        else:
            m = folium.Map(
                location=[matched_df["위도"].mean(), matched_df["경도"].mean()],
                zoom_start=12,
                tiles="CartoDB positron"
            )
            for _, row in matched_df.iterrows():
                folium.CircleMarker(
                    location=[row["위도"], row["경도"]],
                    radius=5,
                    color="green",
                    fill=True,
                    fill_color="green",
                    fill_opacity=0.7,
                    popup=row["회사명"],
                    tooltip=row["회사명"]
                ).add_to(m)
            html(m._repr_html_(), height=600)
            st.caption(f"※ '{st.session_state.search_keyword}'를 포함한 기업 {len(matched_df)}곳을 지도에 표시했습니다.")
    else:
        html(st.session_state.map_html, height=600)
        st.caption("※ 입력 없이 전체 기업 분포를 확인 중입니다.")
