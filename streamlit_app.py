import streamlit as st  # Streamlit 라이브러리 임포트
import pandas as pd  # 데이터 처리용 pandas 임포트
import folium  # 지도 시각화를 위한 folium 임포트
from streamlit.components.v1 import html  # Streamlit에서 HTML 삽입을 위해 import
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode  # AgGrid를 통한 테이블 렌더링
from langchain_community.vectorstores import FAISS  # 벡터 DB로 FAISS 사용
from langchain.chains import RetrievalQA  # 질의응답 체인
from langchain.prompts import PromptTemplate  # 프롬프트 템플릿
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage  # LangChain 메시지 스키마
from langchain.chat_models.base import BaseChatModel  # 사용자 정의 LLM 인터페이스 상속
from langchain.embeddings import HuggingFaceEmbeddings  # 임베딩 모델
from langchain.schema import ChatResult  # 채팅 결과 타입
from groq import Groq  # Groq API 클라이언트

# GroqLlamaChat: BaseChatModel을 상속하여 Groq API 호출 구현
class GroqLlamaChat(BaseChatModel):
    groq_api_key: str  # Groq API 키
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # 사용할 LLM 모델
    _client: Groq = None  # Groq 클라이언트 초기화 변수

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
        return ChatResult(generations=[{"text": content, "message": AIMessage(content=content)}])

    @property
    def _llm_type(self):
        return "groq-llama-4"

    @property
    def _identifying_params(self):
        return {"model": self.model}

# 시크릿에서 API 키 로딩 함수
def load_api_key():
    return st.secrets["general"]["API_KEY"]

# 프롬프트 템플릿 로딩 함수
def load_template():
    with open("template.txt", "r", encoding="utf-8") as f:
        return f.read()

# QA 체인 및 자료 로딩 (캐시)
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
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html = f.read()
    return qa_chain, company_df, map_html

# 페이지 설정
st.set_page_config(page_title="부산 기업 RAG", layout="wide")
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top:0 !important;}
header[data-testid=\"stHeader\"] {display:none;}
</style>
""", unsafe_allow_html=True)

# 사이드바 메뉴
menu = st.sidebar.radio("페이지 선택", ["📊 부산 기업 RAG 시스템", "💬 Groq Chatbot"], key="menu_select")
job_rag = menu == "📊 부산 기업 RAG 시스템"
chatbot = menu == "💬 Groq Chatbot"

# JOB BUSAN 탭
if job_rag:
    st.title("🚢 부산 취업 상담 챗봇(JOB BUSAN)")
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain, st.session_state.company_df, st.session_state.map_html = init_qa_chain()
    if "main_query" not in st.session_state:
        st.session_state.main_query = ""

    query = st.text_input("🎯 질문을 입력하세요:", value=st.session_state.main_query, key="query_input")
    if st.button("💬 질문 실행"):
        with st.spinner("검색 중..."):
            result = st.session_state.qa_chain.invoke(query)
            st.session_state.gpt_result = result["result"]
            st.session_state.source_documents = result["source_documents"]
            st.session_state.main_query = ""
        st.rerun()
    else:
        st.session_state.main_query = query

    tabs = st.tabs(["✅ 답변","📚 참고 문서","🌍 위치","🔍 검색"])
    with tabs[0]: st.write(st.session_state.get("gpt_result","GPT 응답이 없습니다."))
    with tabs[1]:
        for i, d in enumerate(st.session_state.get("source_documents",[])):
            with st.expander(f"문서 {i+1}"): st.write(d.page_content)
    with tabs[2]:
        docs = st.session_state.get("source_documents",[])
        names = [d.metadata.get("company") for d in docs if "company" in d.metadata]
        df = st.session_state.company_df[st.session_state.company_df['회사명'].isin(names)]
        if not df.empty:
            m=folium.Map(location=[df['위도'].mean(),df['경도'].mean()],zoom_start=12)
            for _,r in df.iterrows(): folium.CircleMarker(location=[r['위도'],r['경도']],radius=5, color='blue',fill=True).add_to(m)
            html(m._repr_html_(),height=600)
        else: st.info("위치 정보 없음")
    with tabs[3]:
        kw=st.text_input("🔎 회사명 검색",key="search_input")
        df = st.session_state.company_df[df['회사명'].str.contains(kw,case=False,na=False)] if kw else st.session_state.company_df
        st.write(df)
        if kw: _=html(st.session_state.map_html,height=480)

# Groq Chatbot 탭
if chatbot:
    # 질문 실행 확인
    if not st.session_state.get("source_documents"):
        st.info("먼저 JOB BUSAN 탭에서 '질문 실행'을 눌러주세요.")
    else:
        # 시스템 메시지로 참고 문서 학습
        if not st.session_state.get("system_prompt"):  
            preload = "\n---\n".join([d.page_content for d in st.session_state.source_documents[:3]])
            st.session_state.groq_history = [{"role":"system","content":f"참고 문서 내용:\n{preload}"}]
            st.session_state.system_prompt = True
        # 챗봇 인스턴스 및 히스토리
        if "groq_chat" not in st.session_state:
            st.session_state.groq_chat = GroqLlamaChat(groq_api_key=load_api_key())
        if "groq_history" not in st.session_state:
            st.session_state.groq_history += [{"role":"assistant","content":"무엇을 도와드릴까요?"}]
                # UI 및 대화
        st.markdown("### 💬 Groq Chatbot")
        # 대화 내역 표시
        for msg in st.session_state.groq_history:
            if msg['role'] == 'user':
                cols = st.columns([3,1])
                with cols[1]:
                    st.write(msg['content'])
            elif msg['role'] == 'assistant':
                cols = st.columns([1,3])
                with cols[0]:
                    st.write(msg['content'])
        # 사용자 입력 처리
        user_input = st.text_input("메시지를 입력하세요...", key="groq_input")
        if user_input:
            st.session_state.groq_history.append({"role":"user","content":user_input})
            history = []
            for m in st.session_state.groq_history:
                if m['role'] == 'user':
                    history.append(HumanMessage(content=m['content']))
                elif m['role'] == 'assistant' or m['role'] == 'system':
                    history.append(AIMessage(content=m['content']))
            ans = st.session_state.groq_chat._call(history)
            st.session_state.groq_history.append({"role":"assistant","content":ans})
            st.rerun()
