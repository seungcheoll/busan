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
        # 생성자에서 Groq 클라이언트 설정
        self._client = Groq(api_key=self.groq_api_key)

    def _call(self, messages, **kwargs):
        # 채팅 메시지를 API 호출 형식으로 변환
        formatted = []
        for m in messages:
            if isinstance(m, HumanMessage):
                formatted.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                formatted.append({"role": "assistant", "content": m.content})
        # Groq API에 요청
        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted,
        )
        # 첫 번째 응답 반환
        return response.choices[0].message.content

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        # RetrievalQA에서 사용하는 generate 메서드 구현
        content = self._call(messages, **kwargs)
        return ChatResult(generations=[{"text": content, "message": AIMessage(content=content)}])

    @property
    def _llm_type(self):
        # LLM 타입 식별자
        return "groq-llama-4"

    @property
    def _identifying_params(self):
        # 모델 식별 파라미터
        return {"model": self.model}

# 시크릿에서 API 키 로딩 함수
def load_api_key():
    return st.secrets["general"]["API_KEY"]

# 템플릿 파일 로딩 함수
def load_template():
    with open("template.txt", "r", encoding="utf-8") as file:
        return file.read()

# QA 체인 초기화 (최초 1회만 수행하도록 캐싱)
@st.cache_resource
def init_qa_chain():
    api_key = load_api_key()  # API 키 가져오기
    template = load_template()  # 프롬프트 템플릿 불러오기
    # 임베딩 모델 설정
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    # 로컬 FAISS 벡터스토어 로드
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 검색 시 상위 5개 문서 반환
    # GroqLlamaChat 인스턴스화
    llm = GroqLlamaChat(groq_api_key=api_key)
    # 프롬프트 템플릿 적용
    prompt = PromptTemplate.from_template(template)
    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    # 기업 정보 데이터 로드
    company_df = pd.read_excel("map_busan.xlsx")
    # 전체 분포 HTML 로드
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()
    return qa_chain, company_df, map_html_content

# Streamlit 페이지 설정: 와이드 레이아웃, 기본 메뉴/헤더/푸터 숨김
st.set_page_config(page_title="부산 기업 RAG", layout="wide")
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 상단 여백 제거 및 커스텀 헤더 숨김 스타일
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# 사이드바 메뉴: JOB BUSAN vs Groq Chatbot 선택
menu = st.sidebar.radio("페이지 선택", ["📊 부산 기업 RAG 시스템", "💬 Groq Chatbot"], key="menu_select")
job_rag = menu == "📊 부산 기업 RAG 시스템"
chatbot = menu == "💬 Groq Chatbot"

# JOB BUSAN 페이지 흐름
if job_rag:
    st.title("🚢 부산 취업 상담 챗봇(JOB BUSAN)")
    # 최초 실행 시 QA 체인과 데이터 로드
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain, st.session_state.company_df, st.session_state.map_html = init_qa_chain()

    # 세션 상태 초기화
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "main_query" not in st.session_state:
        st.session_state["main_query"] = ""

    # 질문 입력 UI
    query = st.session_state["main_query"]
    query = st.text_input(
        "🎯 질문을 입력하세요:",
        value=query,
        key="query_input",
        placeholder="예: 연봉 3000만원 이상 선박 제조업 추천"
    )
    # 질문 실행 버튼
    if st.button("💬 질문 실행"):
        with st.spinner("🤖 JOB BUSAN이 부산 기업 정보를 검색 중입니다..."):
            result = st.session_state.qa_chain.invoke(query)
            st.session_state.gpt_result = result["result"]
            st.session_state.source_docs = result["source_documents"]
            st.session_state["main_query"] = ""
            st.rerun()
    else:
        st.session_state["main_query"] = query

    # 결과 보기용 탭 구성
    selected_tabs = st.tabs([
        "✅ JOB BUSAN의 답변",
        "📚 참고 문서",
        "🌍 관련 기업 위치",
        "🔍 부산 기업 분포 및 검색"
    ])

    # 1) GPT 응답 탭
    with selected_tabs[0]:
        st.write(st.session_state.get("gpt_result", "🔹 GPT 응답 결과가 여기에 표시됩니다."))

    # 2) 참고 문서 탭
    with selected_tabs[1]:
        source_docs = st.session_state.get("source_documents", [])
        for i, doc in enumerate(source_docs):
            with st.expander(f"문서 {i+1}"):
                st.write(doc.page_content)

    # 3) 관련 기업 위치 탭
    with selected_tabs[2]:
        docs = st.session_state.get("source_documents", [])
        company_names = [doc.metadata.get("company") for doc in docs if "company" in doc.metadata]
        matched_df = st.session_state.company_df[st.session_state.company_df["회사명"].isin(company_names)]
        if not matched_df.empty:
            # Folium 지도 생성 및 마커 추가
            m = folium.Map(location=[matched_df["위도"].mean(), matched_df["경도"].mean()], zoom_start=12)
            for _, row in matched_df.iterrows():
                folium.CircleMarker(
                    location=[row["위도"], row["경도"]],
                    radius=5,
                    color="blue", fill=True, fill_color="blue", fill_opacity=0.7,
                    popup=row["회사명"], tooltip=row["회사명"]
                ).add_to(m)
            html(m._repr_html_(), height=600)
        else:
            st.info("해당 기업 위치 정보가 없습니다.")

    # 4) 부산 기업 분포 및 검색 탭
    with selected_tabs[3]:
        # 검색 관련 상태 초기화
        if "search_keyword" not in st.session_state:
            st.session_state.search_keyword = ""
        if "reset_triggered" not in st.session_state:
            st.session_state.reset_triggered = False

        def reset_search():
            # 검색 초기화 함수
            st.session_state.search_keyword = ""
            st.session_state["search_input"] = ""
            st.session_state.reset_triggered = True

        # 검색 UI 레이아웃
        col1, col2 = st.columns([2, 1])
        with col1:
            search_input = st.text_input("", key="search_input", placeholder="🔎 회사명으로 검색 (예: 현대, 시스템, 조선 등)")
        with col2:
            if search_input:
                st.markdown("<div style='padding-top:27px;'></div>", unsafe_allow_html=True)
                st.button("검색 초기화", on_click=reset_search)

        # 검색 상태 업데이트 및 초기화 반영
        st.session_state.search_keyword = search_input
        if st.session_state.reset_triggered:
            st.session_state.reset_triggered = False
            st.rerun()

        # 검색 결과 필터링
        matched_df = pd.DataFrame()
        keyword = st.session_state.search_keyword.strip()
        if keyword:
            matched_df = st.session_state.company_df[
                st.session_state.company_df["회사명"].str.contains(keyword, case=False, na=False)
            ]

        # 결과 테이블 및 지도 렌더링
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("### 🧾 검색 기업 정보")
            if not matched_df.empty:
                # 그리드 설정
                PINLEFT = {'pinned': 'left'}
                PRECISION_TWO = {'type': ['numericColumn'], 'precision': 6}
                formatter = {
                    '회사명': ('회사명', PINLEFT),
                    '도로명': ('도로명', {'width': 200}),
                    '업종명': ('업종명', {'width': 150}),
                    '전화번호': ('전화번호', {'width': 120}),
                    '위도': ('위도', {**PRECISION_TWO, 'width': 100}),
                    '경도': ('경도', {**PRECISION_TWO, 'width': 100}),
                }
                gb = GridOptionsBuilder.from_dataframe(matched_df)
                for col, (header, opts) in formatter.items():
                    if col in matched_df.columns:
                        gb.configure_column(col, header_name=header, **opts)
                gb.configure_column('위도', hide=True)
                gb.configure_column('경도', hide=True)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_side_bar()
                gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
                gridOptions = gb.build()
                grid_response = AgGrid(
                    matched_df,
                    gridOptions=gridOptions,
                    data_return_mode=DataReturnMode.AS_INPUT,
                    update_mode=GridOptionsBuilder.MODEL_CHANGED,
                    fit_columns_on_grid_load=True,
                    theme='blue',
                    enable_enterprise_modules=True,
                    height=420,
                    width='100%',
                    allow_unsafe_jscode=True
                )
                # 선택된 행 처리
                sr = grid_response.get('selected_rows')
                selected = sr if isinstance(sr, list) else []
                st.session_state.selected_rows = selected
                if selected:
                    selected_df = pd.DataFrame(selected)[matched_df.columns]
            else:
                st.info("기업을 검색해주세요.")

        # 지도 렌더링
        with col1:
            selected = st.session_state.get('selected_rows', [])
            df_map = pd.DataFrame(selected) if selected else None
            if df_map is not None and not df_map.empty:
                m = folium.Map(location=[df_map['위도'].mean(), df_map['경도'].mean()], zoom_start=12)
                for _, row in df_map.iterrows():
                    folium.CircleMarker(
                        location=[row['위도'], row['경도']], radius=6,
                        color='green', fill=True, fill_color='green', fill_opacity=0.8,
                        popup=row['회사명'], tooltip=row['회사명']
                    ).add_to(m)
                html(m._repr_html_(), height=480)
                st.caption(f"✅ 선택된 기업 {len(df_map)}곳을 지도에 표시했습니다.")
            elif not matched_df.empty:
                m = folium.Map(location=[matched_df['위도'].mean(), matched_df['경도'].mean()], zoom_start=12)
                for _, row in matched_df.iterrows():
                    folium.CircleMarker(
                        location=[row['위도'], row['경도']], radius=5,
                        color='green', fill=True, fill_color='green', fill_opacity=0.7,
                        popup=row['회사명'], tooltip=row['회사명']
                    ).add_to(m)
                html(m._repr_html_(), height=480)
                st.caption(f"※ '{keyword}'를 포함한 기업 {len(matched_df)}곳을 지도에 표시했습니다.")
            elif keyword:
                st.warning("🛑 해당 기업이 존재하지 않습니다.")
            else:
                html(st.session_state.map_html, height=480)
                st.caption("※ 전체 기업 분포를 표시 중입니다.")

# Groq Chatbot 페이지 흐름
if chatbot:
    # GroqLlamaChat 인스턴스 초기화
    if "groq_chat" not in st.session_state:
        st.session_state.groq_chat = GroqLlamaChat(groq_api_key=load_api_key())
    # 세션 히스토리 초기화
    if "groq_history" not in st.session_state:
        st.session_state.groq_history = [
            {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
        ]

    # 챗봇 헤더 UI
    st.markdown("""
        <div style='background-color:#f9f9f9; padding:20px; border-radius:12px; border:1px solid #ddd; width:20%; margin: 0 auto; text-align: center;'>
            <h1 style='margin:0; font-size:24px;'>💬 Groq Chatbot</h1>
        </div>
    """, unsafe_allow_html=True)

    # 대화 내역 출력
    for msg in st.session_state.groq_history:
        if msg["role"] == "user":
            _, right = st.columns([3, 1])
            with right:
                st.markdown(
                    f"<div style='padding:12px; border-radius:8px; background-color:#e0f7fa; width:fit-content; margin-left:auto;'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
        else:
            left, _ = st.columns([1, 3])
            with left:
                bubble = st.chat_message("assistant")
                bubble.markdown(
                    f"<div style='background-color:#f0f0f0; padding:12px; border-radius:8px'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )

    # 사용자 입력
    prompt = st.chat_input("메시지를 입력하세요...", key="groq_input")
    if prompt:
        # 사용자 메시지 추가
        st.session_state.groq_history.append({"role": "user", "content": prompt})
        # 히스토리 메시지 객체 변환
        history = [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in st.session_state.groq_history
        ]
        # Groq API 호출 및 응답 처리
        answer = st.session_state.groq_chat._call(history)
        st.session_state.groq_history.append({"role": "assistant", "content": answer})
        st.rerun()
