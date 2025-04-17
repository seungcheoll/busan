import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import ChatResult
from groq import Groq

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
        return ChatResult(generations=[{"text": content, "message": AIMessage(content=content)}])

    @property
    def _llm_type(self):
        return "groq-llama-4"

    @property
    def _identifying_params(self):
        return {"model": self.model}

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
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()
    return qa_chain, company_df, map_html_content

st.set_page_config(page_title="부산 기업 RAG", layout="wide")
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 상단 여백 제거 스타일
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

menu = st.sidebar.radio("페이지 선택", ["📊 부산 기업 RAG 시스템", "💬 Groq Chatbot"], key="menu_select")
job_rag = menu == "📊 부산 기업 RAG 시스템"
chatbot = menu == "💬 Groq Chatbot"


if job_rag:
    st.title("🚢 부산 취업 상담 챗봇(JOB BUSAN)")
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain, st.session_state.company_df, st.session_state.map_html = init_qa_chain()

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "main_query" not in st.session_state:
        st.session_state["main_query"] = ""

    query = st.session_state["main_query"]

    query = st.text_input(
        "🎯 질문을 입력하세요:",
        value=query,
        key="query_input",
        placeholder="예: 연봉 3000만원 이상 선박 제조업 추천"
    )

    if st.button("💬 질문 실행"):
        with st.spinner("🤖 JOB BUSAN이 부산 기업 정보를 검색 중입니다..."):
            result = st.session_state.qa_chain.invoke(query)
            st.session_state.gpt_result = result["result"]
            st.session_state.source_docs = result["source_documents"]
            st.session_state["main_query"] = ""
            st.rerun()
    else:
        st.session_state["main_query"] = query

    selected_tabs = st.tabs([
        "✅ JOB BUSAN의 답변",
        "📚 참고 문서",
        "🌍 관련 기업 위치",
        "🔍 부산 기업 분포 및 검색"
    ])

    with selected_tabs[0]:
        st.write(st.session_state.get("gpt_result", "🔹 GPT 응답 결과가 여기에 표시됩니다."))

    with selected_tabs[1]:
        source_docs = st.session_state.get("source_docs", [])
        for i, doc in enumerate(source_docs):
            with st.expander(f"문서 {i+1}"):
                st.write(doc.page_content)

    with selected_tabs[2]:
        docs = st.session_state.get("source_docs", [])
        company_names = [doc.metadata.get("company") for doc in docs if "company" in doc.metadata]
        matched_df = st.session_state.company_df[st.session_state.company_df['회사명'].isin(company_names)]
        if not matched_df.empty:
            m = folium.Map(location=[matched_df["위도"].mean(), matched_df["경도"].mean()], zoom_start=12)
            for _, row in matched_df.iterrows():
                folium.CircleMarker(
                    location=[row["위도"], row["경도"]],
                    radius=5,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.7,
                    popup=row["회사명"],
                    tooltip=row["회사명"]
                ).add_to(m)
            html(m._repr_html_(), height=600)
        else:
            st.info("해당 기업 위치 정보가 없습니다.")

    with selected_tabs[3]:
        if "search_keyword" not in st.session_state:
            st.session_state.search_keyword = ""
        if "reset_triggered" not in st.session_state:
            st.session_state.reset_triggered = False

        def reset_search():
            st.session_state.search_keyword = ""
            st.session_state["search_input"] = ""
            st.session_state.reset_triggered = True

        col1, col2 = st.columns([2, 1])
        with col1:
            search_input = st.text_input("", key="search_input", placeholder="🔎 회사명으로 검색 (예: 현대, 시스템, 조선 등)")
        with col2:
            if search_input:
                st.markdown("<div style='padding-top:27px;'></div>", unsafe_allow_html=True)
                st.button("검색 초기화", on_click=reset_search)

        st.session_state.search_keyword = search_input

        if st.session_state.reset_triggered:
            st.session_state.reset_triggered = False
            st.rerun()

        matched_df = pd.DataFrame()
        keyword = st.session_state.search_keyword.strip()
        if keyword:
            matched_df = st.session_state.company_df[
                st.session_state.company_df["회사명"].str.contains(keyword, case=False, na=False)
            ]

        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("### 🧾 검색 기업 정보")
            if not matched_df.empty:
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
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    fit_columns_on_grid_load=True,
                    theme='blue',
                    enable_enterprise_modules=True,
                    height=420,
                    width='100%',
                    allow_unsafe_jscode=True
                )

                sr = grid_response.get('selected_rows')
                if sr is None:
                    selected = []
                elif isinstance(sr, pd.DataFrame):
                    selected = sr.to_dict('records')
                elif isinstance(sr, list):
                    selected = sr
                else:
                    selected = []

                st.session_state.selected_rows = selected

                if selected:
                    selected_df = pd.DataFrame(selected)[matched_df.columns]
            else:
                st.info("기업을 검색해주세요.")

        with col1:
            selected = st.session_state.get('selected_rows', [])
            if selected:
                df_map = pd.DataFrame(selected)
                m = folium.Map(location=[df_map['위도'].mean(), df_map['경도'].mean()], zoom_start=12)
                for _, row in df_map.iterrows():
                    folium.CircleMarker(
                        location=[row['위도'], row['경도']],
                        radius=6,
                        color='green',
                        fill=True,
                        fill_color='green',
                        fill_opacity=0.8,
                        popup=row['회사명'],
                        tooltip=row['회사명']
                    ).add_to(m)
                html(m._repr_html_(), height=480)
                st.caption(f"✅ 선택된 기업 {len(df_map)}곳을 지도에 표시했습니다.")
            elif not matched_df.empty:
                m = folium.Map(location=[matched_df['위도'].mean(), matched_df['경도'].mean()], zoom_start=12)
                for _, row in matched_df.iterrows():
                    folium.CircleMarker(
                        location=[row['위도'], row['경도']],
                        radius=5,
                        color='green',
                        fill=True,
                        fill_color='green',
                        fill_opacity=0.7,
                        popup=row['회사명'],
                        tooltip=row['회사명']
                    ).add_to(m)
                html(m._repr_html_(), height=480)
                st.caption(f"※ '{keyword}'를 포함한 기업 {len(matched_df)}곳을 지도에 표시했습니다.")
            elif keyword:
                st.warning("🛑 해당 기업이 존재하지 않습니다.")
            else:
                html(st.session_state.map_html, height=480)
                st.caption("※ 전체 기업 분포를 표시 중입니다.")


if chatbot:
    # ✅ 좁은 영역을 위한 스타일 삽입
    st.markdown("""
        <style>
        .chat-wrapper {
            max-width: 720px;
            margin: 0 auto;
            padding: 1rem;
        }
        </style>
        <div class='chat-wrapper'>
    """, unsafe_allow_html=True)

    # ✅ 챗봇 상단 타이틀
    st.markdown("""
        <div style='background-color:#f9f9f9; padding:20px; border-radius:12px; border:1px solid #ddd; text-align: center;'>
            <h1 style='margin:0; font-size:24px;'>💬 Groq Chatbot</h1>
        </div>
    """, unsafe_allow_html=True)

    # ✅ 메시지 출력
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

    # ✅ 입력창
    prompt = st.chat_input("메시지를 입력하세요...", key="groq_input")
    if prompt:
        st.session_state.groq_history.append({"role": "user", "content": prompt})
        history = [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in st.session_state.groq_history
        ]
        answer = st.session_state.groq_chat._call(history)
        st.session_state.groq_history.append({"role": "assistant", "content": answer})
        st.rerun()

    # ✅ div 닫기
    st.markdown("</div>", unsafe_allow_html=True)
