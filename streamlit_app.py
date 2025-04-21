# ───────────────────────────────────────────
# [1] 라이브러리 임포트 및 GroqLlamaChat 클래스 정의
# ───────────────────────────────────────────
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import folium
from streamlit.components.v1 import html
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import ChatResult
from groq import Groq
# ✅ Groq API를 활용한 LangChain용 LLM 클래스 정의
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
        
# ───────────────────────────────────────────
# [2] API 키 및 사용자 템플릿 로딩 함수
# ───────────────────────────────────────────
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

# ───────────────────────────────────────────
# [3] 벡터 DB 및 QA 체인 초기화 함수
# ───────────────────────────────────────────
# 🧠 벡터 DB 및 QA 체인 초기화
@st.cache_resource
def init_qa_chain():
    api_key = load_api_key()
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = GroqLlamaChat(groq_api_key=api_key)

    company_df = pd.read_excel("map_busan.xlsx")
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()

    return llm, retriever, company_df, map_html_content

# ───────────────────────────────────────────
# [4] Streamlit 기본 설정 및 스타일 커스터마이징
# ───────────────────────────────────────────
# 🧭 Streamlit 기본 설정 및 스타일 숨기기
st.set_page_config(
    page_title="JobBusan",
    page_icon="https://raw.githubusercontent.com/seungcheoll/busan/main/image/chatbot.png",
    layout="wide"
)

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 📏 상단 여백 제거 스타일
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

# ───────────────────────────────────────────
# [5] 사용자 입력값 세션 초기화
# ───────────────────────────────────────────
for key in ["university", "major", "gpa", "field_pref", "job_pref", "activities", "certificates"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ───────────────────────────────────────────
# [6] 사이드바: 메뉴 선택 및 사용자 프로필 입력
# ───────────────────────────────────────────
# 🔘 사이드바 라디오 메뉴 설정
with st.sidebar:
    # 🔘 페이지 선택 메뉴
    choice = option_menu(
        menu_title="Page",
        options=["Guide","Job-Bu", "Job-Bu Chatbot"],
        icons=["info-circle","", ""],
        menu_icon="",
        default_index=0,
        styles={
            "container": {
                "padding": "4!important",
                "background-color": "transparent"
            },
            "icon": {"display": "none"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#e5e9f2"
            },
            "nav-link-selected": {
                "background-color": "#3498db",
                "color": "white"
            },
        }
    )

    # ▼ 사용자 프로필 입력 (expander로 접기/펼치기)
    with st.expander("📋 사용자 프로필 입력", expanded=False):
        with st.form("profile_form"):
            university_temp   = st.text_input("대학교", value=st.session_state.get("university", ""))
            major_temp        = st.text_input("전공", value=st.session_state.get("major", ""))
            gpa_temp          = st.text_input("학점", value=st.session_state.get("gpa", ""))
            field_pref_temp   = st.text_input("선호분야(산업군)", value=st.session_state.get("field_pref", ""))
            job_pref_temp     = st.text_input("선호직무", value=st.session_state.get("job_pref", ""))
            activities_temp   = st.text_area("경력사항", value=st.session_state.get("activities", ""))
            certificates_temp = st.text_area("보유 자격증", value=st.session_state.get("certificates", ""))

            submitted = st.form_submit_button("입력 완료")
            if submitted:
                st.session_state.university   = university_temp
                st.session_state.major        = major_temp
                st.session_state.gpa          = gpa_temp
                st.session_state.field_pref   = field_pref_temp
                st.session_state.job_pref     = job_pref_temp
                st.session_state.activities   = activities_temp
                st.session_state.certificates = certificates_temp

                st.success("✅ 입력 완료!")
info = choice == "Guide"
job_rag = choice == "Job-Bu"
chatbot = choice == "Job-Bu Chatbot"

# ───────────────────────────────────────────
# [7] 이용 가이드 페이지
# ───────────────────────────────────────────
if info:
    st.markdown("<h1 style='text-align: center;'>🧾 JobBusan 이용 가이드</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
      .gbox {
        background-color: #f0f4f8;
        padding: 30px;
        border: 2px solid #d1dce5;
        border-radius: 15px;
        margin: 20px 0;
      }
      .split {
        display: flex;
        gap: 20px;
      }
      /* 왼쪽 이미지 영역 */
      .image-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #1f77b4;
        flex: 1;
      }
      hr.custom {
        border: none;
        border-top: 1px solid #d1dce5;
        margin: 10px 0 15px;
      }
      .image-section p {
        font-size: 20px;
        font-weight: bold;
        margin: 0;
        text-align: center;
      }
      .image-section img {
        width: 100%;
        border-radius: 8px;
      }
      /* 오른쪽 텍스트 영역: 세로로 두 개 박스 */
      .right-section {
        display: flex;
        flex-direction: column;
        gap: 20px;
        flex: 1;
      }
      .textbox {
        background-color: #ffffff;
        border-left: 6px solid #1f77b4;
        padding: 25px;
        border-radius: 12px;
        color: black;
      }
      .textbox h4 {
        margin-top: 0;
        margin-bottom: 10px;
      }
      .textbox ul {
        margin: 0;
        padding-left: 1.2em;
      }
    </style>
    <div class="gbox">
      <div class="split">
        <div class="image-section"
          <p style="font-size:30px; font-weight:bold; text-align:center; margin:0;">
            JobBusan Pipeline
          </p>
          <hr class="custom"/>
          <img
            src="https://raw.githubusercontent.com/seungcheoll/busan/main/image/pipeline.png"
            alt="JobBusan RAG 처리 구조도"
            style="width:80%; width:80%; display:block; margin:0 auto;"
          />
        </div>
        <div class="right-section">
          <div class="textbox">
            <h4>1️⃣ Job‑Bu 페이지 (기업 추천형 챗봇)</h4>
            <ul>
              <li>📋 먼저 사이드바에서 사용자 프로필을 입력하세요.</li>
              <li>❓ 조건 입력 후 질문 실행 버튼을 클릭하세요.</li>
              <li>📁 결과는 3개의 탭으로 구성되어 있습니다:
                <ul>
                  <li>✅ Job‑Bu 답변: 부산 내 강소기업 추천</li>
                  <li>📚 추천 기업 상세</li>
                  <li>🌍 추천 기업 위치</li>
                  <li>🔍 부산 기업 분포</li>
                </ul>
              </li>
            </ul>
          </div>
          <div class="textbox">
            <h4>2️⃣ Job‑Bu Chatbot (상담형 챗봇)</h4>
            <ul>
              <li>🤖 기업 추천 이후 추가 질문 가능</li>
              <li>📝 Job‑Bu 프로필과 문서를 바탕으로 정밀한 답변</li>
              <li>💡 예시 질문:
                <ul>
                  <li>"이 기업의 복지제도는 어떻게 되나요?"</li>
                  <li>"평균 연봉은 얼마인가요?"</li>
                  <li>"이 분야의 전망은?"</li>
                </ul>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
# ───────────────────────────────────────────
# [8] Job-Bu 페이지: LLM QA + 지도 시각화
# ───────────────────────────────────────────
# 📌 Job Busan 페이지 구성
if job_rag:
    st.markdown("""
        <div style='padding: 10px 0px;'>
            <h1 style='margin:0; font-size:28px; display: flex; align-items: center; gap: 12px;'>
                <img src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/chatbot.png' 
                     style='width: 60px; height: auto; vertical-align: middle;'>
                부산시 취업 상담 챗봇(Job-Bu)
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # 세션 상태 초기화
    if "llm" not in st.session_state:
        st.session_state.llm, st.session_state.retriever, st.session_state.company_df, st.session_state.map_html = init_qa_chain()
    if "templates" not in st.session_state:
        st.session_state.templates = load_all_templates()
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "main_query" not in st.session_state:
        st.session_state["main_query"] = ""
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""
    if "user_type" not in st.session_state:
        st.session_state["user_type"] = "대학생"
    if "saved_user_type" not in st.session_state:
        st.session_state["saved_user_type"] = ""
    if "saved_query" not in st.session_state:
        st.session_state["saved_query"] = ""

    # 🔐 입력값 저장 콜백 함수
    def save_user_inputs():
        st.session_state["saved_user_type"] = st.session_state["user_type"]
        st.session_state["saved_query"] = st.session_state["query_input"]

    # 🔎 질문 입력 및 유형 선택 영역
    col1, col2 = st.columns([3, 2])
    with col1:
        st.text_input(
            "❓ 질문으로 상담을 시작하세요!",
            key="query_input",
            value=st.session_state["main_query"],
            placeholder="예: 연봉 3000만원 이상 선박 제조업 추천",
            on_change=save_user_inputs
        )
    with col2:
        st.selectbox(
            "🏷️ 유형을 선택하세요!",
            ["대학생", "첫 취업 준비", "이직 준비"],
            key="user_type",
            on_change=save_user_inputs
        )

    query = st.session_state["query_input"]
    user_type = st.session_state["user_type"]

    # 💬 질문 실행 버튼
    if st.button("💬 질문 실행"):
        with st.spinner("🤖 Job-Bu가 부산 기업 정보를 검색 중입니다..."):
            selected_template = st.session_state.templates[user_type]
            formatted_template = selected_template.format(
                university   = st.session_state.university,
                major        = st.session_state.major,
                gpa          = st.session_state.gpa,
                field_pref   = st.session_state.field_pref,
                job_pref     = st.session_state.job_pref,
                activities   = st.session_state.activities,
                certificates = st.session_state.certificates
            )
    
            # 4) 포맷된 문자열로 PromptTemplate 생성
            prompt = PromptTemplate.from_template(formatted_template)
            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                retriever=st.session_state.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            result = qa_chain.invoke({"query": query})

            st.session_state.gpt_result = result["result"]
            st.session_state.source_docs = result["source_documents"]

            # 다시 비우기 전 최종 저장
            st.session_state["saved_query"] = query
            st.session_state["saved_user_type"] = user_type

            st.session_state["main_query"] = ""
            st.rerun()
    else:
        st.session_state["main_query"] = query
        
    # ───────────────────────────────────────
    # [8-1] 결과 탭 구성
    # ───────────────────────────────────────
    # 📁 결과 탭 구성
    selected_tabs = st.tabs([
        "✅ Job-Bu 답변",
        "📚 추천 기업 상세",
        "🌍 추천 기업 위치",
        "🔍 부산 기업 분포 및 검색"
    ])

    # 1️⃣ 답변 탭
    with selected_tabs[0]:
        st.write(st.session_state.get("gpt_result", "🔹 Job-Bu의 응답 결과가 여기에 표시됩니다."))

    # 2️⃣ 문서 탭
    with selected_tabs[1]:
        source_docs = st.session_state.get("source_docs", [])
        for i, doc in enumerate(source_docs):
            with st.expander(f"문서 {i+1}"):
                st.write(doc.page_content)

    # 3️⃣ 기업 위치 지도
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
            html(m._repr_html_(), height=550)
        else:
            st.info("해당 기업 위치 정보가 없습니다.")

    # 4️⃣ 기업 검색 및 지도 표시
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
            search_input = st.text_input(" ", key="search_input", label_visibility="collapsed", placeholder="🔎 회사명으로 검색 (예: 현대, 시스템, 조선 등)")
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

# ───────────────────────────────────────────
# [9] Groq Chatbot 페이지 (Job-Bu Chatbot)
# ───────────────────────────────────────────
# 🤖 Groq Chatbot 페이지
if chatbot:
    if "groq_chat" not in st.session_state:
        st.session_state.groq_chat = GroqLlamaChat(groq_api_key=load_api_key())

    if "groq_history" not in st.session_state:
        st.session_state.groq_history = [
            {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
        ]

    if "source_docs" not in st.session_state or not st.session_state.source_docs:
        st.warning("💡 'Job-Bu' 페이지에서 먼저 '질문 실행'을 눌러 상담에 필요한 참고자료를 확보해 주세요.")
        st.stop()

    # 🔹 사용자 유형과 질문 가져오기
    user_type = st.session_state.get("saved_user_type", "알 수 없음")
    user_query = st.session_state.get("saved_query", "입력된 질문이 없습니다")
    # 🔹 참고자료 포함 system prompt 구성
    context_text = "\n\n".join(doc.page_content for doc in st.session_state.source_docs)
    with open("template/sys_template.txt", "r", encoding="utf-8") as file:
        template=file.read()
    
    system_prompt = template.format(
        university   = st.session_state.university,
        major        = st.session_state.major,
        gpa          = st.session_state.gpa,
        field_pref   = st.session_state.field_pref,
        job_pref     = st.session_state.job_pref,
        activities   = st.session_state.activities,
        certificates = st.session_state.certificates,
        user_type    = user_type,
        user_query   = user_query,
        context_text = context_text
    )
    st.markdown("""
        <div style='background-color:#f9f9f9; padding:0px 0px; border-radius:12px; border:1px solid #ddd; 
                    width:20%; margin: 0 auto; text-align: center;'>
            <h1 style='margin:0; font-size:24px; display: flex; align-items: center; justify-content: center; gap: 10px; color: #000;'>
                <img src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/GPT_image2.png' 
                     style='width: 40px; height: auto; vertical-align: middle;'/>
                Job-Bu Chatbot
            </h1>
        </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.groq_history:
        if msg["role"] == "user":
            _, right = st.columns([3, 1])
            with right:
                st.markdown(
                    f"""
                    <div 
                        style='
                            padding: 12px; 
                            border-radius: 8px; 
                            background-color: #e0f7fa; 
                            width: fit-content; 
                            margin-left: auto; 
                            color: black;
                        '
                    >
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            left, _ = st.columns([2, 3])
            with left:
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: flex-start; gap: 10px;'>
                        <img 
                            src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/chatbot.png' 
                            style='width: 40px; height: auto; margin-top: 4px;'
                        />
                        <div 
                            style='
                                background-color: #f0f0f0; 
                                padding: 12px; 
                                border-radius: 8px; 
                                max-width: 100%; 
                                color: black;
                            '
                        >
                            {msg['content']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    prompt = st.chat_input("메시지를 입력하세요...", key="groq_input")
    if prompt:
        st.session_state.groq_history.append({"role": "user", "content": prompt})
        
        # ✅ 최근 5개만 포함
        recent_messages = st.session_state.groq_history[-5:]
        
        # ✅ system_prompt 고정 + 최근 메시지 순차 삽입
        history = [HumanMessage(content=system_prompt)]
        for m in recent_messages:
            history.append(
                (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            )

        answer = st.session_state.groq_chat._call(history)
        st.session_state.groq_history.append({"role": "assistant", "content": answer})
        st.rerun()
