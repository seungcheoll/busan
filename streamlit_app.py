# 🔚 전체 앱 구조 요약
# 1) start_page → check_login → input_profile 순으로 초기 단계 진입
# 2) 사이드바 메뉴에 따라 Guide / Career / Dream 페이지 렌더링
# 3) Career는 기업추천(RAG) + 채용정보 + 지도 + 챗봇 응답 포함
# 4) Dream은 사용자 프로필만으로 GPT 상담 제공


# ───────────────────────────────────────────
# [1] 라이브러리 임포트
# ───────────────────────────────────────────
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import folium
from folium import Popup, Marker
from streamlit.components.v1 import html
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import ChatResult
from openai import OpenAI
import json
import streamlit.components.v1 as components

# ───────────────────────────────────────────
# [2] 기본 설정 및 사용자 인증 처리
# ───────────────────────────────────────────
# Streamlit 기본 설정 (타이틀, 아이콘, 레이아웃)
st.set_page_config(
    page_title="JOB-IS",
    page_icon="https://raw.githubusercontent.com/seungcheoll/busan/main/image/jobis.png",
    layout="wide"
)

# 시작 페이지 (로고 및 이용 버튼 표시)
def start_page():
    if "started" not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])  # 가운데 정렬
        with col2:
            st.markdown("""
                <div style="
                    background-color: #FFFFFF;
                    padding: 0px;
                    border-radius: 10px;
                    text-align: center;
                    width: 500px;
                    margin: 0 auto;
                ">
                    <img src="https://raw.githubusercontent.com/seungcheoll/busan/main/image/logo_raw.png" 
                         style="width: 500px; height: 250px; display: block; margin: 0 auto;">
                </div>
            """, unsafe_allow_html=True)

        # 버튼만 따로 가운데 정렬
        btn_col1, btn_col2, btn_col3 = st.columns([1.75, 1, 1])
        with btn_col2:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)  # 버튼 위 여백 최소
            if st.button("이용하러 가기"):
                st.session_state.started = True
                st.rerun()

        st.stop()

# 로그인 페이지 (로고 및 이용 버튼 표시)     
def check_login():
    if not st.session_state.get("authenticated", False):
        col1, col2, col3 = st.columns([1, 2, 1])  # 가운데 열이 넓도록 설정

        with col2:
            st.markdown('<h2 style="text-align:center;">😊 JOB-IS에 오신 걸 환영합니다!</h2>', unsafe_allow_html=True)

            with st.form("login_form"):
                pw = st.text_input("", type="password",label_visibility="collapsed", placeholder="로그인 코드를 입력하세요")
                submitted = st.form_submit_button("로그인")
                if submitted:
                    if pw == st.secrets["general"]["APP_PASSWORD"]:
                        st.session_state.authenticated = True
                        st.success("✅ 로그인 성공!")
                        st.rerun()
                    else:
                        st.error("❗ 코드가 올바르지 않습니다.")
        st.stop()


# 사용자 프로필 입력 함수
def input_profile():
    if "profile_done" not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])  # 가운데 열을 더 넓게

        with col2:  # 두 번째 컬럼에만 폼 표시
            st.markdown('<h2 style="text-align:center;">📋 사용자 정보를 입력해주세요</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align:center; color:red; font-weight:bold; font-size:16px;">
            ※ 프로필 정보 미입력 시 진로 상담 챗봇(Dream Chat)의 사용이 제한됩니다. ※
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("profile_form"):
                university   = st.text_input("대학교", placeholder="예: OO대학교")
                major        = st.text_input("전공", placeholder="예: OO학과")
                gpa          = st.text_input("학점", placeholder="예: 4.5")
                field_pref   = st.text_input("선호분야(산업군)", placeholder="예: 제조업")
                job_pref     = st.text_input("선호직무", placeholder="예: 개발자")
                activities   = st.text_area("경력사항", placeholder="예: OO공모전 수상 \n OO서포터즈 ...")
                certificates = st.text_area("보유 자격증", placeholder="예: ADsP\nSQLD")
                
                agree = st.checkbox("개인정보 수집 및 이용 동의(※ 입력된 정보는 맞춤형 취업 상담을 위한 용도로만 사용됩니다.)")
                submitted = st.form_submit_button("입력 완료")

                if submitted:
                    if not agree:
                        st.warning("❗ 원활한 상담을 위한 개인정보 수집 및 이용 동의를 부탁드립니다.")
                    else:
                        st.session_state.university   = university
                        st.session_state.major        = major
                        st.session_state.gpa          = gpa
                        st.session_state.field_pref   = field_pref
                        st.session_state.job_pref     = job_pref
                        st.session_state.activities   = activities
                        st.session_state.certificates = certificates
                        st.session_state.profile_done = True
                        st.success("✅ 프로필 정보 저장 완료!")
                        st.rerun()
        st.stop()
start_page()
check_login()
input_profile()

# ───────────────────────────────────────────
# [3] 유틸 함수: JSON 파싱 및 텍스트 처리
# ───────────────────────────────────────────
def strip_code_blocks(text):
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    return text

def text_to_json(text):
    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError as e:
        return f"JSON 변환 오류: {e}"
        
# ───────────────────────────────────────────
# [4] GPT LLM 래퍼 클래스 정의
# ───────────────────────────────────────────
# ✅ GPT용 LLM 클래스 정의
class GPTChatWrapper(BaseChatModel):
    openai_api_key: str
    model: str = "gpt-4o"
    _client: OpenAI = None

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI(api_key=self.openai_api_key)

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
            temperature=0
        )
        return response.choices[0].message.content

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        content = self._call(messages, **kwargs)
        return ChatResult(generations=[{"text": content, "message": AIMessage(content=content)}])

    @property
    def _llm_type(self):
        return "openai-gpt"

    @property
    def _identifying_params(self):
        return {"model": self.model}
        
# ───────────────────────────────────────────
# [5] API 키 및 템플릿 로딩 함수 정의
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
# [6] 벡터 DB 및 QA 체인 초기화
# ───────────────────────────────────────────
# ✅ GPTChatWrapper 적용을 위해 init_qa_chain 함수
@st.cache_resource
def init_qa_chain():
    api_key = load_api_key()
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = GPTChatWrapper(openai_api_key=api_key)
    company_df_for_gpt = pd.read_excel("main.xlsx")
    company_df_for_map = pd.read_excel("map_busan.xlsx")
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()

    return llm, retriever, company_df_for_gpt, company_df_for_map, map_html_content


# ───────────────────────────────────────────
# [7] 스타일 커스터마이징 (Streamlit UI 숨기기 등)
# ───────────────────────────────────────────
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
# [8] 사용자 프로필 세션 상태 초기화
# ───────────────────────────────────────────
for key in ["university", "major", "gpa", "field_pref", "job_pref", "activities", "certificates"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ───────────────────────────────────────────
# [9] 사이드바 메뉴 및 시연 영상
# ───────────────────────────────────────────
# 🔘 사이드바 라디오 메뉴 설정
with st.sidebar:
    # 🔘 페이지 선택 메뉴
    choice = option_menu(
        menu_title="Page",
        options=["Guide","Career Chat", "Dream Chat"],
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

    # ▶️ 시연 영상 (YouTube 삽입)
    st.markdown("")
    st.sidebar.markdown("#### ▶️ 시연 영상")
    st.sidebar.video("https://youtu.be/XwpaQ3lSH88")  # 여기에 유튜브 링크 삽입
Guide = choice == "Guide"
Career = choice == "Career Chat"
Dreamer = choice == "Dream Chat"

# ───────────────────────────────────────────
# [10] Guide 페이지 렌더링
# ───────────────────────────────────────────
if Guide:
    st.markdown("""
    <h1 style='text-align: center;'>
      <img src="https://raw.githubusercontent.com/seungcheoll/busan/main/image/logo_guide.png" 
           style="width: 180px; height: 70px; vertical-align: middle; margin-right: 0px;">
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
      .gbox {
        background-color: #f0f4f8;
        padding: 30px;
        border: 2px solid #d1dce5;
        border-radius: 15px;
        margin: 20px 0;
      }

      /* 모든 텍스트를 검정색으로 고정 */
      .gbox, .gbox * {
        color: black !important;
      }

      .split {
        display: flex;
        gap: 20px;
      }

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
        <div class="image-section">
          <p style="font-size:30px; font-weight:bold; text-align:center; margin:0;">
            System Workflow
          </p>
          <hr class="custom"/>
          <img
            src="https://raw.githubusercontent.com/seungcheoll/busan/main/image/flow.png"
            alt="JobBusan RAG 처리 구조도"
            style="width:600px; height:510px; display:block; margin:0 auto; padding-top: 20px;"
          />
        </div>
        <div class="right-section">
            <div class="textbox">
              <h4>1️⃣ Career Chat (기업 매칭 서비스)</h4>
              <ul>
                <li>❓ 질문 입력 및 유형 선택 후 질문 실행 버튼을 클릭하세요.</li>
                <li>📁 결과는 5개의 탭으로 구성되어 있습니다.
                  <ul>
                    <li>✅ Job-Busan 답변: 부산 내 강소기업 추천</li>
                    <li>📚 추천 기업 상세</li>
                    <li>📢 관련 채용 정보(JobKorea)</li>
                    <li>🌍 추천 기업 위치</li>
                    <li>🔍 부산 기업 분포 : 원하는 기업 검색</li>
                  </ul>
                </li>
                <li>💬 JOB-IS 답변 기반 취업 상담 챗봇이 함께 제공됩니다.</li>
              </ul>
            </div>
            <div class="textbox">
              <h4>2️⃣ Dream Chat (진로 상담 서비스)</h4>
              <ul>
                <li>🧠 사용자가 입력한 프로필 정보를 바탕으로 진로 상담을 제공합니다.</li>
                <li>💡 예시 질문
                  <ul>
                    <li>"제 전공에 맞는 직무가 궁금해요."</li>
                    <li>"관심 분야에서 필요한 자격증은 뭔가요?"</li>
                    <li>"해외 취업도 고려 중인데, 어떤 준비가 필요할까요?"</li>
                    <li>"저와 비슷한 경력을 가진 사람들은 어떤 기업에 입사하나요?"</li>
                  </ul>
                </li>
              </ul>
            </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
# ───────────────────────────────────────────
# [11] Career Chat 페이지 (기업 추천 + 지도 시각화 + 취업 상담 챗봇)
# ───────────────────────────────────────────
# 📌 Career Chat 페이지 구성
if Career:
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Career_rag"
        
    if st.session_state.current_page == "Career_rag":
        if st.button("◀️ Career Chat 이용하기"):
            st.session_state.current_page = "Career_chatbot"
            st.rerun()
            
        st.markdown("""
            <div style='padding: 10px 0px; display: flex; align-items: center; gap: 10px;'>
                <img src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/jobis.png' 
                     style='width: 45px; height: 45px;'>
                <span style='font-size:28px; font-weight: bold;'>Career Chat(기업 매칭 서비스)</span>
            </div>
        """, unsafe_allow_html=True)
    
        # 세션 상태 초기화
        if "llm" not in st.session_state:
            st.session_state.llm, st.session_state.retriever, st.session_state.company_df_for_gpt, st.session_state.company_df_for_map, st.session_state.map_html = init_qa_chain()
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
    

        if st.button("💬 질문 실행"):
            # 🔄 이전 챗봇 대화 내용 초기화
            for key in ["career_history", "dream_history","content_to_gpt"]:
                st.session_state.pop(key,None)
            with st.spinner("🔎 JOB-IS가 기업 정보를 검색 중입니다."):
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
                for _ in range(3):  # 최대 3번만 시도
                    try:
                        result = qa_chain.invoke({"query": query})
                        text = result["result"]
                        text = strip_code_blocks(text)
                        text = text_to_json(text)
                        st.session_state.gpt_result = text["전체 출력 결과"]
                        st.session_state.company_name_by_gpt = text["기업명"]
                        break
                    except:
                        continue
                else:
                    st.error(f"응답 처리에 실패했습니다: {last_error}")
                st.session_state.source_docs = result["source_documents"]
    
                # 다시 비우기 전 최종 저장
                st.session_state["saved_query"] = query
                st.session_state["saved_user_type"] = user_type
    
                st.session_state["main_query"] = ""
                st.rerun()
        else:
            st.session_state["main_query"] = query
            
        # ───────────────────────────────────────
        # [11-1] 결과 탭 구성
        # ───────────────────────────────────────
        # 📁 결과 탭 구성
        selected_tabs = st.tabs([
            "✅ JOB-IS 답변",
            "📚 추천 기업 상세",
            "📢 관련 채용 정보(JobKorea)",
            "🌍 추천 기업 위치",
            "🔍 부산 기업 분포 및 검색"
        ])
    
        # 1️⃣ 답변 탭
        with selected_tabs[0]:
            st.write(st.session_state.get("gpt_result", "🔹 JOB-IS의 응답 결과가 여기에 표시됩니다."))
    
        # 2️⃣ 문서 탭
        with selected_tabs[1]:
            raw_names = st.session_state.get("company_name_by_gpt", "")
            company_name_by_gpt = [name.strip() for name in raw_names.split(",")]
            # 2. isin()으로 필터링
            matched_df_by_gpt = st.session_state.company_df_for_gpt[
                st.session_state.company_df_for_gpt['회사명'].isin(company_name_by_gpt)
            ]
            if matched_df_by_gpt.empty:
                st.warning("일치하는 기업이 없습니다.")
            else:        
                # 필드 분류
                basic_fields = [
                    '회사명', '잡코리아 주소','홈페이지', '업 종', '상세업종', '사업분야',
                    '평균초임', '평균연봉', '기업분류',
                    '매출액', '직원수(계)',
                     '도로명', '주요제품 / 서비스'
                ]
                work_life_fields = [f'워라벨{i}' for i in range(1, 11)]
                training_fields = [f'직무교육{i}' for i in range(1, 7)]
                welfare_fields = [f'복리후생{i}' for i in range(1, 13)]
        
                # 병합 유틸
                def join_fields(row, fields):
                    values = [str(row[f]).strip() for f in fields if pd.notna(row[f]) and str(row[f]).strip() != '']
                    return ' / '.join(values)
        
                # 행 포맷 함수
                def format_row(row):
                    lines = []
                    for field in basic_fields:
                        value = row.get(field, '')
                        if pd.notna(value) and str(value).strip() != '':
                            lines.append(f"{field}: {str(value).strip()}")
                    lines.append(f"워라벨: {join_fields(row, work_life_fields)}")
                    lines.append(f"직무교육: {join_fields(row, training_fields)}")
                    lines.append(f"복리후생: {join_fields(row, welfare_fields)}")
                    info = "\n\n".join(lines)
        
                    desc = str(row.get("기업설명", "")).strip()
                    return f"1. 기업정보\n\n{info}\n\n\n2. 기업설명\n\n{desc}"
                    
                st.session_state.setdefault("content_to_gpt", [])
                # 👉 Expander에 표시
                for _, row in matched_df_by_gpt.iterrows():
                    content_to_gpt={}
                    with st.expander(f"**{row['회사명']}** 상세 정보"):
                        content = format_row(row)
                        st.session_state.content_to_gpt.append(content)
                        st.write(content)
            
        # 3️⃣ JOBKOREA
        with selected_tabs[2]:
            raw_names = st.session_state.get("company_name_by_gpt", "")
            company_name_by_gpt = [name.strip() for name in raw_names.split(",")]
            # 2. isin()으로 필터링
            matched_df_by_gpt = st.session_state.company_df_for_gpt[
                st.session_state.company_df_for_gpt['회사명'].isin(company_name_by_gpt)
            ]
            if matched_df_by_gpt.empty:
                st.warning("⚠️ 본 페이지는 상업적 목적이나 원본 UI 복제를 의도하지 않았으며, JOB-IS 서비스 컨셉 구현을 위해 참고용으로 제작되었습니다.\n\n※ 추후 정식 서비스 개발 시, JobKorea API 등 공식 연동 방식으로 교체할 예정입니다.")
                st.warning("일치하는 기업이 없습니다.")
            else:
                st.warning("⚠️ 본 페이지는 상업적 목적이나 원본 UI 복제를 의도하지 않았으며, JOB-IS 서비스 컨셉 구현을 위해 참고용으로 제작되었습니다.\n\n※ 추후 정식 서비스 개발 시, JobKorea API 등 공식 연동 방식으로 교체할 예정입니다.")
                for _, row in matched_df_by_gpt.iterrows():
                    name = row['회사명']
                    jk_url = row['잡코리아 주소']
                    # expander 생성
                    with st.expander(f"**{name}** 채용 정보"):
                        # iframe으로 잡코리아 페이지 임베딩
                        components.iframe(
                            src=jk_url,
                            height=1000,      # iframe 높이 (필요에 따라 조정)
                            scrolling=True
                        )
        # 4️⃣ 기업 위치 지도
        with selected_tabs[3]:
            raw_names = st.session_state.get("company_name_by_gpt", "")
            company_name_by_gpt = [name.strip() for name in raw_names.split(",")]
            matched_df = st.session_state.company_df_for_map[st.session_state.company_df_for_map['기업명'].isin(company_name_by_gpt)]
            if not matched_df.empty:
                m = folium.Map(location=[matched_df["위도"].mean(), matched_df["경도"].mean()], zoom_start=12)
                
                for _, row in matched_df.iterrows():
                    # 원으로 시각화
                    folium.CircleMarker(
                        location=[row["위도"], row["경도"]],
                        radius=6,
                        color="blue",
                        fill=True,
                        fill_color="blue",
                        fill_opacity=0.6
                    ).add_to(m)
                
                    # 이름 팝업 항상 열기 (Marker + Popup 조합)
                    popup = folium.Popup(row["기업명"], max_width=200, show=True)
                    folium.Marker(
                        location=[row["위도"], row["경도"]],
                        popup=popup,
                        icon=folium.DivIcon(icon_size=(0, 0))  # 아이콘 숨김 (텍스트만 표시)
                    ).add_to(m)
                
                html(m._repr_html_(), height=550)
            else:
                st.warning("일치하는 기업이 없습니다.")
    
        # 5️⃣ 기업 검색 및 지도 표시
        with selected_tabs[4]:
            if "search_keyword" not in st.session_state:
                st.session_state.search_keyword = ""
            if "reset_triggered" not in st.session_state:
                st.session_state.reset_triggered = False
            if "search_field" not in st.session_state:
                st.session_state.search_field = "기업명"
        
            def reset_search():
                st.session_state.search_keyword = ""
                st.session_state["search_input"] = ""
                st.session_state["search_field"] = "기업명"
                st.session_state.reset_triggered = True
        
            col1, col2, col3 = st.columns([2, 1, 1])
        
            with col1:
                search_input = st.text_input(" ", key="search_input", label_visibility="collapsed", placeholder="🔎 기업명 또는 산업 분야 입력(예 : 조선/소프트웨어)")
        
            with col2:
                st.selectbox("",["기업명", "산업 분야"], key="search_field", label_visibility="collapsed")
        
            with col3:
                if search_input:
                    st.button("검색 초기화", on_click=reset_search)
        
            st.session_state.search_keyword = search_input
        
            if st.session_state.reset_triggered:
                st.session_state.reset_triggered = False
                st.rerun()
        
            matched_df = pd.DataFrame()
            keyword = st.session_state.search_keyword.strip()
            if keyword:
                search_column = "기업명" if st.session_state.search_field == "기업명" else "산업 분야"
                matched_df = st.session_state.company_df_for_map[
                    st.session_state.company_df_for_map[search_column].str.contains(keyword, case=False, na=False)
                ]
        
            col1, col2 = st.columns([2, 1])
            with col2:
                st.markdown("### 🧾 검색 기업 정보")
                if not matched_df.empty:
                    PINLEFT = {'pinned': 'left'}
                    PRECISION_TWO = {'type': ['numericColumn'], 'precision': 6}
                    formatter = {
                        '기업명': ('기업명', PINLEFT),
                        '주소': ('주소', {'width': 200}),
                        '산업 분야': ('산업 분야', {'width': 150}),
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
                        height=418,
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
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.8,
                            tooltip=row['기업명']
                        ).add_to(m)
        
                        popup = folium.Popup(row["기업명"], max_width=200, show=True)
                        folium.Marker(
                            location=[row["위도"], row["경도"]],
                            popup=popup,
                            icon=folium.DivIcon(icon_size=(0, 0))
                        ).add_to(m)
        
                    html(m._repr_html_(), height=480)
                    st.caption(f"✅ 선택된 기업 {len(df_map)}곳을 지도에 표시했습니다.")
                elif not matched_df.empty:
                    m = folium.Map(location=[matched_df['위도'].mean(), matched_df['경도'].mean()], zoom_start=12)
                    for _, row in matched_df.iterrows():
                        folium.CircleMarker(
                            location=[row['위도'], row['경도']],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.7,
                            popup=row['기업명'],
                            tooltip=row['기업명']
                        ).add_to(m)
                    html(m._repr_html_(), height=480)
                    st.caption(f"※ 검색 결과 기업 {len(matched_df)}곳을 지도에 표시했습니다.")
                elif keyword:
                    st.warning("🛑 해당 기업이 존재하지 않습니다.")
                else:
                    html(st.session_state.map_html, height=480)
                    st.caption("※ 전체 기업 분포를 표시 중입니다.")

    # ───────────────────────────────────────────
    # [11-2] Career Chat 페이지 (취업 상담 전용 챗봇)
    # ───────────────────────────────────────────
    elif st.session_state.current_page == "Career_chatbot":
        if st.button("◀️ 이전 페이지로 돌아가기"):
            st.session_state.current_page = "Career_rag"
            st.rerun()
        if "career_chat" not in st.session_state:
            st.session_state.career_chat = GPTChatWrapper(openai_api_key=load_api_key())
        
        if "career_history" not in st.session_state:
            st.session_state.career_history = [
                {"role": "assistant", "content": "안녕하세요! 취업 상담 챗봇 Career Chat입니다! 무엇을 도와드릴까요?"}
            ]
    
        if "source_docs" not in st.session_state or not st.session_state.source_docs:
            st.warning("💡 이전 페이지에서 먼저 질문 입력 후, '질문 실행'을 눌러 상담에 필요한 참고자료를 확보해 주세요.")
            st.stop()
    
        # 🔹 사용자 유형과 질문 가져오기
        user_type = st.session_state.get("saved_user_type", "알 수 없음")
        user_query = st.session_state.get("saved_query", "입력된 질문이 없습니다")
        # 🔹 content_to_gpt 가 없거나 비어 있으면 빈 리스트, 아니면 그 값을 사용
        context_list = st.session_state.get("content_to_gpt", [])
        
        # 🔹 context_text 생성: 리스트에 내용이 있으면 join, 없으면 빈 문자열
        if context_list:
            context_text = "\n\n".join(context_list)
        else:
            context_text = ""
        with open("template/sys_template_gpt_rag.txt", "r", encoding="utf-8") as file:
            template=file.read()
        
        system_prompt_career = template.format(
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
                    Career Chat
                </h1>
            </div>
        """, unsafe_allow_html=True)
    
        for msg in st.session_state.career_history:
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
                                src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/jobis.png' 
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
    
        user_career = st.chat_input("메시지를 입력하세요...", key="career_input")
        if user_career:
            st.session_state.career_history.append({"role": "user", "content": user_career})
            
            # ✅ 최근 10개만 포함
            recent_messages = st.session_state.career_history[-10:]
            
            # ✅ system_prompt 고정 + 최근 메시지 순차 삽입
            history_career = [HumanMessage(content=system_prompt_career)]
            for m in recent_messages:
                history_career.append(
                    (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
                )
            with st.spinner("💬 Career Chat이 답변을 생성 중입니다..."):
                answer_career = st.session_state.career_chat._call(history_career)
                st.session_state.career_history.append({"role": "assistant", "content": answer_career})
            st.rerun()
            
# ───────────────────────────────────────────
# [12] Dream Chat 페이지 (진로 상담 전용 챗봇)
# ───────────────────────────────────────────

# 🤖 Dream Chat 페이지
if Dreamer:
    if "dream_chat" not in st.session_state:
        st.session_state.dream_chat = GPTChatWrapper(openai_api_key=load_api_key())

    if "dream_history" not in st.session_state:
        st.session_state.dream_history = [
            {"role": "assistant", "content": "안녕하세요! 진로 상담 챗봇 Dream Chat입니다! 무엇을 도와드릴까요?"}
        ]

    # 🔹 사용자 프로필 입력값 확인
    university   = st.session_state.get("university", "").strip()
    major        = st.session_state.get("major", "").strip()
    gpa          = st.session_state.get("gpa", "").strip()
    field_pref   = st.session_state.get("field_pref", "").strip()
    job_pref     = st.session_state.get("job_pref", "").strip()
    activities   = st.session_state.get("activities", "").strip()
    certificates = st.session_state.get("certificates", "").strip()

    # 🔒 모두 비었으면 중단
    if not any([university, major, gpa, field_pref, job_pref, activities, certificates]):
        st.warning("⚠️ 사용자 프로필이 입력되지 않았습니다. 프로필 정보를 입력해 주세요.")
        st.stop()

    # 🔧 system prompt 구성 (user_type, user_query, context_text 제거)
    with open("template/sys_template_gpt_job.txt", "r", encoding="utf-8") as file:
        template = file.read()

    system_prompt_dream = template.format(
        university   = university,
        major        = major,
        gpa          = gpa,
        field_pref   = field_pref,
        job_pref     = job_pref,
        activities   = activities,
        certificates = certificates
    )

    # 💬 타이틀
    st.markdown("""
        <div style='background-color:#f9f9f9; padding:0px 0px; border-radius:12px; border:1px solid #ddd; 
                    width:20%; margin: 0 auto; text-align: center;'>
            <h1 style='margin:0; font-size:24px; display: flex; align-items: center; justify-content: center; gap: 10px; color: #000;'>
                <img src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/GPT_image2.png' 
                     style='width: 40px; height: auto; vertical-align: middle;'/>
                Dream Chat
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # 💬 히스토리
    for msg in st.session_state.dream_history:
        if msg["role"] == "user":
            _, right = st.columns([3, 1])
            with right:
                st.markdown(
                    f"""
                    <div style='padding:12px; border-radius:8px; background-color:#e0f7fa; 
                                width:fit-content; margin-left:auto; color:black;'>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True
                )
        else:
            left, _ = st.columns([2, 3])
            with left:
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: flex-start; gap: 10px;'>
                        <img 
                            src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/jobis.png' 
                            style='width: 40px; height: auto; margin-top: 4px;' />
                        <div style='background-color:#f0f0f0; padding:12px; border-radius:8px; max-width:100%; color:black;'>
                            {msg['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

    # 📥 입력받기
    user_dream = st.chat_input("메시지를 입력하세요...", key="dream_input")
    if user_dream:
        st.session_state.dream_history.append({"role": "user", "content": user_dream})

        # 최근 10개 메시지로 히스토리 구성
        recent_messages = st.session_state.dream_history[-10:]
        history_dream = [HumanMessage(content=system_prompt_dream)]
        for m in recent_messages:
            history_dream.append(
                (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            )
        with st.spinner("💬 Dream Chat이 답변을 생성 중입니다..."):
            answer_dream = st.session_state.dream_chat._call(history_dream)
            st.session_state.dream_history.append({"role": "assistant", "content": answer_dream})
        st.rerun()
