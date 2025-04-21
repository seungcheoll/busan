# main.py
import streamlit as st
from components.sidebar import sidebar_ui
from components.guide_page import show_guide_page
from components.job_bu_page import show_job_bu_page
from components.chatbot_page import show_chatbot_page
from components.qa_utils import load_api_key

# ───────────────────────────────────────────
# [초기 세팅] Streamlit 환경 설정 및 세션 초기화
# ───────────────────────────────────────────

# ✅ 페이지 기본 설정
st.set_page_config(
    page_title="JobBusan",
    page_icon="https://raw.githubusercontent.com/seungcheoll/busan/main/image/chatbot.png",
    layout="wide"
)

# ✅ 기본 UI 숨기기
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ✅ 상단 여백 제거
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

# ✅ 사용자 입력값 초기화
for key in ["university", "major", "gpa", "field_pref", "job_pref", "activities", "certificates"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ───────────────────────────────────────────
# [라우팅] 사이드바 메뉴 선택 및 페이지 이동
# ───────────────────────────────────────────
choice = sidebar_ui()

if choice == "Guide":
    show_guide_page()
elif choice == "Job-Bu":
    show_job_bu_page()
elif choice == "Job-Bu Chatbot":
    show_chatbot_page()
