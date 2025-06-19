import streamlit as st

# 기본 설정
st.set_page_config(page_title="사이드바 테스트", layout="wide")

# 사이드바 구성
with st.sidebar:
    st.title("📚 사이드바 메뉴")
    page = st.radio("이동할 페이지를 선택하세요", ["홈", "분석", "설정"])
    user_input = st.text_input("사용자 입력", placeholder="이름을 입력하세요")

# 본문 영역
st.write(f"### 현재 선택한 메뉴: {page}")

if page == "홈":
    st.success("🏠 홈 페이지에 오신 것을 환영합니다!")
elif page == "분석":
    st.info("📊 여기는 분석 페이지입니다.")
elif page == "설정":
    st.warning("⚙️ 설정 페이지입니다.")

if user_input:
    st.write(f"👋 안녕하세요, {user_input} 님!")
