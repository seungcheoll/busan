# components/sidebar.py
import streamlit as st
from streamlit_option_menu import option_menu

def sidebar_ui():
    with st.sidebar:
        # 🔘 페이지 선택 메뉴
        choice = option_menu(
            menu_title="Page",
            options=["Guide", "Job-Bu", "Job-Bu Chatbot"],
            icons=["info-circle", "", ""],
            default_index=0,
            styles={
                "container": {"padding": "4!important", "background-color": "transparent"},
                "icon": {"display": "none"},
                "nav-link": {"font-size": "14px", "--hover-color": "#e5e9f2"},
                "nav-link-selected": {"background-color": "#3498db", "color": "white"},
            }
        )

        # ▼ 사용자 프로필 입력
        with st.expander("📋 사용자 프로필 입력", expanded=False):
            # ✅ 사용자 입력값 초기화
            for key in ["university", "major", "gpa", "field_pref", "job_pref", "activities", "certificates"]:
                if key not in st.session_state:
                    st.session_state[key] = ""
            with st.form("profile_form"):
                university_temp   = st.text_input("대학교", value=st.session_state.get("university", ""))
                major_temp        = st.text_input("전공", value=st.session_state.get("major", ""))
                gpa_temp          = st.text_input("학점", value=st.session_state.get("gpa", ""))
                field_pref_temp   = st.text_input("선호분야(산업군)", value=st.session_state.get("field_pref", ""))
                job_pref_temp     = st.text_input("선호직무", value=st.session_state.get("job_pref", ""))
                activities_temp   = st.text_area("경력사항", value=st.session_state.get("activities", ""))
                certificates_temp = st.text_area("보유 자격증", value=st.session_state.get("certificates", ""))

                if st.form_submit_button("입력 완료"):
                    st.session_state.university   = university_temp
                    st.session_state.major        = major_temp
                    st.session_state.gpa          = gpa_temp
                    st.session_state.field_pref   = field_pref_temp
                    st.session_state.job_pref     = job_pref_temp
                    st.session_state.activities   = activities_temp
                    st.session_state.certificates = certificates_temp

                    st.success("✅ 입력 완료!")

        # ▶️ 유튜브 시연 영상
        st.markdown("---")
        st.markdown("#### ▶️ 시연 영상")
        st.video("https://youtu.be/G_MKtEmmJt8")

    return choice, {
    "university": st.session_state.university,
    "major": st.session_state.major,
    "gpa": st.session_state.gpa,
    "field_pref": st.session_state.field_pref,
    "job_pref": st.session_state.job_pref,
    "activities": st.session_state.activities,
    "certificates": st.session_state.certificates
}
