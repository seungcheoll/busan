# components/chatbot_page.py
import streamlit as st
from langchain.schema.messages import HumanMessage, AIMessage
from components.qa_utils import load_api_key
from components.llm import GroqLlamaChat

def show_chatbot_page():
    if "groq_chat" not in st.session_state:
        st.session_state.groq_chat = GroqLlamaChat(groq_api_key=load_api_key())

    if "groq_history" not in st.session_state:
        st.session_state.groq_history = [
            {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
        ]

    if "source_docs" not in st.session_state or not st.session_state.source_docs:
        st.warning("💡 'Job-Bu' 페이지에서 먼저 '질문 실행'을 눌러 상담에 필요한 참고자료를 확보해 주세요.")
        st.stop()

    user_type = st.session_state.get("saved_user_type", "알 수 없음")
    user_query = st.session_state.get("saved_query", "입력된 질문이 없습니다")
    st.write(user_query)
    st.write(user_type)
    context_text = "\n\n".join(doc.page_content for doc in st.session_state.source_docs)

    with open("template/sys_template.txt", "r", encoding="utf-8") as file:
        template = file.read()

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
                st.markdown(f"""
                    <div style='padding: 12px; border-radius: 8px; background-color: #e0f7fa; width: fit-content; margin-left: auto; color: black;'>
                        {msg['content']}
                    </div>
                """, unsafe_allow_html=True)
        else:
            left, _ = st.columns([2, 3])
            with left:
                st.markdown(f"""
                    <div style='display: flex; align-items: flex-start; gap: 10px;'>
                        <img src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/chatbot.png' 
                             style='width: 40px; height: auto; margin-top: 4px;'/>
                        <div style='background-color: #f0f0f0; padding: 12px; border-radius: 8px; max-width: 100%; color: black;'>
                            {msg['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    prompt = st.chat_input("메시지를 입력하세요...", key="groq_input")
    if prompt:
        st.session_state.groq_history.append({"role": "user", "content": prompt})

        recent_messages = st.session_state.groq_history[-5:]
        history = [HumanMessage(content=system_prompt)]
        for m in recent_messages:
            role = HumanMessage if m["role"] == "user" else AIMessage
            history.append(role(content=m["content"]))

        answer = st.session_state.groq_chat._call(history)
        st.session_state.groq_history.append({"role": "assistant", "content": answer})
        st.rerun()
