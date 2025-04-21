# components/guide_page.py
import streamlit as st

def show_guide_page():
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
            JobBusan Pipeline
          </p>
          <hr class="custom"/>
          <img
            src="https://raw.githubusercontent.com/seungcheoll/busan/main/image/pipeline.png"
            alt="JobBusan RAG 처리 구조도"
            style="width:680px; height:505px; display:block; margin:0 auto;"
          />
        </div>
        <div class="right-section">
          <div class="textbox">
            <h4>1️⃣ Job‑Bu 페이지 (기업 추천형 챗봇)</h4>
            <ul>
              <li>📋 먼저 사이드바에서 사용자 프로필을 입력하세요.</li>
              <li>❓ 조건 입력 후 질문 실행 버튼을 클릭하세요.</li>
              <li>📁 결과는 4개의 탭으로 구성되어 있습니다:
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