# 📦 필요한 라이브러리 불러오기
import streamlit as st                  # 웹 앱 프레임워크 (간단한 인터페이스로 앱 만들 수 있음)
import pandas as pd                    # 데이터 처리용 라이브러리 (엑셀이나 테이블 다루기)
import folium                          # 지도 시각화 도구 (지도 위에 마커 표시 가능)
from streamlit.components.v1 import html  # Streamlit에서 HTML 코드 삽입할 때 사용
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# LangChain 관련: 질문-답변 체계 구축용
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import ChatResult

# Groq API 연동용
from groq import Groq

# ✅ 1. 사용자 정의 챗봇 클래스 정의 (Groq + LLaMA 모델 연결용)
class GroqLlamaChat(BaseChatModel):
    groq_api_key: str
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    _client: Groq = None

    def __init__(self, **data):
        super().__init__(**data)
        self._client = Groq(api_key=self.groq_api_key)

    # 🧠 대화 메시지 정리 후 API 호출
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

    # 응답 생성
    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        content = self._call(messages, **kwargs)
        return ChatResult(
            generations=[{"text": content, "message": AIMessage(content=content)}]
        )

    # 모델 정보 속성 정의
    @property
    def _llm_type(self):
        return "groq-llama-4"

    @property
    def _identifying_params(self):
        return {"model": self.model}

# ✅ 2. API 키 불러오기 (Streamlit Secrets에서 가져옴)
def load_api_key():
    return st.secrets["general"]["API_KEY"]

# ✅ 3. 프롬프트 템플릿 로드 (질문에 사용할 틀)
def load_template():
    with open("template.txt", "r", encoding="utf-8") as file:
        return file.read()

# ✅ 4. 체인 초기화 함수 (질문 응답 체계 준비 + 지도 로딩)
@st.cache_resource
def init_qa_chain():
    api_key = load_api_key()
    template = load_template()

    # 문장 임베딩 모델 (한국어 문장을 벡터로 변환)
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    # 벡터 데이터베이스 로드
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)

    # 검색기 정의 (상위 5개 문서 검색)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM 초기화 (Groq 모델)
    llm = GroqLlamaChat(groq_api_key=api_key)

    # 프롬프트 구성
    prompt = PromptTemplate.from_template(template)

    # QA 체인 구성 (질문하면 관련 문서를 찾아서 답변 생성)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # 기업 위치 엑셀 파일 로드
    company_df = pd.read_excel("map_busan.xlsx")

    # 전체 기업 지도 HTML 로드
    with open("map_company.html", "r", encoding="utf-8") as f:
        map_html_content = f.read()

    return qa_chain, company_df, map_html_content

# ✅ 5. Streamlit 앱 초기 설정
st.set_page_config(page_title="부산 기업 RAG", layout="wide")
st.title("🚢 부산 취업 상담 챗봇(JOB MAN)")

# ✅ 6. 체인 로딩 (최초 1회만 실행)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.company_df, st.session_state.map_html = init_qa_chain()

# 세션 상태에 query 변수 초기화
if "query" not in st.session_state:
    st.session_state.query = ""

# 텍스트 입력값 저장용 상태
if "main_query" not in st.session_state:
    st.session_state["main_query"] = ""

# 상태 값 가져오기
query = st.session_state["main_query"]

# ✅ 7. 사용자 질문 입력창 표시
query = st.text_input(
    "🎯 질문을 입력하세요:",
    value=query,
    key="query_input",  # 위 상태와 연결된 key는 아님
    placeholder="예: 연봉 3000만원 이상 선박 제조업 추천"
)

# ✅ 8. 질문 실행 버튼 누르면 처리
if st.button("💬 질문 실행"):
    with st.spinner("🤖 JOB MAN이 부산 기업 정보를 검색 중입니다..."):
        result = st.session_state.qa_chain.invoke(query)  # 질문 실행
        st.session_state.gpt_result = result["result"]    # 응답 저장
        st.session_state.source_docs = result["source_documents"]  # 문서 저장
        st.session_state["main_query"] = ""  # 입력창 비우기
        st.rerun()
else:
    st.session_state["main_query"] = query  # 입력 중일 때 실시간 저장

# ✅ 9. 결과 보여줄 탭 구성
selected_tabs = st.tabs(["✅ JOB MAN의 답변", "📚 참고 문서", "🌍 관련 기업 위치", "🔍 부산 기업 분포 및 검색"])

# GPT 응답 결과 출력
with selected_tabs[0]:
    st.write(st.session_state.get("gpt_result", "🔹 GPT 응답 결과가 여기에 표시됩니다."))

# 참조 문서 내용 출력
with selected_tabs[1]:
    source_docs = st.session_state.get("source_docs", [])
    for i, doc in enumerate(source_docs):
        with st.expander(f"문서 {i+1}"):
            st.write(doc.page_content)

# 관련 기업 위치 지도 출력
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
        html(m._repr_html_(), height=700)
    else:
        st.info("해당 기업 위치 정보가 없습니다.")

# 기업명 검색 기반 지도 시각화
with selected_tabs[3]:
    if "search_keyword" not in st.session_state:
        st.session_state.search_keyword = ""
    if "reset_triggered" not in st.session_state:
        st.session_state.reset_triggered = False

    def reset_search():
        st.session_state.search_keyword = ""
        st.session_state["search_input"] = ""
        st.session_state.reset_triggered = True

    search_input = st.text_input(
        label="",
        key="search_input",
        placeholder="🔎 회사명으로 검색 (예: 현대, 시스템, 조선 등)"
    )
    st.session_state.search_keyword = st.session_state.get("search_input", "")

    if st.session_state.search_keyword:
        st.button("검색 초기화", on_click=reset_search)

    if st.session_state.reset_triggered:
        st.session_state.reset_triggered = False
        st.rerun()

    matched_df = pd.DataFrame()
    if st.session_state.search_keyword.strip():
        matched_df = st.session_state.company_df[
            st.session_state.company_df["회사명"].str.contains(
                st.session_state.search_keyword,
                case=False,
                na=False
            )
        ]

    col1, col2 = st.columns([2, 1])  # 지도:테이블 비율
# 초기화: 선택한 회사명 상태
if "selected_company_name" not in st.session_state:
    st.session_state.selected_company_name = None

with col1:
    # 👇 선택된 회사명으로 지도에 표시할 기업만 필터링
    selected_name = st.session_state.get("selected_company_name", None)
    filtered_df = matched_df

    if selected_name:
        filtered_df = matched_df[matched_df["회사명"] == selected_name]

    if not filtered_df.empty:
        m = folium.Map(
            location=[filtered_df["위도"].mean(), filtered_df["경도"].mean()],
            zoom_start=12
        )
        for _, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[row["위도"], row["경도"]],
                radius=5,
                color="crimson" if selected_name else "green",
                fill=True,
                fill_color="crimson" if selected_name else "green",
                fill_opacity=0.7,
                popup=row["회사명"],
                tooltip=row["회사명"]
            ).add_to(m)
        html(m._repr_html_(), height=600)
        st.caption(f"※ '{st.session_state.search_keyword}'를 포함한 기업 {len(filtered_df)}곳을 지도에 표시했습니다.")
    elif st.session_state.search_keyword.strip():
        st.warning("🛑 해당 기업이 존재하지 않습니다.")
    else:
        html(st.session_state.map_html, height=600)
        st.caption("※ 전체 기업 분포를 표시 중입니다.")

with col2:
    st.markdown("### 🧾 검색 기업 정보")

    if not matched_df.empty:
        gb = GridOptionsBuilder.from_dataframe(
            matched_df[["회사명", "도로명", "업종명", "전화번호"]]
        )
        gb.configure_selection("single", use_checkbox=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            matched_df[["회사명", "도로명", "업종명", "전화번호"]],
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=535,
            fit_columns_on_grid_load=True,
            return_mode='AS_INPUT'
        )

        selected_rows = grid_response["selected_rows"]

        if isinstance(selected_rows, list) and len(selected_rows) > 0:
            selected_company = selected_rows[0]
            if isinstance(selected_company, dict):
                selected_company_name = selected_company.get("회사명")
                if selected_company_name:
                    st.session_state.selected_company_name = selected_company_name
                    st.success(f"✅ 선택한 기업: {selected_company_name}")
                else:
                    st.warning(f"❌ '회사명' 키 없음: {list(selected_company.keys())}")
            else:
                st.error("선택된 행이 dict가 아닙니다.")
        else:
            if st.session_state.selected_company_name:
                st.info(f"🔁 최근 선택: {st.session_state.selected_company_name}")
            else:
                st.info("👈 테이블에서 기업을 선택해주세요.")
    else:
        st.info("기업을 검색해주세요.")
