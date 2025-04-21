# components/job_bu_page.py
import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from components.qa_utils import init_qa_chain, load_all_templates, load_api_key
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
def show_job_bu_page(profile):
    st.markdown("""
        <div style='padding: 10px 0px;'>
            <h1 style='margin:0; font-size:28px; display: flex; align-items: center; gap: 12px;'>
                <img src='https://raw.githubusercontent.com/seungcheoll/busan/main/image/chatbot.png' 
                     style='width: 60px; height: auto; vertical-align: middle;'>
                부산시 취업 상담 챗봇(Job-Bu)
            </h1>
        </div>
    """, unsafe_allow_html=True)

    if "llm" not in st.session_state:
        st.session_state.llm, st.session_state.retriever, st.session_state.company_df, st.session_state.map_html = init_qa_chain(load_api_key())
    if "templates" not in st.session_state:
        st.session_state.templates = load_all_templates()
    if "main_query" not in st.session_state:
        st.session_state["main_query"] = ""
    if "user_type" not in st.session_state:
        st.session_state["user_type"] = "대학생"
    if "saved_query" not in st.session_state:
        st.session_state["saved_query"] = ""
    if "saved_user_type" not in st.session_state:
        st.session_state["saved_user_type"] = ""

    def save_user_inputs():
        st.session_state["saved_user_type"] = st.session_state["user_type"]
        st.session_state["saved_query"] = st.session_state["query_input_inputbox"]

    col1, col2 = st.columns([3, 2])
    with col1:
        query = st.text_input(
            "❓ 질문으로 상담을 시작하세요!",
            value=st.session_state["main_query"],
            key="query_input_inputbox",
            placeholder="예: 연봉 3000만원 이상 선박 제조업 추천",
            on_change=save_user_inputs
        )
    with col2:
        st.selectbox("🏷️ 유형을 선택하세요!", ["대학생", "첫 취업 준비", "이직 준비"], key="user_type", on_change=save_user_inputs)

    st.session_state["query_input"] = query
    user_type = st.session_state["user_type"]

    if st.button("💬 질문 실행"):
          with st.spinner("🤖 Job-Bu가 부산 기업 정보를 검색 중입니다..."):
                try:
                formatted_template = st.session_state.templates[user_type].format(**profile)
            except KeyError:
                st.error("⚠️ 사용자 프로필이나 템플릿이 누락되었습니다. 사이드바에서 프로필을 먼저 입력해주세요.")
                return
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
            st.session_state["main_query"] = ""
            st.rerun()
    else:
        st.session_state["main_query"] = query

    selected_tabs = st.tabs(["✅ Job-Bu 답변", "📚 추천 기업 상세", "🌍 추천 기업 위치", "🔍 부산 기업 분포 및 검색"])

    with selected_tabs[0]:
        st.write(st.session_state.get("gpt_result", "🔹 Job-Bu의 응답 결과가 여기에 표시됩니다."))

    with selected_tabs[1]:
        for i, doc in enumerate(st.session_state.get("source_docs", [])):
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
            html(m._repr_html_(), height=550)
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
            search_input = st.text_input(" ", key="search_input", label_visibility="collapsed", placeholder="🔎 회사명으로 검색 (예: 현대, 시스템, 조선 등)")
        with col2:
            if search_input:
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
