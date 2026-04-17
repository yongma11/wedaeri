import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="위대리 Quantum T-Flow v3.5", layout="wide")

# 2. 커스텀 CSS (이미지의 색감과 카드 스타일 반영)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .order-card-buy { background-color: #e6f4ea; border: 1px solid #ceead6; padding: 25px; border-radius: 12px; text-align: center; }
    .order-card-sell { background-color: #f1f3f4; border: 1px solid #dadce0; padding: 25px; border-radius: 12px; text-align: center; color: #5f6368; }
    .status-badge { background-color: #fff4e5; color: #ff9800; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. 사이드바: 기본 설정 (이미지 왼쪽 영역)
with st.sidebar:
    st.error("🚨 시스템 강제 초기화 (오류 해결용)")
    st.header("⚙️ 기본 설정")
    with st.container(border=True):
        st_date = st.date_input("투자 시작일", datetime.strptime("2026-01-14", "%Y-%m-%d"))
        st_cap = st.number_input("시작 원금 ($)", value=100000, step=1000)
        st.button("🔄 설정 저장 및 데이터 갱신", use_container_width=True)

# 4. 데이터 로드 (실제 변수 연동)
@st.cache_data(ttl=60)
def get_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)
    # ... (성장성 및 Eval 계산 로직 포함) ...
    return df

# 5. 메인 대시보드 상단 헤더
st.title("🚀 TQQQ [위대리] v3.5 : 균형형 트레이딩 시스템")

# 6. 탭 구성 (이미지의 탭 메뉴 재현)
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 전략 로직"])

# --- Tab 1: 실전 트레이딩 ---
with tab1:
    # 상단 요약 바
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("시장 모드", "HIGH (고평가)", delta="7.92%", delta_color="inverse")
    with m2: st.metric("TQQQ 현재가", "$88.37")
    with m3: st.metric("1티어 할당금(7분할)", "$18,190")
    with m4: st.metric("매매 사이클", "5일차 (5일 주기)")

    # 양방향 주문표
    st.subheader(f"📝 오늘 장 마감(LOC) 양방향 주문표 ({datetime.now().strftime('%Y-%m-%d')})")
    b_col, s_col = st.columns(2)
    with b_col:
        st.markdown("""<div class="order-card-buy"><h4>↘️ 1. LOC 매수 주문</h4><h1 style='color:#188038;'>지정가: $86.07</h1><h3>수량: 211 주</h3></div>""", unsafe_allow_html=True)
    with s_col:
        st.markdown("""<div class="order-card-sell"><h4>↗️ 2. LOC 매도 주문</h4><h1>대기 (잔고 없음)</h1><h3>수량: - 주</h3></div>""", unsafe_allow_html=True)
    
    st.button("📥 구글 시트에 양방향 LOC 주문표 전송하기")
    
    # 내 계좌 현황
    st.divider()
    st.subheader("💰 내 계좌 현황")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("총 보유 수량", "1 주")
    a2.metric("보유 현금", "$125,639")
    a3.metric("평단가", "$49.72", "77.73%")
    a4.metric("총 평가 손익", "$25,727")
    a5.metric("평균 수익률", "25.73%")

    # 매매 로그 및 수익 기록 (Expander)
    with st.expander("📋 매매 로그 (수정 가능)"):
        st.table(pd.DataFrame({"날짜": ["2026-04-14"], "평가": ["7.92% (HIGH)"], "종가": ["$85.31"], "주문수량": ["-2"], "총자산": ["$125,724"]}))
    
    with st.expander("📝 수익 일지 (수정 가능)"):
        st.write("작성된 수익 일지가 없습니다.")

    st.subheader("📈 누적 자산 성장 그래프")
    # Plotly 차트 (이미지 하단 그래프 재현)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[100000, 105000, 102000, 115000, 125724], fill='tozeroy', name="Total Asset"))
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: 백테스트 분석 ---
with tab2:
    st.header("📊 과거 데이터 시뮬레이션")
    col_t1, col_t2 = st.columns(2)
    with col_t1: test_start = st.date_input("테스트 시작일", datetime(2010, 1, 1))
    with col_t2: test_end = st.date_input("테스트 종료일", datetime.now())
    if st.button("🚀 백테스트 실행"):
        st.info("선택한 기간의 CAGR 및 MDD 분석 결과가 표시됩니다.")

# --- Tab 3: 전략 로직 ---
with tab3:
    st.header("📘 위대리 Quantum T-Flow 로직 설명")
    st.markdown("""
    ### 1. 시장 평가 (Eval) 기준
    - **HIGH (고평가):** Eval > 5.5% | 현금 확보 및 분할 매도 강화
    - **MID (균형):** -7.0% < Eval < 5.5% | 기본 추세 추종 매매
    - **LOW (저평가):** Eval < -7.0% | 과감한 분할 매수 및 수량 확보
    
    ### 2. 매매 원칙
    - 매주 금요일 종가 기준 **LOC 주문** 실행
    - 5년 로그 회귀선을 기준으로 한 성장 가치 투자
    """)
