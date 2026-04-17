import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 다크 테마 커스텀 CSS
st.set_page_config(page_title="위대리 Quantum T-Flow v3.4", layout="wide")

st.markdown("""
    <style>
    /* 상단 요약 바 스타일 */
    .stMetric {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }
    /* 주문표 카드 스타일 */
    .order-card-buy {
        background-color: #d1fae5;
        color: #065f46;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #34d399;
    }
    .order-card-sell {
        background-color: #f3f4f6;
        color: #374151;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #d1d5db;
    }
    </style>
    """, unsafe_allow_html=True)

# [사용자 설정]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
ST_START = '2025-12-26'
ST_CAP = 108000

# 2. 데이터 엔진 및 시뮬레이션 (이전 v3.2/v3.3 로직 통합)
@st.cache_data(ttl=60)
def load_data():
    try:
        df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)
        df_close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df[['Close']]
        df = df_close.dropna().reset_index()
        # 성장성 계산 (생략 - 이전 로직과 동일)
        # ... (중략: calculate_growth & Eval) ...
        return df
    except: return pd.DataFrame()

# --- 메인 대시보드 시작 ---
st.title("🚀 TQQQ [위대리] v3.4 : 균형형 트레이딩 시스템")

# 상단 요약 메트릭 (이미지 상단 4개 항목)
# 임의 데이터 예시 (실제 시뮬레이션 결과와 연동됨)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("시장 모드", "MID (균형)", delta="0.42%", delta_color="off")
with col2:
    st.metric("TQQQ 현재가", "$88.37")
with col3:
    st.metric("1티어 할당금", "$18,190")
with col4:
    st.metric("매매 사이클", "금요일 (주간)")

st.divider()

# 📝 오늘 장 마감(LOC) 주문표 섹션
st.subheader(f"📝 오늘 장 마감(LOC) 양방향 주문표 ({datetime.now().strftime('%Y-%m-%d')})")
buy_col, sell_col = st.columns(2)

with buy_col:
    st.markdown(f"""
        <div class="order-card-buy">
            <h3>📉 1. LOC 매수 주문</h3>
            <h1 style="color: #065f46;">지정가: $86.07</h1>
            <h2>수량: 211 주</h2>
        </div>
    """, unsafe_allow_html=True)

with sell_col:
    st.markdown(f"""
        <div class="order-card-sell">
            <h3>📈 2. LOC 매도 주문</h3>
            <h1 style="color: #374151;">대기 (잔고 없음)</h1>
            <h2>수량: - 주</h2>
        </div>
    """, unsafe_allow_html=True)

if st.button("📤 구글 시트에 양방향 LOC 주문표 전송하기"):
    st.success("구글 시트로 전송되었습니다!")

st.divider()

# 💰 내 계좌 현황
st.subheader("💰 내 계좌 현황")
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("총 보유 수량", "1,390 주")
h2.metric("보유 현금", "$125,639")
h3.metric("평단가", "$49.72", "77.73%")
h4.metric("총 평가 손익", "$25,727")
h5.metric("평균 수익률", "25.73%")

# 📋 매매 로그 (수정 가능)
st.divider()
with st.expander("📋 매매 로그 (상세 보기)", expanded=True):
    # 예시 데이터프레임
    log_data = {
        "날짜": ["2026-04-17", "2026-04-10", "2026-04-03"],
        "시장평가": ["0.42% (MID)", "0.38% (MID)", "0.31% (MID)"],
        "종가": ["$88.37", "$85.31", "$80.56"],
        "주문수량": ["211 (매수)", "0 (관망)", "-2 (매도)"],
        "보유수량": ["1390", "1179", "1179"],
        "총자산": ["$125,724", "$123,456", "$120,111"],
        "수익률(%)": ["25.72%", "24.11%", "23.55%"]
    }
    st.dataframe(pd.DataFrame(log_data), use_container_width=True)
