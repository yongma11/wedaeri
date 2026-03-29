import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="위대리 Quantum T-Flow v3.0", layout="wide", initial_sidebar_state="expanded")

# 커스텀 CSS (이전의 깔끔한 다크모드 스타일 유지)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# 2. 사이드바 - 핵심 파라미터 설정 (이전 인터페이스 복구)
with st.sidebar:
    st.header("⚙️ 전략 파라미터")
    st.info("3단계 시장 분류 로직 적용 중")
    
    start_date = st.date_input("매매 시작일", datetime.strptime("2025-12-26", "%Y-%m-%d"))
    initial_cap = st.number_input("초기 자본 ($)", value=108000, step=1000)
    cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, 40) / 100
    
    st.divider()
    st.subheader("📊 티어 구간 설정")
    high_cut = st.slider("고평가 기준 (Eval >)", 0.0, 0.15, 0.055, format="%.3f")
    low_cut = st.slider("저평가 기준 (Eval <)", -0.15, 0.0, -0.07, format="%.3f")
    
    st.divider()
    st.subheader("⚖️ 매매 비율 (Ratios)")
    col_s, col_b = st.columns(2)
    with col_s:
        s_high = st.number_input("고평가 매도", value=1.5, step=0.1)
        s_mid = st.number_input("중간 매도", value=0.6, step=0.1)
        s_low = st.number_input("저평가 매도", value=0.33, step=0.1)
    with col_b:
        b_high = st.number_input("고평가 매수", value=0.5, step=0.1)
        b_mid = st.number_input("중간 매수", value=0.6, step=0.1)
        b_low = st.number_input("저평가 매수", value=2.0, step=0.1)

# 3. 데이터 엔진 (성장성 및 Eval 계산)
@st.cache_data(ttl=3600)
def load_market_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)
    df_close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df[['Close']]
    df = df_close.dropna().reset_index()
    if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)
    
    # 25년 로그 회귀 엔진
    results = [np.nan] * len(df)
    date_nums = df['Date'].map(pd.Timestamp.toordinal).values
    values = df['QQQ'].values
    for i in range(1260, len(df)):
        fit = np.polyfit(date_nums[i-1260:i], np.log(values[i-1260:i]), 1)
        results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    
    df['Growth'] = results
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    return df

# 4. 구글 시트 연결 함수
def connect_gspread():
    try:
        creds_info = st.secrets["gcp_service_account"]
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        return gspread.authorize(creds)
    except: return None

# --- 메인 대시보드 구현 ---
df = load_market_data()
st.title("🚀 위대리 Quantum T-Flow v3.0")

# 상단 요약 지표 (Metrics)
sim_df = df[df['Date'] >= pd.to_datetime(start_date)].copy()
weekly = sim_df[sim_df['Date'].dt.weekday == 4].reset_index(drop=True)

if not weekly.empty:
    # 시뮬레이션 실행 로직
    cash, shares = initial_cap * cash_ratio, int((initial_cap * (1 - cash_ratio)) / weekly['TQQQ'].iloc[0])
    history = []
    
    for i in range(len(weekly)):
        p, m_eval = weekly.loc[i, 'TQQQ'], weekly.loc[i, 'Eval']
        tier = 'MID'
        if m_eval >= high_cut: tier = 'HIGH'
        elif m_eval <= low_cut: tier = 'LOW'
        
        s_ratio = {'HIGH': s_high, 'MID': s_mid, 'LOW': s_low}[tier]
        b_ratio = {'HIGH': b_high, 'MID': b_mid, 'LOW': b_low}[tier]
        
        if i > 0:
            diff = (shares * p) - (shares * weekly.loc[i-1, 'TQQQ'])
            if diff > 0:
                q = int(min(round((diff * s_ratio) / p), shares))
                shares -= q; cash += q * p
            elif diff < 0:
                q = int(min(cash, abs(diff) * b_ratio) / p)
                shares += q; cash -= q * p
        history.append(cash + (shares * p))
    
    weekly['Total_Asset'] = history
    cur_asset = weekly['Total_Asset'].iloc[-1]
    profit_pct = (cur_asset / initial_cap - 1) * 100
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 총 자산", f"${cur_asset:,.2f}")
    m2.metric("누적 수익률", f"{profit_pct:.2f}%", f"{(profit_pct/len(weekly)*52):.1f}% (연)")
    m3.metric("현재 Eval (과열도)", f"{df['Eval'].iloc[-1]:.2%}")
    m4.metric("보유 수량", f"{shares} 주")

    # 메인 차트 (Asset Growth)
    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly['Date'], y=weekly['Total_Asset'], name="자산 성장", fill='tozeroy', line=dict(color='#00FFCC')))
    fig.update_layout(title="퀀텀 T-Flow 자산 성장 곡선", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # 하단 2분할 레이아웃 (주문표 & 데이터 테이블)
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📋 실시간 구글 시트 주문 현황")
        gc = connect_gspread()
        if gc:
            try:
                ws = gc.open_by_key("1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI").worksheet("위대리")
                data = ws.get('L4:O4')
                st.table(pd.DataFrame(data, columns=["액션", "방법", "예상가", "주문수량"]))
            except: st.error("시트 데이터를 가져올 수 없습니다.")
        else: st.warning("Secrets 설정을 확인해 주세요.")

    with col_right:
        st.subheader("📈 최근 5주 매매 기록")
        st.dataframe(weekly[['Date', 'TQQQ', 'Eval', 'Total_Asset']].tail(5).sort_index(ascending=False), use_container_width=True)

else:
    st.error("설정한 시작일 이후의 데이터가 존재하지 않습니다.")
