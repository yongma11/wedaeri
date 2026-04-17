import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 디자인 (시그마2 스타일)
st.set_page_config(page_title="위대리 Quantum T-Flow v3.8", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .order-card-buy { background-color: #e6f4ea; border: 1px solid #ceead6; padding: 25px; border-radius: 12px; text-align: center; }
    .order-card-sell { background-color: #f1f3f4; border: 1px solid #dadce0; padding: 25px; border-radius: 12px; text-align: center; color: #5f6368; }
    </style>
    """, unsafe_allow_html=True)

# 2. 사이드바: 위대리 기본 설정
with st.sidebar:
    st.error("🚨 시스템 강제 초기화 (오류 해결용)")
    st.header("⚙️ 위대리 기본 설정")
    with st.container(border=True):
        st_start = st.date_input("투자 시작일", datetime.strptime("2025-12-26", "%Y-%m-%d"))
        st_cap = st.number_input("시작 원금 ($)", value=108000.00, step=1000.0)
        st_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, 40) / 100
        if st.button("🔄 데이터 갱신", use_container_width=True):
            st.cache_data.clear()

# 3. 데이터 엔진
@st.cache_data(ttl=3600)
def load_wedaeri_data():
    try:
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)
        
        results = [np.nan] * len(df)
        date_nums = df['Date'].map(pd.Timestamp.toordinal).values
        values = df['QQQ'].values
        for i in range(1260, len(df)):
            fit = np.polyfit(date_nums[i-1260:i], np.log(values[i-1260:i]), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        df['Growth'] = results
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

# 4. 시뮬레이션 엔진 (정밀 포맷팅 반영)
def run_wedaeri_sim(data, start_dt, init_cap, cash_ratio):
    settings = {
        'high_cut': 0.055, 'low_cut': -0.07,
        'sell_ratios': {'HIGH': 1.5, 'MID': 0.6, 'LOW': 0.33},
        'buy_ratios': {'HIGH': 0.5, 'MID': 0.6, 'LOW': 2.0}
    }
    sim_df = data[data['Date'] >= pd.to_datetime(start_dt)].copy()
    weekly = sim_df[sim_df['Date'].dt.weekday == 4].reset_index(drop=True)
    if weekly.empty: return pd.DataFrame()
    
    cash, shares = init_cap * cash_ratio, int((init_cap * (1-cash_ratio)) / weekly['TQQQ'].iloc[0])
    logs = []
    
    for i in range(len(weekly)):
        p, m_eval = weekly.loc[i, 'TQQQ'], weekly.loc[i, 'Eval']
        tier = 'HIGH' if m_eval >= settings['high_cut'] else ('LOW' if m_eval <= settings['low_cut'] else 'MID')
        
        action, order_qty, display_qty = "관망", 0, 0
        if i > 0:
            prev_p = weekly.loc[i-1, 'TQQQ']
            diff = (shares * p) - (shares * prev_p)
            if diff > 0: # 매도 상황
                order_qty = int(min(round((diff * settings['sell_ratios'][tier]) / p), shares))
                if order_qty > 0:
                    action = "매도"
                    display_qty = -order_qty # 매도 시 마이너스 표시
                    shares -= order_qty
                    cash += order_qty * p
            elif diff < 0: # 매수 상황
                order_qty = int(min(cash, abs(diff) * settings['buy_ratios'][tier]) / p)
                if order_qty > 0:
                    action = "매수"
                    display_qty = order_qty
                    shares += order_qty
                    cash -= order_qty * p
        
        total_asset = cash + (shares * p)
        logs.append({
            '날짜': weekly.loc[i, 'Date'].strftime('%Y-%m-%d'),
            '시장평가': f"{m_eval:.2%}",
            '티어': tier,
            '액션': action,
            '주문수량': display_qty,
            '보유수량': shares,
            '현금': round(cash, 2),
            '총자산': round(total_asset, 2),
            '수익률': f"{(total_asset/init_cap-1)*100:.2f}%" # % 기호 추가
        })
    return pd.DataFrame(logs)

# --- 메인 실행 루틴 ---
df = load_wedaeri_data()

if df.empty:
    st.error("데이터 로드 실패")
else:
    log_df = run_wedaeri_sim(df, st_start, st_cap, st_cash_ratio)
    
    if not log_df.empty:
        last = log_df.iloc[-1]
        
        st.title("🚀 TQQQ [위대리] v3.8 : 균형형 트레이딩 시스템")
        tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 전략 로직"])

        with tab1:
            # 상단 요약 바
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("시장 모드", f"{last['티어']}", last['시장평가'])
            c2.metric("TQQQ 현재가", f"${df['TQQQ'].iloc[-1]:.2f}")
            c3.metric("현재 총 자산", f"${last['총자산']:,.2f}")
            c4.metric("매매 주기", "금요일 (주간)")

            # 양방향 주문표
            st.subheader(f"📝 금주 장 마감(LOC) 주문표 ({datetime.now().strftime('%Y-%m-%d')})")
            b_col, s_col = st.columns(2)
            
            with b_col:
                buy_val = f"수량: {last['주문수량']} 주" if last['액션'] == "매수" else "대기 (신호 없음)"
                st.markdown(f'<div class="order-card-buy"><h4>📉 1. LOC 매수 주문</h4><h1 style="color:#188038;">{buy_val}</h1></div>', unsafe_allow_html=True)
            with s_col:
                sell_val = f"수량: {abs(last['주문수량'])} 주" if last['액션'] == "매도" else "대기 (신호 없음)"
                st.markdown(f'<div class="order-card-sell"><h4>📈 2. LOC 매도 주문</h4><h1>{sell_val}</h1></div>', unsafe_allow_html=True)

            st.divider()
            
            # 내 계좌 현황
            st.subheader("💰 내 계좌 현황")
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("보유 수량", f"{last['보유수량']:,} 주")
            a2.metric("보유 현금", f"${last['현금']:,.2f}")
            a3.metric("수익률", f"{last['수익률']}")
            a4.metric("평가 금액", f"${(last['보유수량']*df['TQQQ'].iloc[-1]):,.2f}")
            a5.metric("현금 비중", f"{(last['현금']/last['총자산']*100):.2f}%")

            with st.expander("📋 상세 매매 로그 (수정 가능)", expanded=True):
                # 주문수량이 0인 행은 가독성을 위해 하이픈 처리 가능하나 일단 숫자 유지
                st.dataframe(log_df.iloc[::-1], use_container_width=True)
