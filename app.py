import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="위대리 Quantum T-Flow v3.0", layout="wide")

# [사용자 설정값]
START_DATE = '2025-12-26'
INITIAL_CAP = 108000
INITIAL_CASH_RATIO = 0.40

st.title("🚀 위대리 Quantum T-Flow v3.0")
st.subheader("3단계 시장 분류 및 정정 파라미터 적용 모델")

@st.cache_data(ttl=3600)
def get_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01")['Close'].dropna()
    return df.reset_index()

def calculate_growth(df):
    results = [np.nan] * len(df)
    date_nums = df['Date'].map(pd.Timestamp.toordinal).values
    values = df['QQQ'].values
    for i in range(1260, len(df)):
        fit = np.polyfit(date_nums[i-1260:i], np.log(values[i-1260:i]), 1)
        results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    return results

df = get_data()
df['Growth'] = calculate_growth(df)
df['Eval'] = (df['QQQ'] / df['Growth']) - 1

# 시뮬레이션 로직
sim_df = df[df['Date'] >= pd.to_datetime(START_DATE)].copy()
weekly = sim_df[sim_df['Date'].dt.weekday == 4].reset_index(drop=True)

settings = {
    'high_cut': 0.055, 'low_cut': -0.07,
    'sell_ratios': {'HIGH': 1.5, 'MID': 0.6, 'LOW': 0.33},
    'buy_ratios': {'HIGH': 0.5, 'MID': 0.6, 'LOW': 2.0}
}

cash = INITIAL_CAP * INITIAL_CASH_RATIO
shares = int((INITIAL_CAP * (1-INITIAL_CASH_RATIO)) / weekly['TQQQ'].iloc[0])
history = []

for i in range(len(weekly)):
    p = weekly.loc[i, 'TQQQ']
    m_eval = weekly.loc[i, 'Eval']
    
    tier = 'MID'
    if m_eval >= settings['high_cut']: tier = 'HIGH'
    elif m_eval <= settings['low_cut']: tier = 'LOW'
    
    if i > 0:
        prev_p = weekly.loc[i-1, 'TQQQ']
        diff = (shares * p) - (shares * prev_p)
        if diff > 0:
            q = int(min(round((diff * settings['sell_ratios'][tier]) / p), shares))
            shares -= q; cash += q*p
        elif diff < 0:
            q = int(min(cash, abs(diff) * settings['buy_ratios'][tier]) / p)
            shares += q; cash -= q*p
            
    history.append(cash + (shares * p))

weekly['Total_Asset'] = history

# 화면 출력
col1, col2, col3 = st.columns(3)
col1.metric("현재 자산", f"${weekly['Total_Asset'].iloc[-1]:,.2f}")
col2.metric("누적 수익률", f"{(weekly['Total_Asset'].iloc[-1]/INITIAL_CAP-1)*100:.2f}%")
col3.metric("현재 시장 평가", f"{df['Eval'].iloc[-1]:.2%}")

st.line_chart(weekly.set_index('Date')['Total_Asset'])
st.dataframe(weekly[['Date', 'TQQQ', 'Eval', 'Total_Asset']].tail(10))
