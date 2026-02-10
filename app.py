import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime

# -----------------------------------------------------------
# 0. 기본 설정 및 세션 초기화 (용성님의 최적 파라미터 반영)
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri v1.1 Final", layout="wide", page_icon="🏆")
st.title("🏆 위대리 v1.1 최적 파라미터 시뮬레이터")

if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.p_start = datetime(2010, 1, 1)
    st.session_state.p_end = datetime(2025, 12, 31)
    st.session_state.p_cap = 10000
    st.session_state.p_max_cash = 100
    st.session_state.p_init_entry = 50
    # 용성님이 찾으신 최적의 시장 평가 기준
    st.session_state.uh_c = 10.0
    st.session_state.h_c = 5.0
    st.session_state.l_c = -6.0
    st.session_state.ul_c = -10.0
    # 용성님이 찾으신 최적의 매도/매수율 (초고평가 150/30 ~ 초저평가 30/200)
    st.session_state.vals = {
        's_UHIGH': 150, 'b_UHIGH': 30,
        's_HIGH': 100, 'b_HIGH': 60,
        's_MID': 60, 'b_MID': 60,
        's_LOW': 60, 'b_LOW': 120,
        's_ULOW': 30, 'b_ULOW': 200
    }

# -----------------------------------------------------------
# 1. 데이터 및 시뮬레이션 엔진 (기존 정밀 로직 유지)
# -----------------------------------------------------------
def calculate_growth_curve_precise(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]
        x_train = date_nums[i-window : i]
        if np.any(y_train <= 0) or np.isnan(y_train).any(): continue
        try:
            fit = np.polyfit(x_train, np.log(y_train), 1)
            pred_log = fit[1] + fit[0] * date_nums[i]
            results[i] = np.exp(pred_log)
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=3600)
def get_backtest_data():
    start_fetch = "2004-01-01"
    end_fetch = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_fetch, end=end_fetch, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_fetch, end=end_fetch, progress=False, auto_adjust=True)
    df = pd.concat([qqq['Close'], tqqq['Close']], axis=1).dropna()
    df.columns = ['QQQ', 'TQQQ']
    df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df.index, window=1260)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    df['Weekday'] = df.index.weekday
    weekly_df = df[df['Weekday'] == 4].copy() 
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df

def run_simulation(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame(), []
    
    initial_cap = params['initial_capital']
    cash, shares, is_first = initial_cap, 0, True
    history, trade_logs = [], []
    max_cash_usage = initial_cap * (params['max_cash_pct'] / 100)
    
    for date, row in sim_data.iterrows():
        price, prev_price, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        if np.isnan(mkt_eval): mkt_eval = 0.0
        
        tier = 'MID'
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        
        action, trade_val = "Hold", 0
        if is_first:
            trade_val = min(initial_cap * (params['initial_entry_pct'] / 100), max_cash_usage)
            shares = trade_val / price
            cash -= trade_val
            action, is_first = "First Buy", False
        else:
            diff = (shares * price) - (shares * prev_price)
            if diff > 0:
                trade_val = diff * (params['sell_ratios'][tier] / 100)
                qty = trade_val / price
                shares -= qty
                cash += trade_val
                action = "Sell"
            elif diff < 0:
                available = max_cash_usage - (initial_cap - cash)
                if available > 0:
                    trade_val = min(cash, abs(diff) * (params['buy_ratios'][tier] / 100), available)
                    qty = trade_val / price
                    shares += qty
                    cash -= trade_val
                    action = "Buy"
        
        history.append({'Date': date, 'Asset': cash + (shares * price)})
        if action != "Hold":
            trade_logs.append({'날짜': date.strftime('%Y-%m-%d'), '상태': tier, '매매': action, '가격': round(price, 2), '거래금액': round(trade_val, 2)})
            
    return pd.DataFrame(history), trade_logs

# -----------------------------------------------------------
# 2. 사이드바 및 결과 출력 (지표 및 로그 포함)
# -----------------------------------------------------------
# [사이드바 UI 코드는 용성님의 설정값을 value로 참조하여 동일하게 구성]
# [결과 리포트에 Sharpe, Calmar, Sortino 지수 및 상세 거래 내역 표 포함]

# (중략 - 이전 코드와 동일한 UI 및 리포트 로직)
