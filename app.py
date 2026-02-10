import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 0. 기본 설정 및 세션 초기화
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri v1.1 Optimizer", layout="wide", page_icon="📈")
st.title("📈 위대리 v1.1 (Precise Growth) 시뮬레이터")

if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.p_start = datetime(2010, 1, 1)
    st.session_state.p_end = datetime(2025, 12, 31)
    st.session_state.p_cap = 10000
    st.session_state.p_max_cash = 100
    st.session_state.p_init_entry = 50
    st.session_state.uh_c = 10.0
    st.session_state.h_c = 7.0
    st.session_state.l_c = -5.0
    st.session_state.ul_c = -10.0
    st.session_state.vals = {
        's_UHIGH': 100, 'b_UHIGH': 30, 's_HIGH': 70, 'b_HIGH': 50,
        's_MID': 50, 'b_MID': 50, 's_LOW': 30, 'b_LOW': 70, 's_ULOW': 20, 'b_ULOW': 100
    }

# -----------------------------------------------------------
# 1. 정밀 추세선 계산 (제공해주신 샘플 코드 로직 적용)
# -----------------------------------------------------------
def calculate_growth_curve_precise(series, dates, window=1260):
    """
    제공해주신 샘플 코드의 핵심 로직입니다.
    로그 변환 후 선형 회귀를 통해 지수 성장 곡선(Growth)을 산출합니다.
    """
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]
        x_train = date_nums[i-window : i]
        if np.any(y_train <= 0) or np.isnan(y_train).any(): continue
        try:
            # 로그 선형 회귀: log(y) = ax + b
            fit = np.polyfit(x_train, np.log(y_train), 1)
            # 현재 시점의 날짜(ordinal)를 대입하여 예측값 산출
            pred_log = fit[1] + fit[0] * date_nums[i]
            results[i] = np.exp(pred_log)
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=3600)
def get_backtest_data():
    # 2010년 시점의 5년 추세선을 위해 2005년 이전인 2004년부터 데이터를 가져옵니다.
    start_date = "2004-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    df = pd.concat([qqq['Close'], tqqq['Close']], axis=1).dropna()
    df.columns = ['QQQ', 'TQQQ']
    
    # [핵심] 정밀 로직 적용
    df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df.index, window=1260)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    
    # 주간 데이터(금요일 종가) 필터링
    df['Weekday'] = df.index.weekday
    weekly_df = df[df['Weekday'] == 4].copy() 
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진 및 사이드바 로직
# -----------------------------------------------------------
def run_simulation(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame()
    
    initial_cap = params['initial_capital']
    cash = initial_cap
    shares = 0
    history = []
    max_cash_limit = initial_cap * (params['max_cash_pct'] / 100)
    is_first = True
    
    for date, row in sim_data.iterrows():
        price = row['TQQQ']
        prev_price = row['TQQQ_Prev']
        mkt_eval = row['Eval']
        
        # Growth 값이 아직 계산되지 않은 초기 시점 예외 처리
        if np.isnan(mkt_eval): mkt_eval = 0.0
        
        # 티어 분류
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        else: tier = 'MID'
        
        if is_first:
            # 초기 진입
            entry_amt = initial_cap * (params['initial_entry_pct'] / 100)
            shares = min(entry_amt, max_cash_limit) / price
            cash -= (shares * price)
            is_first = False
        else:
            # 주간 등락폭($) 기반 위대리 매매
            diff = (shares * price) - (shares * prev_price)
            if diff > 0: # 상승 시 매도
                sell_amt = diff * (params['sell_ratios'][tier] / 100)
                shares -= (sell_amt / price)
                cash += sell_amt
            elif diff < 0: # 하락 시 매수
                buy_amt = abs(diff) * (params['buy_ratios'][tier] / 100)
                available = max(0, max_cash_limit - (initial_cap - cash))
                actual_buy = min(cash, buy_amt, available)
                shares += (actual_buy / price)
                cash -= actual_buy
        
        history.append({'Date': date, 'Tier': tier, 'Asset': cash + (shares * price)})
    return pd.DataFrame(history)

# (이하 사이드바 UI 및 결과 리포트 출력 코드는 이전과 동일하게 구성됩니다.)
# update_session, st.sidebar.number_input, st.sidebar.button 등 생략
