import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 1. 데이터 처리 로직 (수정됨: 2010년 결과를 위해 2005년부터 예열)
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_backtest_data():
    # 2010년의 5년 전인 2005년 데이터가 '반드시' 필요하므로 2004년부터 불러옵니다.
    start_date = "2004-01-01" 
    end_date = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    df = pd.concat([qqq['Close'], tqqq['Close']], axis=1).dropna()
    df.columns = ['QQQ', 'TQQQ']
    
    # [수정] 제공해주신 정밀 회귀 로직 적용
    # 이 함수가 실행되면 2004+5년 = 2009년경부터 이미 Growth 값이 생성됩니다.
    df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df.index, window=1260)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    
    # 주간 데이터 추출
    df['Weekday'] = df.index.weekday
    weekly_df = df[df['Weekday'] == 4].copy() 
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    
    # 2010년 이전 데이터가 포함된 상태로 반환 (나중에 엔진에서 잘라씀)
    return weekly_df

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진 (수정됨: 시작일 필터링 최적화)
# -----------------------------------------------------------
def run_simulation(df, start_dt, end_dt, params):
    # 여기서 비로소 사용자가 원하는 2010년으로 데이터를 자릅니다.
    # 이미 위에서 2005년부터 계산된 Growth 값이 있으므로 2010년 데이터는 '안전'합니다.
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    
    if sim_data.empty:
        st.error("데이터가 비어있습니다. 예열 기간 혹은 시작일을 확인하세요.")
        return pd.DataFrame()

    # ... (이후 매매 로직은 이전과 동일) ...
