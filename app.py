import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime

# -----------------------------------------------------------
# 0. 기본 설정 및 데이터 로직 (기존과 동일)
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri v1.1 Optimizer", layout="wide", page_icon="📈")
st.title("📈 위대리 v1.1 (5-Tier) 최적화 시뮬레이터")

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
    start_date = "2005-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    df = pd.DataFrame({'QQQ': qqq['Close'].squeeze(), 'TQQQ': tqqq['Close'].squeeze()}).dropna()
    df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df.index, window=1260)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    df['Weekday'] = df.index.weekday
    weekly_df = df[df['Weekday'] == 4].copy() 
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df.dropna()

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진 (초기 진입 비중 로직 적용)
# -----------------------------------------------------------
def run_simulation(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    initial_cap = params['initial_capital']
    cash = initial_cap
    shares = 0
    history = []
    max_cash_usage = initial_cap * (params['max_cash_pct'] / 100)
    
    is_first_trade = True # 초기 진입 여부 체크
    
    for date, row in sim_data.iterrows():
        price = row['TQQQ']
        prev_price = row['TQQQ_Prev']
        mkt_eval = row['Eval']
        
        # 티어 분류
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        else: tier = 'MID'
        
        action = "Hold"
        
        # 1. 초기 진입 로직
        if is_first_trade:
            # 설정된 '초기 진입 비중'만큼만 매수
            first_buy_amt = initial_cap * (params['initial_entry_pct'] / 100)
            # 단, 최대 현금 사용 한도를 넘을 수는 없음
            first_buy_amt = min(first_buy_amt, max_cash_usage)
            
            shares = int(first_buy_amt / price)
            cash -= (shares * price)
            action = f"First Buy ({params['initial_entry_pct']}%)"
            is_first_trade = False
        
        # 2. 이후 매매 로직
        else:
            current_value = shares * price
            prev_value = shares * prev_price
            diff = current_value - prev_value
            
            if diff > 0: # 수익 구간 매도
                rate = params['sell_ratios'][tier]
                qty_to_sell = int((diff * (rate / 100)) / price)
                if qty_to_sell > 0:
                    shares -= qty_to_sell
                    cash += (qty_to_sell * price)
                    action = "Sell"
            elif diff < 0: # 손실 구간 매수
                rate = params['buy_ratios'][tier]
                # 현금 사용 한도 내에서만 추가 매수 가능
                available_limit = max_cash_usage - (initial_cap - cash)
                if available_limit > 0:
                    trade_val = abs(diff) * (rate / 100)
                    actual_buy_amt = min(cash, trade_val, available_limit)
                    qty_to_buy = int(actual_buy_amt / price)
                    if qty_to_buy > 0:
                        shares += qty_to_buy
                        cash -= (qty_to_buy * price)
                        action = "Buy"
        
        history.append({'Date': date, 'Tier': tier, 'Asset': cash + (shares * price), 'Cash': cash})
    
    return pd.DataFrame(history)

# -----------------------------------------------------------
# 3. 사이드바 설정 (초기 진입 비중 UI 추가)
# -----------------------------------------------------------
st.sidebar.header("⚙️ 전략 파라미터")

if 'params' not in st.session_state:
    st.session_state.params = {
        'initial_capital': 10000, 'max_cash_pct': 100, 'initial_entry_pct': 50, # 기본값 50%
        'uhigh_cut': 10.0, 'high_cut': 7.0, 'low_cut': -5.0, 'ulow_cut': -10.0,
        's_UHIGH': 100, 'b_UHIGH': 30, 's_HIGH': 70, 'b_HIGH': 50,
        's_MID': 50, 'b_MID': 50, 's_LOW': 30, 'b_LOW': 70, 's_ULOW': 20, 'b_ULOW': 100
    }

p_start = st.sidebar.date_input("시작일", pd.to_datetime("2010-01-01"))
p_end = st.sidebar.date_input("종료일", pd.to_datetime("2025-12-31"))
p_cap = st.sidebar.number_input("초기 자본 ($)", value=st.session_state.params['initial_capital'], step=1000)

st.sidebar.divider()
st.sidebar.subheader("🔒 리스크 관리")
p_max_cash = st.sidebar.slider("최대 현금 투입 한도 (%)", 10, 100, st.session_state.params['max_cash_pct'])
# 초기 진입 비중 설정 추가 (5% 단위)
p_init_entry = st.sidebar.slider("초기 진입 비중 (%)", 0, 100, st.session_state.params['initial_entry_pct'], step=5)

st.sidebar.divider()
st.sidebar.subheader("📉 시장 평가 기준 (%)")
uh_c = st.sidebar.number_input("초고평가(UHIGH) >", value=st.session_state.params['uhigh_cut'], step=0.5, format="%.1f") / 100
h_c = st.sidebar.number_input("고평가(HIGH) >", value=st.session_state.params['high_cut'], step=0.5, format="%.1f") / 100
l_c = st.sidebar.number_input("저평가(LOW) <", value=st.session_state.params['low_cut'], step=0.5, format="%.1f") / 100
ul_c = st.sidebar.number_input("초저평가(ULOW) <", value=st.session_state.params['ulow_cut'], step=0.5, format="%.1f") / 100

st.sidebar.subheader("💰 티어별 매매율 (%)")
def tier_control_v4(label):
    st.sidebar.write(f"**[{label}]**")
    c1, c2 = st.sidebar.columns(2)
    s = c1.number_input(f"매도%", 0, 500, st.session_state.params[f's_{label}'], step=5, key=f"input_s_{label}")
    b = c2.number_input(f"매수%", 0, 500, st.session_state.params[f'b_{label}'], step=5, key=f"input_b_{label}")
    return s, b

uh_s, uh_b = tier_control_v4("UHIGH")
h_s, h_b = tier_control_v4("HIGH")
m_s, m_b = tier_control_v4("MID")
l_s, l_b = tier_control_v4("LOW")
ul_s, ul_b = tier_control_v4("ULOW")

def save_params():
    st.session_state.params.update({
        'initial_capital': p_cap, 'max_cash_pct': p_max_cash, 'initial_entry_pct': p_init_entry,
        'uhigh_cut': uh_c*100, 'high_cut': h_c*100, 'low_cut': l_c*100, 'ulow_cut': ul_c*100,
        's_UHIGH': uh_s, 'b_UHIGH': uh_b, 's_HIGH': h_s, 'b_HIGH': h_b,
        's_MID': m_s, 'b_MID': m_b, 's_LOW': l_s, 'b_LOW': l_b, 's_ULOW': ul_s, 'b_ULOW': ul_b
    })

params = {
    'initial_capital': p_cap, 'max_cash_pct': p_max_cash, 'initial_entry_pct': p_init_entry,
    'uhigh_cut': uh_c, 'high_cut': h_c, 'low_cut': l_c, 'ulow_cut': ul_c,
    'sell_ratios': {'UHIGH': uh_s, 'HIGH': h_s, 'MID': m_s, 'LOW': l_s, 'ULOW': ul_s},
    'buy_ratios': {'UHIGH': uh_b, 'HIGH': h_b, 'MID': m_b, 'LOW': l_b, 'ULOW': ul_b}
}

# -----------------------------------------------------------
# 4. 실행 및 리포트 (기존과 동일)
# -----------------------------------------------------------
if st.sidebar.button("🚀 시뮬레이션 실행", type="primary", on_click=save_params):
    with st.spinner("백테스팅 진행 중..."):
        df_weekly = get_backtest_data()
        res = run_simulation(df_weekly, p_start, p_end, params)
        
    if not res.empty:
        # 결과 리포트 및 그래프 코드 생략 (기존과 동일)
        st.success("시뮬레이션 완료!")
        # ... 성과 요약 및 그래프 출력 부분 ...
