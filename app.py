import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime

# -----------------------------------------------------------
# 0. 기본 설정 및 세션 초기화
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri v1.1 Optimizer", layout="wide", page_icon="📈")
st.title("📈 위대리 v1.1 (5-Tier) 최적화 시뮬레이터")

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
# 1. 데이터 처리 로직 (수정: dropna 범위 제한)
# -----------------------------------------------------------
def calculate_growth_curve_fast(series, window=1260):
    log_series = np.log(series)
    # 데이터가 부족해도 평균을 낼 수 있도록 min_periods=1 설정
    return np.exp(log_series.rolling(window=window, min_periods=1).mean())

@st.cache_data(ttl=3600)
def get_backtest_data():
    # 충분한 데이터 확보를 위해 2000년부터 가져옴
    start_date = "2000-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # 핵심 데이터 합치기
    df = pd.concat([qqq['Close'], tqqq['Close']], axis=1).dropna()
    df.columns = ['QQQ', 'TQQQ']
    
    # 추세선 및 평가 지표 계산
    df['Growth'] = calculate_growth_curve_fast(df['QQQ'], window=1260)
    # Growth가 NaN인 초기 데이터도 0(중립)으로 처리하여 날짜가 잘리지 않게 함
    df['Eval'] = (df['QQQ'] / df['Growth']).fillna(1.0) - 1
    
    # 주간 데이터(금요일) 추출
    df['Weekday'] = df.index.weekday
    weekly_df = df[df['Weekday'] == 4].copy() 
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    
    # TQQQ_Prev가 없는 첫 줄만 제외하고 나머지는 유지
    return weekly_df.iloc[1:]

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진 및 3. 사이드바 (동일)
# -----------------------------------------------------------
def run_simulation(df, start_dt, end_dt, params):
    # 사용자가 선택한 날짜 범위로 필터링
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame()
    
    initial_cap = params['initial_capital']
    cash = initial_cap
    shares = 0
    history = []
    max_cash_usage = initial_cap * (params['max_cash_pct'] / 100)
    is_first_trade = True 
    
    for date, row in sim_data.iterrows():
        price = row['TQQQ']
        prev_price = row['TQQQ_Prev']
        mkt_eval = row['Eval']
        
        # 티어 분류 로직
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        else: tier = 'MID'
        
        if is_first_trade:
            first_buy_amt = initial_cap * (params['initial_entry_pct'] / 100)
            shares = int(min(first_buy_amt, max_cash_usage) / price)
            cash -= (shares * price)
            is_first_trade = False
        else:
            diff = (shares * price) - (shares * prev_price)
            if diff > 0:
                rate = params['sell_ratios'][tier]
                qty = int((diff * (rate / 100)) / price)
                if qty > 0:
                    shares -= qty
                    cash += (qty * price)
            elif diff < 0:
                rate = params['buy_ratios'][tier]
                available = max_cash_usage - (initial_cap - cash)
                if available > 0:
                    amt = min(cash, abs(diff) * (rate / 100), available)
                    qty = int(amt / price)
                    if qty > 0:
                        shares += qty
                        cash -= (qty * price)
        
        history.append({'Date': date, 'Tier': tier, 'Asset': cash + (shares * price)})
    return pd.DataFrame(history)

# 사이드바 및 버튼 로직 (동일)
def update_session():
    st.session_state.p_start = p_start
    st.session_state.p_end = p_end
    st.session_state.p_cap = p_cap
    st.session_state.p_max_cash = p_max_cash
    st.session_state.p_init_entry = p_init_entry
    st.session_state.uh_c = uh_c
    st.session_state.h_c = h_c
    st.session_state.l_c = l_c
    st.session_state.ul_c = ul_c
    st.session_state.vals.update({
        's_UHIGH': uh_s, 'b_UHIGH': uh_b, 's_HIGH': h_s, 'b_HIGH': h_b,
        's_MID': m_s, 'b_MID': m_b, 's_LOW': l_s, 'b_LOW': l_b, 's_ULOW': ul_s, 'b_ULOW': ul_b
    })

p_start = st.sidebar.date_input("시작일", value=st.session_state.p_start)
p_end = st.sidebar.date_input("종료일", value=st.session_state.p_end)
p_cap = st.sidebar.number_input("초기 자본 ($)", value=st.session_state.p_cap, step=1000)
st.sidebar.divider()
p_max_cash = st.sidebar.slider("최대 현금 투입 한도 (%)", 10, 100, value=st.session_state.p_max_cash)
p_init_entry = st.sidebar.slider("초기 진입 비중 (%)", 0, 100, value=st.session_state.p_init_entry, step=5)
st.sidebar.divider()
uh_c = st.sidebar.number_input("초고평가(UHIGH) >", value=st.session_state.uh_c, step=0.5)
h_c = st.sidebar.number_input("고평가(HIGH) >", value=st.session_state.h_c, step=0.5)
l_c = st.sidebar.number_input("저평가(LOW) <", value=st.session_state.l_c, step=0.5)
ul_c = st.sidebar.number_input("초저평가(ULOW) <", value=st.session_state.ul_c, step=0.5)

def tier_control_v5(label):
    st.sidebar.write(f"**[{label}]**")
    c1, c2 = st.sidebar.columns(2)
    s = c1.number_input(f"매도%", 0, 500, value=st.session_state.vals[f's_{label}'], step=5, key=f"s_{label}")
    b = c2.number_input(f"매수%", 0, 500, value=st.session_state.vals[f'b_{label}'], step=5, key=f"b_{label}")
    return s, b

uh_s, uh_b = tier_control_v5("UHIGH")
h_s, h_b = tier_control_v5("HIGH")
m_s, m_b = tier_control_v5("MID")
l_s, l_b = tier_control_v5("LOW")
ul_s, ul_b = tier_control_v5("ULOW")

params = {
    'initial_capital': p_cap, 'max_cash_pct': p_max_cash, 'initial_entry_pct': p_init_entry,
    'uhigh_cut': uh_c/100, 'high_cut': h_c/100, 'low_cut': l_c/100, 'ulow_cut': ul_c/100,
    'sell_ratios': {'UHIGH': uh_s, 'HIGH': h_s, 'MID': m_s, 'LOW': l_s, 'ULOW': ul_s},
    'buy_ratios': {'UHIGH': uh_b, 'HIGH': h_b, 'MID': m_b, 'LOW': l_b, 'ULOW': ul_b}
}

# -----------------------------------------------------------
# 4. 결과 출력
# -----------------------------------------------------------
if st.sidebar.button("🚀 시뮬레이션 실행", type="primary", on_click=update_session):
    df_weekly = get_backtest_data()
    res = run_simulation(df_weekly, p_start, p_end, params)
    
    if not res.empty:
        final_asset = res.iloc[-1]['Asset']
        total_ret = (final_asset / p_cap - 1) * 100
        res['Peak'] = res['Asset'].cummax()
        res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
        
        st.subheader("🚩 성과 요약 리포트")
        m1, m2, m3 = st.columns(3)
        m1.metric("총수익률", f"{total_ret:,.1f}%")
        m2.metric("CAGR", f"{((final_asset/p_cap)**(365/max(1,(pd.to_datetime(p_end)-pd.to_datetime(p_start)).days))-1)*100:.2f}%")
        m3.metric("MDD", f"{res['DD'].min():.1f}%")

        st.subheader("📈 통합 그래프 (자산 & MDD)")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(res['Date'], res['Asset'], color='#1E88E5', lw=2)
        ax1.set_yscale('log')
        ax2 = ax1.twinx()
        ax2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.2)
        ax2.set_ylim(-100, 5)
        st.pyplot(fig)

        st.subheader("📅 연도별 성과 요약")
        res['Year'] = res['Date'].dt.year
        y_perf = []
        for year, group in res.groupby('Year'):
            ret = (group.iloc[-1]['Asset'] / group.iloc[0]['Asset'] - 1) * 100
            y_perf.append({'연도': year, '수익률': f"{ret:.1f}%", 'MDD': f"{group['DD'].min():.1f}%"})
        st.table(pd.DataFrame(y_perf).set_index('연도').T)
