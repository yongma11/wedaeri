import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime

# -----------------------------------------------------------
# 0. 기본 설정 및 세션 초기화 (2010년 시작 고정)
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri v1.1 Optimizer", layout="wide", page_icon="📈")
st.title("📈 위대리 v1.1 (5-Tier) 성과 복원 시뮬레이터")

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
# 1. 데이터 처리 로직 (2010년 데이터 잘림 방지)
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_backtest_data():
    start_date = "2000-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    df = pd.concat([qqq['Close'], tqqq['Close']], axis=1).dropna()
    df.columns = ['QQQ', 'TQQQ']
    
    # 5년 지수 추세선 (2010년 데이터 보존을 위해 min_periods=1)
    df['Growth'] = np.exp(np.log(df['QQQ']).rolling(window=1260, min_periods=1).mean())
    df['Eval'] = (df['QQQ'] / df['Growth']).fillna(1.0) - 1
    
    # 금요일 종가 데이터만 추출
    df['Weekday'] = df.index.weekday
    weekly_df = df[df['Weekday'] == 4].copy() 
    
    # 주간 등락폭($) 계산을 위한 전주 종가
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df.dropna()

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진 (핵심 매매 로직 복원)
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
        
        # 티어 결정
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        else: tier = 'MID'
        
        if is_first:
            # 1. 초기 진입: 초기 자본의 설정 비율만큼 주식 매수
            entry_amt = initial_cap * (params['initial_entry_pct'] / 100)
            entry_amt = min(entry_amt, max_cash_limit)
            shares = entry_amt / price
            cash -= entry_amt
            is_first = False
        else:
            # 2. 주간 등락에 따른 위대리 매매 로직
            price_diff = price - prev_price
            
            if price_diff > 0: # 상승 시 매도
                sell_ratio = params['sell_ratios'][tier] / 100
                # "보유 수량 x 가격 상승분"의 설정 비율만큼 매도
                sell_amt = (shares * price_diff) * sell_ratio
                # 보유 주식 가치를 넘어서 팔 수 없음
                sell_amt = min(sell_amt, shares * price)
                
                shares -= (sell_amt / price)
                cash += sell_amt
                
            elif price_diff < 0: # 하락 시 매수
                buy_ratio = params['buy_ratios'][tier] / 100
                # "보유 수량 x 가격 하락분(절대값)"의 설정 비율만큼 매수
                buy_amt = abs(shares * price_diff) * buy_ratio
                
                # 현금 사용 한도 및 잔고 내에서만 매수
                current_used_cash = initial_cap - cash
                available_limit = max(0, max_cash_limit - current_used_cash)
                actual_buy = min(cash, buy_amt, available_limit)
                
                shares += (actual_buy / price)
                cash -= actual_buy
        
        total_asset = cash + (shares * price)
        history.append({'Date': date, 'Tier': tier, 'Asset': total_asset})
        
    return pd.DataFrame(history)

# -----------------------------------------------------------
# 3. 사이드바 및 결과 리포트 (통합 그래프 포함)
# -----------------------------------------------------------
st.sidebar.header("⚙️ 전략 파라미터")

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
p_max_cash = st.sidebar.slider("최대 현금 투입 한도 (%)", 10, 100, value=st.session_state.p_max_cash)
p_init_entry = st.sidebar.slider("초기 진입 비중 (%)", 0, 100, value=st.session_state.p_init_entry, step=5)

st.sidebar.divider()
uh_c = st.sidebar.number_input("초고평가(UHIGH) >", value=st.session_state.uh_c, step=0.5)
h_c = st.sidebar.number_input("고평가(HIGH) >", value=st.session_state.h_c, step=0.5)
l_c = st.sidebar.number_input("저평가(LOW) <", value=st.session_state.l_c, step=0.5)
ul_c = st.sidebar.number_input("초저평가(ULOW) <", value=st.session_state.ul_c, step=0.5)

def tier_control(label):
    st.sidebar.write(f"**[{label}]**")
    c1, c2 = st.sidebar.columns(2)
    s = c1.number_input(f"매도%", 0, 500, value=st.session_state.vals[f's_{label}'], step=5, key=f"s_{label}")
    b = c2.number_input(f"매수%", 0, 500, value=st.session_state.vals[f'b_{label}'], step=5, key=f"b_{label}")
    return s, b

uh_s, uh_b = tier_control("UHIGH")
h_s, h_b = tier_control("HIGH")
m_s, m_b = tier_control("MID")
l_s, l_b = tier_control("LOW")
ul_s, ul_b = tier_control("ULOW")

params = {
    'initial_capital': p_cap, 'max_cash_pct': p_max_cash, 'initial_entry_pct': p_init_entry,
    'uhigh_cut': uh_c/100, 'high_cut': h_c/100, 'low_cut': l_c/100, 'ulow_cut': ul_c/100,
    'sell_ratios': {'UHIGH': uh_s, 'HIGH': h_s, 'MID': m_s, 'LOW': l_s, 'ULOW': ul_s},
    'buy_ratios': {'UHIGH': uh_b, 'HIGH': h_b, 'MID': m_b, 'LOW': l_b, 'ULOW': ul_b}
}

if st.sidebar.button("🚀 시뮬레이션 실행", type="primary", on_click=update_session):
    df_weekly = get_backtest_data()
    res = run_simulation(df_weekly, p_start, p_end, params)
    
    if not res.empty:
        final_asset = res.iloc[-1]['Asset']
        total_ret = (final_asset / p_cap - 1) * 100
        days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
        cagr = ((final_asset / p_cap) ** (365 / max(1, days)) - 1) * 100
        res['Peak'] = res['Asset'].cummax()
        res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
        
        st.subheader("🚩 성과 요약 리포트")
        m1, m2, m3 = st.columns(3)
        m1.metric("총수익률", f"{total_ret:,.1f}%")
        m2.metric("CAGR (연평균)", f"{cagr:.2f}%")
        m3.metric("MDD (최대낙폭)", f"{res['DD'].min():.1f}%")

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
