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

# 처음 접속 시 기본값 설정 (시작일 2015년으로 변경)
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.p_start = datetime(2015, 1, 1)
    st.session_state.p_end = datetime(2025, 12, 31)
    st.session_state.p_cap = 10000
    st.session_state.p_max_cash = 100
    st.session_state.p_init_entry = 50
    st.session_state.uh_c = 10.0
    st.session_state.h_c = 7.0
    st.session_state.l_c = -5.0
    st.session_state.ul_c = -10.0
    # 티어별 기본값
    st.session_state.vals = {
        's_UHIGH': 100, 'b_UHIGH': 30,
        's_HIGH': 70, 'b_HIGH': 50,
        's_MID': 50, 'b_MID': 50,
        's_LOW': 30, 'b_LOW': 70,
        's_ULOW': 20, 'b_ULOW': 100
    }

# -----------------------------------------------------------
# 1. 데이터 처리 로직 (기존과 동일)
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
# 2. 시뮬레이션 엔진
# -----------------------------------------------------------
def run_simulation(df, start_dt, end_dt, params):
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
        
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        else: tier = 'MID'
        
        if is_first_trade:
            first_buy_amt = initial_cap * (params['initial_entry_pct'] / 100)
            first_buy_amt = min(first_buy_amt, max_cash_usage)
            shares = int(first_buy_amt / price)
            cash -= (shares * price)
            is_first_trade = False
        else:
            current_value = shares * price
            prev_value = shares * prev_price
            diff = current_value - prev_value
            
            if diff > 0:
                rate = params['sell_ratios'][tier]
                qty_to_sell = int((diff * (rate / 100)) / price)
                if qty_to_sell > 0:
                    shares -= qty_to_sell
                    cash += (qty_to_sell * price)
            elif diff < 0:
                rate = params['buy_ratios'][tier]
                available_limit = max_cash_usage - (initial_cap - cash)
                if available_limit > 0:
                    trade_val = abs(diff) * (rate / 100)
                    actual_buy_amt = min(cash, trade_val, available_limit)
                    qty_to_buy = int(actual_buy_amt / price)
                    if qty_to_buy > 0:
                        shares += qty_to_buy
                        cash -= (qty_to_buy * price)
        
        history.append({'Date': date, 'Tier': tier, 'Asset': cash + (shares * price)})
    return pd.DataFrame(history)

# -----------------------------------------------------------
# 3. 사이드바 (설정 유지 로직 강화)
# -----------------------------------------------------------
st.sidebar.header("⚙️ 전략 파라미터")

# 사이드바 위젯들 - session_state 직접 연결
p_start = st.sidebar.date_input("시작일", value=st.session_state.p_start)
p_end = st.sidebar.date_input("종료일", value=st.session_state.p_end)
p_cap = st.sidebar.number_input("초기 자본 ($)", value=st.session_state.p_cap, step=1000)

st.sidebar.divider()
st.sidebar.subheader("🔒 리스크 관리")
p_max_cash = st.sidebar.slider("최대 현금 투입 한도 (%)", 10, 100, value=st.session_state.p_max_cash)
p_init_entry = st.sidebar.slider("초기 진입 비중 (%)", 0, 100, value=st.session_state.p_init_entry, step=5)

st.sidebar.divider()
st.sidebar.subheader("📉 시장 평가 기준 (%)")
uh_c = st.sidebar.number_input("초고평가(UHIGH) >", value=st.session_state.uh_c, step=0.5)
h_c = st.sidebar.number_input("고평가(HIGH) >", value=st.session_state.h_c, step=0.5)
l_c = st.sidebar.number_input("저평가(LOW) <", value=st.session_state.l_c, step=0.5)
ul_c = st.sidebar.number_input("초저평가(ULOW) <", value=st.session_state.ul_c, step=0.5)

st.sidebar.subheader("💰 티어별 매매율 (%)")
def tier_control_final(label):
    st.sidebar.write(f"**[{label}]**")
    c1, c2 = st.sidebar.columns(2)
    s = c1.number_input(f"매도%", 0, 500, value=st.session_state.vals[f's_{label}'], step=5, key=f"s_in_{label}")
    b = c2.number_input(f"매수%", 0, 500, value=st.session_state.vals[f'b_{label}'], step=5, key=f"b_in_{label}")
    return s, b

uh_s, uh_b = tier_control_final("UHIGH")
h_s, h_b = tier_control_final("HIGH")
m_s, m_b = tier_control_final("MID")
l_s, l_b = tier_control_final("LOW")
ul_s, ul_b = tier_control_final("ULOW")

# 실행 버튼 클릭 시 모든 값을 세션에 강제 저장하는 함수
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

params = {
    'initial_capital': p_cap, 'max_cash_pct': p_max_cash, 'initial_entry_pct': p_init_entry,
    'uhigh_cut': uh_c/100, 'high_cut': h_c/100, 'low_cut': l_c/100, 'ulow_cut': ul_c/100,
    'sell_ratios': {'UHIGH': uh_s, 'HIGH': h_s, 'MID': m_s, 'LOW': l_s, 'ULOW': ul_s},
    'buy_ratios': {'UHIGH': uh_b, 'HIGH': h_b, 'MID': m_b, 'LOW': l_b, 'ULOW': ul_b}
}

# -----------------------------------------------------------
# 4. 실행 및 결과 출력
# -----------------------------------------------------------
if st.sidebar.button("🚀 시뮬레이션 실행", type="primary", on_click=update_session):
    with st.spinner("데이터 분석 중..."):
        df_weekly = get_backtest_data()
        res = run_simulation(df_weekly, p_start, p_end, params)
        
    if not res.empty:
        # 결과 계산
        final_asset = res.iloc[-1]['Asset']
        total_return = (final_asset / p_cap - 1) * 100
        days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
        cagr = ((final_asset / p_cap) ** (365 / days) - 1) * 100 if days > 0 else 0
        res['Peak'] = res['Asset'].cummax()
        res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
        mdd = res['DD'].min()
        
        # UI 출력
        st.subheader("🚩 성과 요약 리포트")
        m1, m2, m3 = st.columns(3)
        m1.metric("총수익률", f"{total_return:,.1f}%")
        m2.metric("CAGR", f"{cagr:.2f}%")
        m3.metric("MDD", f"{mdd:.1f}%")

        st.subheader("📈 자산 성장 및 하락폭(MDD)")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(res['Date'], res['Asset'], color='#1E88E5', lw=2, label="Asset")
        ax1.set_yscale('log')
        ax1.set_ylabel("Asset Value ($)")
        ax1.grid(True, which='both', linestyle='--', alpha=0.4)
        
        ax2 = ax1.twinx()
        ax2.fill_between(res['Date'], res['DD'], 0, color='#E57373', alpha=0.3)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(-100, 5)
        st.pyplot(fig)

        st.subheader("📅 연도별 성과 요약")
        res['Year'] = res['Date'].dt.year
        yearly_perf = []
        for year, group in res.groupby('Year'):
            y_ret = (group.iloc[-1]['Asset'] / group.iloc[0]['Asset'] - 1) * 100
            yearly_perf.append({'연도': year, '수익률': f"{y_ret:.1f}%", 'MDD': f"{group['DD'].min():.1f}%"})
        st.table(pd.DataFrame(yearly_perf).set_index('연도').T)
    else:
        st.warning("데이터가 부족합니다. 검증 기간을 확인해주세요.")
