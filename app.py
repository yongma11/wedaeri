import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 0. 기본 설정
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri v1.1 Optimizer", layout="wide", page_icon="📈")
st.title("📈 위대리 v1.1 (5-Tier) 최적화 시뮬레이터")

# -----------------------------------------------------------
# 1. 데이터 처리 로직
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
    initial_cap = params['initial_capital']
    cash = initial_cap
    shares = 0
    history = []
    
    max_cash_usage = initial_cap * (params['max_cash_pct'] / 100)
    
    for date, row in sim_data.iterrows():
        price = row['TQQQ']
        prev_price = row['TQQQ_Prev']
        mkt_eval = row['Eval']
        
        # 시장 티어 분류
        if mkt_eval > params['uhigh_cut']: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']: tier = 'ULOW'
        elif mkt_eval < params['low_cut']: tier = 'LOW'
        else: tier = 'MID'
            
        action = "Hold"
        if shares > 0:
            current_value = shares * price
            prev_value = shares * prev_price
            diff = current_value - prev_value
            
            if diff > 0: # 상승 시 매도
                rate = params['sell_ratios'][tier]
                trade_val = diff * (rate / 100)
                qty_to_sell = int(trade_val / price)
                if qty_to_sell > 0:
                    shares -= qty_to_sell
                    cash += qty_to_sell * price
                    action = "Sell"
            elif diff < 0: # 하락 시 매수
                rate = params['buy_ratios'][tier]
                current_used_cash = initial_cap - cash
                available_limit = max_cash_usage - current_used_cash
                
                if available_limit > 0:
                    trade_val = abs(diff) * (rate / 100)
                    actual_buy_amt = min(cash, trade_val, available_limit)
                    qty_to_buy = int(actual_buy_amt / price)
                    if qty_to_buy > 0:
                        shares += qty_to_buy
                        cash -= qty_to_buy * price
                        action = "Buy"
        else:
            first_buy_amt = min(cash, max_cash_usage * 0.5)
            shares = int(first_buy_amt / price)
            cash -= shares * price
            action = "First Buy"
        
        total_asset = cash + (shares * price)
        history.append({
            'Date': date, 'Tier': tier, 'Price': round(price, 2), 
            'Action': action, 'Asset': total_asset, 'Cash': cash, 'DD': 0
        })
        
    return pd.DataFrame(history)

# -----------------------------------------------------------
# 3. 사이드바 설정 (단위 및 범위 수정)
# -----------------------------------------------------------
st.sidebar.header("⚙️ 전략 파라미터")
p_start = st.sidebar.date_input("시작일", pd.to_datetime("2010-01-01"))
p_end = st.sidebar.date_input("종료일", pd.to_datetime("2025-12-31"))
p_cap = st.sidebar.number_input("초기 자본 ($)", 10000, step=1000)

st.sidebar.divider()
st.sidebar.subheader("🔒 리스크 관리")
p_max_cash = st.sidebar.slider("최대 현금 투입 한도 (%)", 10, 100, 100)

st.sidebar.subheader("📉 시장 평가 기준 (%)")
# 0.5% 단위 조절 (UI상으로는 0.5, 내부적으로는 0.005 적용)
uh_c = st.sidebar.number_input("초고평가(UHIGH) >", value=10.0, step=0.5, format="%.1f") / 100
h_c = st.sidebar.number_input("고평가(HIGH) >", value=7.0, step=0.5, format="%.1f") / 100
l_c = st.sidebar.number_input("저평가(LOW) <", value=-5.0, step=0.5, format="%.1f") / 100
ul_c = st.sidebar.number_input("초저평가(ULOW) <", value=-10.0, step=0.5, format="%.1f") / 100

st.sidebar.subheader("💰 티어별 매매율 (%)")
def tier_control(label, def_s, def_b):
    st.sidebar.write(f"**[{label}]**")
    c1, c2 = st.sidebar.columns(2)
    # 0% ~ 400%, 10% 단위 조절
    s = c1.number_input(f"매도%", 0, 400, int(def_s), step=10, key=f"s_{label}")
    b = c2.number_input(f"매수%", 0, 400, int(def_b), step=10, key=f"b_{label}")
    return s, b

uh_s, uh_b = tier_control("UHIGH", 100, 30)
h_s, h_b = tier_control("HIGH", 70, 50)
m_s, m_b = tier_control("MID", 50, 50)
l_s, l_b = tier_control("LOW", 30, 70)
ul_s, ul_b = tier_control("ULOW", 20, 100)

params = {
    'initial_capital': p_cap, 'max_cash_pct': p_max_cash,
    'uhigh_cut': uh_c, 'high_cut': h_c, 'low_cut': l_c, 'ulow_cut': ul_c,
    'sell_ratios': {'UHIGH': uh_s, 'HIGH': h_s, 'MID': m_s, 'LOW': l_s, 'ULOW': ul_s},
    'buy_ratios': {'UHIGH': uh_b, 'HIGH': h_b, 'MID': m_b, 'LOW': l_b, 'ULOW': ul_b}
}

# -----------------------------------------------------------
# 4. 실행 및 리포트
# -----------------------------------------------------------
if st.sidebar.button("🚀 시뮬레이션 실행", type="primary"):
    with st.spinner("백테스팅 진행 중..."):
        df_weekly = get_backtest_data()
        res = run_simulation(df_weekly, p_start, p_end, params)
        
    if not res.empty:
        # 지표 계산
        final_asset = res.iloc[-1]['Asset']
        total_return = (final_asset / p_cap - 1) * 100
        days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
        cagr = ((final_asset / p_cap) ** (365 / days) - 1) * 100
        res['Peak'] = res['Asset'].cummax()
        res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
        mdd = res['DD'].min()
        
        weekly_returns = res['Asset'].pct_change().dropna()
        sharpe = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52) if weekly_returns.std() != 0 else 0

        # 리포트 출력
        st.subheader("🚩 성과 요약 리포트")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총수익률", f"{total_return:,.2f}%")
        m2.metric("CAGR", f"{cagr:.2f}%")
        m3.metric("MDD", f"{mdd:.2f}%")
        m4.metric("샤프 지수", f"{sharpe:.2f}")

        # 통합 차트
        st.subheader("📈 자산 성장 및 하락폭(MDD)")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(res['Date'], res['Asset'], color='#1E88E5', label='Asset', lw=2)
        ax1.set_yscale('log')
        ax1.set_ylabel("Asset Value ($)", color='#1E88E5')
        ax1.grid(True, which='both', linestyle='--', alpha=0.4)
        
        ax2 = ax1.twinx()
        ax2.fill_between(res['Date'], res['DD'], 0, color='#E57373', alpha=0.3)
        ax2.set_ylabel("Drawdown (%)", color='#C62828')
        ax2.set_ylim(-100, 5)
        st.pyplot(fig)
        
        # 연도별 성과 (표 형식 개선)
        res['Year'] = res['Date'].dt.year
        yearly_perf = []
        for year, group in res.groupby('Year'):
            y_ret = (group.iloc[-1]['Asset'] / group.iloc[0]['Asset'] - 1) * 100
            yearly_perf.append({'연도': year, '수익률': f"{y_ret:.1f}%", 'MDD': f"{group['DD'].min():.1f}%"})
        st.table(pd.DataFrame(yearly_perf).set_index('연도').T)

    else: st.error("데이터가 없습니다.")
