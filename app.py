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
    # 용성님의 최적 시장 평가 기준
    st.session_state.uh_c = 10.0
    st.session_state.h_c = 5.0
    st.session_state.l_c = -6.0
    st.session_state.ul_c = -10.0
    # 용성님의 최적 매도/매수율
    st.session_state.vals = {
        's_UHIGH': 150, 'b_UHIGH': 30,
        's_HIGH': 100, 'b_HIGH': 60,
        's_MID': 60, 'b_MID': 60,
        's_LOW': 60, 'b_LOW': 120,
        's_ULOW': 30, 'b_ULOW': 200
    }

# -----------------------------------------------------------
# 1. 데이터 및 정밀 추세선 (로그 선형 회귀)
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

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진
# -----------------------------------------------------------
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
            trade_logs.append({
                '날짜': date.strftime('%Y-%m-%d'), '상태': tier, '매매': action, 
                '가격': round(price, 2), '거래금액': round(trade_val, 2), '보유수량': round(shares, 2)
            })
            
    return pd.DataFrame(history), trade_logs

# -----------------------------------------------------------
# 3. 사이드바 UI
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

# -----------------------------------------------------------
# 4. 실행 및 결과 출력
# -----------------------------------------------------------
if st.sidebar.button("🚀 시뮬레이션 실행", type="primary", on_click=update_session):
    with st.spinner("백테스팅 중..."):
        df_weekly = get_backtest_data()
        res, logs = run_simulation(df_weekly, p_start, p_end, params)
    
    if not res.empty:
        final_asset = res.iloc[-1]['Asset']
        total_ret = (final_asset / p_cap - 1) * 100
        days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
        cagr = ((final_asset / p_cap) ** (365 / max(1, days)) - 1) * 100
        res['Peak'] = res['Asset'].cummax()
        res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
        mdd = res['DD'].min()

        # 지수 계산 (Sharpe, Calmar, Sortino)
        w_ret = res['Asset'].pct_change().dropna()
        sharpe = (w_ret.mean() / w_ret.std()) * np.sqrt(52) if w_ret.std() != 0 else 0
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        sortino = (w_ret.mean() / w_ret[w_ret < 0].std()) * np.sqrt(52) if not w_ret[w_ret < 0].empty else 0

        st.subheader("🚩 성과 요약 리포트")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총수익률", f"{total_ret:,.1f}%")
        col2.metric("CAGR", f"{cagr:.2f}%")
        col3.metric("MDD", f"{mdd:.1f}%")
        col4.metric("거래 횟수", f"{len(logs)}회")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Calmar 지수", f"{calmar:.2f}")
        col6.metric("샤프 지수", f"{sharpe:.2f}")
        col7.metric("소르티노 지수", f"{sortino:.2f}")
        col8.metric("최종 자산", f"${final_asset:,.0f}")

        # 그래프
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(res['Date'], res['Asset'], color='#1E88E5', lw=2)
        ax1.set_yscale('log')
        ax1.grid(True, which='both', linestyle='--', alpha=0.3)
        ax2 = ax1.twinx()
        ax2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.2)
        ax2.set_ylim(-100, 5)
        st.pyplot(fig)

        # 연도별 성과표
        st.subheader("📅 연도별 성과 요약")
        res['Year'] = res['Date'].dt.year
        y_perf = []
        for year, group in res.groupby('Year'):
            ret = (group.iloc[-1]['Asset'] / group.iloc[0]['Asset'] - 1) * 100
            y_perf.append({'연도': year, '수익률': f"{ret:.1f}%", 'MDD': f"{group['DD'].min():.1f}%"})
        st.table(pd.DataFrame(y_perf).set_index('연도').T)
        
        # 상세 거래 로그
        st.subheader("📋 상세 거래 내역")
        st.dataframe(pd.DataFrame(logs).sort_values('날짜', ascending=False), use_container_width=True)
    else:
        st.error("결과를 생성할 수 없습니다. 기간 설정을 확인하세요.")
