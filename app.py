import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
import json
from datetime import datetime, date

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri Quantum T-Flow v1.2", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .big-metric { font-size: 26px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 25px; border-radius: 12px; font-weight: bold; border: 2px solid #eee; margin-bottom: 20px; }
    .sub-text { font-size: 14px; color: #666; margin-bottom: 5px; }
    .status-bar { background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 6px solid #1E88E5; }
    .program-title { font-size: 32px; font-weight: 800; color: #0D47A1; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 관리
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_final.json'
default_settings = {
    'start_date': '2025-01-01', 
    'initial_capital': 10000,
    'max_cash_pct': 100,
    'initial_entry_pct': 50,
    'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
    'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
    'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                loaded = json.load(f)
                for k, v in default_settings.items():
                    if k not in loaded: loaded[k] = v
                return loaded
            except: return default_settings
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

# -----------------------------------------------------------
# 2. 엔진 로직
# -----------------------------------------------------------
def calculate_growth_curve(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]; x_train = date_nums[i-window : i]
        if np.any(y_train <= 0): continue
        try:
            fit = np.polyfit(x_train, np.log(y_train), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=3600)
def fetch_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
    df['Growth'] = calculate_growth_curve(df['QQQ'], df.index, 1260)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    weekly = df[df.index.weekday == 4].copy()
    weekly['TQQQ_Prev'] = weekly['TQQQ'].shift(1)
    return weekly

def run_engine(df, start_dt, params):
    start_ts = pd.to_datetime(start_dt)
    sim_data = df[df.index >= start_ts].copy()
    if sim_data.empty: return pd.DataFrame(), []

    cap = params['initial_capital']
    cash, shares, is_first = cap, 0, True
    history, logs = [], []
    max_c_usage = cap * (params['max_cash_pct'] / 100)

    for date, row in sim_data.iterrows():
        price, prev_p, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        if np.isnan(mkt_eval): mkt_eval = 0.0
        tier = 'MID'
        if mkt_eval > params['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < params['low_cut']/100: tier = 'LOW'

        action, trade_val = "Hold", 0
        s_r, b_r = params['sell_ratios'][tier]/100, params['buy_ratios'][tier]/100

        if is_first:
            trade_val = min(cap * (params['initial_entry_pct'] / 100), max_c_usage)
            shares = trade_val / price; cash -= trade_val; action, is_first = "First Buy", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0:
                trade_val = diff * s_r
                qty = min(trade_val / price, shares)
                shares -= qty; cash += (qty * price); action = "Sell"; trade_val = qty * price
            elif diff < 0:
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_val = min(cash, abs(diff) * b_r, avail)
                    qty = trade_val / price
                    shares += qty; cash -= trade_val; action = "Buy"

        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({'날짜': date.strftime('%Y-%m-%d'), '티어': tier, '타입': action, '가격': price, '거래금액': trade_val, '수량': shares, '현금': cash, '자산': current_asset})
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.markdown('<div class="program-title">Wedaeri T-Flow</div>', unsafe_allow_html=True)
with st.sidebar.form("sync_settings"):
    cur_date = datetime.strptime(st.session_state.settings['start_date'], '%Y-%m-%d').date()
    s_date = st.date_input("투자 시작일", value=cur_date)
    s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
    s_max_c = st.slider("현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
    s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
    submit = st.form_submit_button("🔄 대시보드 동기화", type="primary")

if submit:
    st.session_state.settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p})
    save_settings(st.session_state.settings); st.rerun()

# -----------------------------------------------------------
# 4. 메인 대시보드
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 실전 대시보드", "📉 백테스트 분석"])

with tab1:
    res_df, res_logs = run_engine(df_weekly, st.session_state.settings['start_date'], st.session_state.settings)
    
    # [요청 반영] 오늘 날짜 및 TQQQ 종가 표시
    today_str = datetime.now().strftime('%Y-%m-%d')
    last_mkt = df_weekly.iloc[-1]
    
    st.markdown(f"""
        <div class="status-bar">
            <b>📅 오늘 날짜:</b> {today_str} | <b>💎 TQQQ 종가(최근):</b> ${last_mkt['TQQQ']:.2f} | <b>🌱 시작일:</b> {st.session_state.settings['start_date']}
        </div>
    """, unsafe_allow_html=True)

    if not res_logs:
        st.error("데이터를 찾을 수 없습니다. 시작일을 조정해 주세요.")
    else:
        last = res_logs[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="sub-text">현재 수량</div><div class="big-metric">{last["수량"]:,.2f} 주</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="sub-text">가용 현금</div><div class="big-metric">${last["현금"]:,.0f}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="sub-text">총 자산</div><div class="big-metric">${last["자산"]:,.0f}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="sub-text">수익률</div><div class="big-metric">{(last["자산"]/st.session_state.settings["initial_capital"]-1)*100:.1f}%</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📋 실전 매매 상세 로그")
        st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)

with tab2:
    st.subheader("🔍 백테스트 정밀 분석")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("테스트 원금 ($)", value=10000)
        bt_start = bc2.date_input("시작일", value=date(2010, 2, 12))
        bt_end = bc3.date_input("종료일", value=date.today())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        bt_params = st.session_state.settings.copy(); bt_params['initial_capital'] = bt_cap
        b_df, b_logs = run_engine(df_weekly[df_weekly.index <= pd.to_datetime(bt_end)], bt_start.strftime('%Y-%m-%d'), bt_params)
        
        if not b_df.empty:
            # [요청 반영] 전문 지표 계산
            final_v = b_df.iloc[-1]['Asset']
            ret = (final_v / bt_cap - 1) * 100
            days = (b_df.iloc[-1]['Date'] - b_df.iloc[0]['Date']).days
            cagr = ((final_v / bt_cap) ** (365 / max(1, days)) - 1) * 100
            b_df['Peak'] = b_df['Asset'].cummax()
            b_df['DD'] = (b_df['Asset'] / b_df['Peak'] - 1) * 100
            mdd = b_df['DD'].min()
            
            calmar = cagr / abs(mdd) if mdd != 0 else 0
            w_ret = b_df['Asset'].pct_change().dropna()
            sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52) if not w_ret[w_ret<0].empty else 0
            
            # 손익비 (Win/Loss Ratio)
            trades = pd.DataFrame(b_logs)
            plus = trades[trades['타입'].isin(['Sell', 'Buy']) & (trades['자산'].diff() > 0)]
            minus = trades[trades['타입'].isin(['Sell', 'Buy']) & (trades['자산'].diff() < 0)]
            win_loss = plus['자산'].diff().mean() / abs(minus['자산'].diff().mean()) if len(minus)>0 else 0

            # 지표 출력
            i1, i2, i3, i4, i5, i6 = st.columns(6)
            i1.metric("최종수익률", f"{ret:.1f}%")
            i2.metric("CAGR", f"{cagr:.1f}%")
            i3.metric("MDD", f"{mdd:.1f}%")
            i4.metric("칼마 지수", f"{calmar:.2f}")
            i5.metric("소르티노", f"{sortino:.2f}")
            i6.metric("손익비", f"{win_loss:.2f}")

            # [요청 반영] 수익률 및 MDD 그래프
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', lw=2)
            ax1.set_yscale('log'); ax1.set_title("자산 성장 곡선 (Log Scale)"); ax1.grid(True, alpha=0.2)
            ax2.fill_between(b_df['Date'], b_df['DD'], 0, color='#E53935', alpha=0.3)
            ax2.set_title("Drawdown (MDD)"); ax2.set_ylim(-100, 5); ax2.grid(True, alpha=0.2)
            st.pyplot(fig)
