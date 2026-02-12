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
st.set_page_config(page_title="Wedaeri Quantum T-Flow v1.9.1", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #eee; text-align: center; }
    .metric-label { font-size: 14px; color: #666; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #1E88E5; }
    .order-result { padding: 25px; border-radius: 10px; font-size: 24px; font-weight: bold; text-align: center; }
    .buy-box { background-color: #f0fff4; color: #2e7d32; border: 1px solid #c8e6c9; }
    .sell-box { background-color: #fff5f5; color: #c62828; border: 1px solid #ffcdd2; }
    .hold-box { background-color: #f8f9fa; color: #616161; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 및 데이터 관리 (KeyError 및 Duplicate Form 에러 방지)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v19.json'
default_settings = {
    'start_date': '2025-01-01', 'initial_capital': 10000, 'max_cash_pct': 100, 'initial_entry_pct': 50,
    'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
    'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
    'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                # 누락된 키가 있으면 기본값으로 채움
                for k, v in default_settings.items():
                    if k not in loaded: loaded[k] = v
                return loaded
        except: return default_settings
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f: json.dump(data, f)

# 세션 상태 초기화
if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

@st.cache_data(ttl=600)
def fetch_weekly_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
    window = 1260
    qqq_vals, date_nums = df['QQQ'].values, df.index.map(pd.Timestamp.toordinal).values
    growth = [np.nan] * len(df)
    for i in range(window, len(df)):
        fit = np.polyfit(date_nums[i-window : i], np.log(qqq_vals[i-window : i]), 1)
        growth[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    df['Growth'], df['Eval'] = growth, (df['QQQ'] / growth) - 1
    weekly_df = df[df.index.weekday == 4].copy()
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df

# -----------------------------------------------------------
# 2. 엔진 로직 (정수 수량 유지)
# -----------------------------------------------------------
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
        tier = 'MID'
        if mkt_eval > params['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < params['low_cut']/100: tier = 'LOW'

        action, trade_qty = "관망", 0
        s_r, b_r = params['sell_ratios'][tier]/100, params['buy_ratios'][tier]/100

        if is_first:
            trade_qty = round(min(cap * (params['initial_entry_pct'] / 100), max_c_usage) / price)
            shares = trade_qty; cash -= (trade_qty * price); action, is_first = "매수", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0:
                trade_qty = int(min(round((diff * s_r) / price), shares))
                shares -= trade_qty; cash += (trade_qty * price); action = "매도"
                trade_qty = -trade_qty
            elif diff < 0:
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_qty = round(min(cash, abs(diff) * b_r, avail) / price)
                    if (trade_qty * price) > cash: trade_qty = int(cash // price)
                    shares += trade_qty; cash -= (trade_qty * price); action = "매수"
        
        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({
            '날짜': date.strftime('%Y-%m-%d'), '시장평가': tier, '매매': action, 
            '가격 ($)': price, '거래수량': trade_qty,
            '보유수량': int(shares), '평가금 ($)': shares * price,
            '예수금 ($)': cash, '총자산 ($)': current_asset
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 (에러 해결 핵심: Unique Key 부여)
# -----------------------------------------------------------
df_weekly = fetch_weekly_data()

with st.sidebar:
    st.header("⚙️ 시스템 설정")
    # [수정] 폼의 Key를 유니크하게 변경하여 중복 폼 에러 해결
    with st.form(key="sidebar_settings_form_unique"):
        s_date = st.date_input("투자 시작일", value=pd.to_datetime(st.session_state.settings['start_date']))
        s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
        s_max_c = st.slider("현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
        s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
        save_btn = st.form_submit_button("💾 설정 저장 및 동기화", type="primary")

if save_btn:
    st.session_state.settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p})
    save_settings(st.session_state.settings); st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면 레이아웃
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🚀 실전 트레이딩", "📊 백테스트 분석", "📘 전략 로직"])

with tab1:
    res_df, res_logs = run_engine(df_weekly, st.session_state.settings['start_date'], st.session_state.settings)
    last_mkt = df_weekly.iloc[-1]
    prev_mkt = df_weekly.iloc[-2]
    
    # 상단 지표 바
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-container"><div class="metric-label">현재 날짜</div><div class="metric-value">{datetime.now().strftime("%Y-%m-%d")}</div></div>', unsafe_allow_html=True)
    with c2:
        eval_val = last_mkt['Eval'] * 100
        tier_str = "MID"
        if eval_val > st.session_state.settings['uhigh_cut']: tier_str = "UHIGH (초고평가)"
        elif eval_val > st.session_state.settings['high_cut']: tier_str = "HIGH (고평가)"
        elif eval_val < st.session_state.settings['ulow_cut']: tier_str = "ULOW (초저평가)"
        elif eval_val < st.session_state.settings['low_cut']: tier_str = "LOW (저평가)"
        st.markdown(f'<div class="metric-container"><div class="metric-label">시장 모드</div><div class="metric-value" style="color:#f57c00;">{tier_str}</div><div style="font-size:12px; color:#999;">평가율 {eval_val:.2f}%</div></div>', unsafe_allow_html=True)
    with c3:
        diff = last_mkt['TQQQ'] - prev_mkt['TQQQ']
        st.markdown(f'<div class="metric-container"><div class="metric-label">TQQQ 현재가</div><div class="metric-value">${last_mkt["TQQQ"]:.2f}</div><div style="font-size:12px; color:{"red" if diff>0 else "blue"};">전일대비 {diff:+.2f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-container"><div class="metric-label">매매 회차</div><div class="metric-value">{len(res_logs)}회차</div><div style="font-size:12px; color:#999;">주간 주기</div></div>', unsafe_allow_html=True)

    st.write("---")
    
    # 오늘 주문표
    st.subheader("📝 오늘 주문표 (Daily Order)")
    o_col1, o_col2 = st.columns([1, 2])
    if res_logs:
        last_st = res_logs[-1]
        with o_col1:
            est_p = st.number_input("예상 종가 입력 ($)", value=float(last_mkt['TQQQ']), step=0.01)
        with o_col2:
            diff_p = est_p - last_mkt['TQQQ']
            order_msg, box_class = "관망 (HOLD)", "hold-box"
            if diff_p > 0:
                s_ratio = st.session_state.settings['sell_ratios'].get(last_st['시장평가'], 60) / 100
                qty = int(min(round((last_st['보유수량'] * diff_p * s_ratio) / est_p), last_st['보유수량']))
                if qty > 0: order_msg, box_class = f"📈 매도 (SELL): 기준가 ${est_p:.2f} ({qty}주)", "sell-box"
            elif diff_p < 0:
                b_ratio = st.session_state.settings['buy_ratios'].get(last_st['시장평가'], 60) / 100
                max_lim = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
                avail = max_lim - (st.session_state.settings['initial_capital'] - last_st['예수금 ($)'])
                qty = round(min(last_st['예수금 ($)'], abs(last_st['보유수량'] * diff_p * b_ratio), max(0, avail)) / est_p)
                if qty > 0: order_msg, box_class = f"📉 매수 (BUY): 기준가 ${est_p:.2f} ({qty}주)", "buy-box"
            st.markdown(f'<div class="order-result {box_class}">{order_msg}</div>', unsafe_allow_html=True)

    st.write("---")

    # 내 계좌 현황
    st.subheader("💰 내 계좌 현황")
    if res_logs:
        acc = res_logs[-1]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("총 보유 수량", f"{acc['보유수량']:,} 주")
        a2.metric("보유 현금", f"${acc['예수금 ($)']:,.0f}")
        profit = acc['총자산 ($)'] - st.session_state.settings['initial_capital']
        profit_p = (profit / st.session_state.settings['initial_capital']) * 100
        a3.metric("총 평가 손익", f"${profit:,.0f}", f"{profit_p:+.1f}%")
        a4.metric("현재 총 자산", f"${acc['총자산 ($)']:,.0f}")

    with st.expander("📜 상세 매매 히스토리 및 그래프 보기"):
        st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)
        res_df['Peak'] = res_df['Asset'].cummax()
        res_df['DD'] = (res_df['Asset'] / res_df['Peak'] - 1) * 100
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(res_df['Date'], res_df['Asset'], color='#1E88E5'); ax1.set_yscale('log')
        ax2 = ax1.twinx(); ax2.fill_between(res_df['Date'], res_df['DD'], 0, color='red', alpha=0.1)
        st.pyplot(fig)
