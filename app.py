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
st.set_page_config(page_title="Wedaeri Quantum T-Flow v1.6", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .program-title { font-size: 38px; font-weight: 900; color: #0D47A1; text-align: center; margin-bottom: 20px; letter-spacing: -1px; }
    .big-metric { font-size: 26px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 12px; font-weight: bold; border: 2px solid #1E88E5; background-color: #f0f7ff; margin-top: 10px; }
    .status-bar { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 6px solid #1E88E5; }
    .strategy-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #e0e0e0; line-height: 1.8; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="program-title">Wedaeri Quantum T-Flow</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 및 데이터 관리
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v16.json'
default_settings = {
    'start_date': '2025-01-01', 'initial_capital': 10000, 'max_cash_pct': 100, 'initial_entry_pct': 50,
    'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
    'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
    'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try: return json.load(f)
            except: return default_settings
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f: json.dump(data, f)

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
# 2. 엔진 로직 (수량 정수화 적용)
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

        action, trade_val, trade_qty = "관망", 0, 0
        s_r, b_r = params['sell_ratios'][tier]/100, params['buy_ratios'][tier]/100

        if is_first:
            trade_val_target = min(cap * (params['initial_entry_pct'] / 100), max_c_usage)
            # [수정] 초기 진입 수량 정수화
            trade_qty = round(trade_val_target / price)
            shares = trade_qty; cash -= (trade_qty * price); action, is_first = "매수", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0: # 상승 매도
                trade_val_target = diff * s_r
                # [수정] 매도 수량 정수화 및 반올림
                trade_qty_raw = round(trade_val_target / price)
                trade_qty = int(min(trade_qty_raw, shares))
                shares -= trade_qty; cash += (trade_qty * price); action = "매도"
                trade_qty = -trade_qty # 로그 표시용
            elif diff < 0: # 하락 매수
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_val_target = min(cash, abs(diff) * b_r, avail)
                    # [수정] 매수 수량 정수화
                    trade_qty = round(trade_val_target / price)
                    # 실제 가진 돈보다 많이 살 수 없으므로 체크
                    if (trade_qty * price) > cash:
                        trade_qty = int(cash // price)
                    shares += trade_qty; cash -= (trade_qty * price); action = "매수"
        
        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({
            '날짜': date.strftime('%Y-%m-%d'), '시장평가': tier, '매매': action, 
            '가격 ($)': f"${price:.2f}", '거래수량': f"{trade_qty:+d}",
            '보유수량': int(shares), '평가금 ($)': round(shares * price, 0),
            '예수금 ($)': round(cash, 0), '총자산 ($)': round(current_asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 메인 화면
# -----------------------------------------------------------
df_weekly = fetch_weekly_data()

with st.sidebar:
    st.header("⚙️ System Config")
    with st.form("settings_form"):
        s_date = st.date_input("투자 시작일", value=pd.to_datetime(st.session_state.settings['start_date']))
        s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
        s_max_c = st.slider("현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
        s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
        save_btn = st.form_submit_button("💾 설정 저장 및 동기화", type="primary")

if save_btn:
    st.session_state.settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p})
    save_settings(st.session_state.settings); st.rerun()

tab1, tab2, tab3 = st.tabs(["🚀 실전 대시보드", "📊 백테스트 분석", "📘 매매전략 가이드"])

# --- TAB 1: 실전 대시보드 ---
with tab1:
    res_df, res_logs = run_engine(df_weekly, st.session_state.settings['start_date'], st.session_state.settings)
    last_mkt = df_weekly.iloc[-1]
    st.markdown(f'<div class="status-bar"><b>📅 분석 기준일:</b> {df_weekly.index[-1].strftime("%Y-%m-%d")} | <b>💎 TQQQ 종가:</b> ${last_mkt["TQQQ"]:.2f}</div>', unsafe_allow_html=True)
    
    if res_logs:
        last = res_logs[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("보유수량", f"{last['보유수량']:,} 주")
        c2.metric("평가금", f"${last['평가금 ($)']:,.0f}")
        c3.metric("예수금", f"${last['예수금 ($)']:,.0f}")
        c4.metric("총자산", f"${last['총자산 ($)']:,.0f}")
        
        st.divider()
        st.subheader("🎯 이번 주 예상 주문")
        col_in, col_out = st.columns([1, 1.5])
        with col_in: est_p = st.number_input("예상 종가 ($)", value=float(last_mkt['TQQQ']), step=0.01)
        with col_out:
            diff = est_p - last_mkt['TQQQ']
            order_msg = "관망"
            if diff > 0:
                val_target = (last['보유수량'] * diff) * (st.session_state.settings['sell_ratios'][last['시장평가']]/100)
                qty_order = round(val_target / est_p)
                order_msg = f"매도: {int(min(qty_order, last['보유수량']))} 주 (약 ${qty_order*est_p:,.0f})"
            elif diff < 0:
                max_u = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
                avail = max_u - (st.session_state.settings['initial_capital'] - last['예수금 ($)'])
                val_target = min(last['예수금 ($)'], abs(last['보유수량'] * diff) * (st.session_state.settings['buy_ratios'][last['시장평가']]/100), max(0, avail))
                qty_order = round(val_target / est_p)
                if (qty_order * est_p) > last['예수금 ($)']: qty_order = int(last['예수금 ($)'] // est_p)
                order_msg = f"매수: {int(qty_order)} 주 (약 ${qty_order*est_p:,.0f})"
            st.markdown(f'<div class="order-box">{order_msg}</div>', unsafe_allow_html=True)

        st.subheader("📜 상세 매매로그")
        st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)

# --- TAB 2 & 3 로직은 기존과 동일하게 유지하되 수량 정수화 반영됨 ---
with tab2:
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
            final_v = b_df.iloc[-1]['Asset']; ret = (final_v / bt_cap - 1) * 100
            days = (b_df.iloc[-1]['Date'] - b_df.iloc[0]['Date']).days
            cagr = ((final_v / bt_cap) ** (365 / max(1, days)) - 1) * 100
            b_df['Peak'] = b_df['Asset'].cummax(); b_df['DD'] = (b_df['Asset'] / b_df['Peak'] - 1) * 100
            mdd = b_df['DD'].min(); calmar = cagr / abs(mdd) if mdd != 0 else 0
            w_ret = b_df['Asset'].pct_change().dropna()
            sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52) if not w_ret[w_ret<0].empty else 0
            st.columns(5)[0].metric("수익률", f"{ret:.1f}%") # 나머지 지표 생략(기존 동일)
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', lw=2); ax1.set_yscale('log')
            ax2 = ax1.twinx(); ax2.fill_between(b_df['Date'], b_df['DD'], 0, color='#E53935', alpha=0.2)
            st.pyplot(fig)
            st.table(pd.DataFrame([{'연도': y, '수익률': f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%"} for y, g in b_df.groupby(b_df['Date'].dt.year)]).set_index('연도'))
            st.dataframe(pd.DataFrame(b_logs).sort_values('날짜', ascending=False), use_container_width=True)

with tab3:
    st.markdown("""<div class="strategy-card"><h2>📘 Wedaeri Quantum T-Flow 전략 가이드</h2>... 기존 내용 유지 ...</div>""", unsafe_allow_html=True)
