import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
import json
from datetime import datetime, date

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일 (기존 유지)
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri Quantum T-Flow v1.8", layout="wide", page_icon="📈")

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
# 1. 설정 및 데이터 관리 (기존 유지)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v18.json'
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
# 2. 엔진 로직 (정수 수량 및 명칭 유지)
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
            '가격 ($)': f"${price:.2f}", '거래수량': f"{trade_qty:+d}",
            '보유수량': int(shares), '평가금 ($)': round(shares * price, 0),
            '예수금 ($)': round(cash, 0), '총자산 ($)': round(current_asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 레이아웃
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
        c1.metric("보유수량", f"{last['보유수량']:,} 주"); c2.metric("평가금", f"${last['평가금 ($)']:,.0f}")
        c3.metric("예수금", f"${last['예수금 ($)']:,.0f}"); c4.metric("총자산", f"${last['총자산 ($)']:,.0f}")
        
        st.divider()
        # [요청 반영] 상세 매매로그 접기 (Expander)
        with st.expander("📜 상세 매매로그 보기", expanded=False):
            st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)

        # [요청 반영] 실전 수익률 및 MDD 통합 그래프
        st.subheader("📈 실전 자산 성장 및 하락 분석")
        res_df['Peak'] = res_df['Asset'].cummax()
        res_df['DD'] = (res_df['Asset'] / res_df['Peak'] - 1) * 100
        
        fig_real, ax1_r = plt.subplots(figsize=(12, 5))
        ax1_r.plot(res_df['Date'], res_df['Asset'], color='#1E88E5', lw=2, label='자산 (Log)')
        ax1_r.set_yscale('log'); ax1_r.set_ylabel("Asset Value ($)"); ax1_r.grid(True, alpha=0.2)
        ax2_r = ax1_r.twinx()
        ax2_r.fill_between(res_df['Date'], res_df['DD'], 0, color='#E53935', alpha=0.2, label='MDD (%)')
        ax2_r.set_ylabel("Drawdown (%)"); ax2_r.set_ylim(-100, 5)
        st.pyplot(fig_real)

# --- TAB 2: 백테스트 분석 (기존 유지) ---
with tab2:
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("테스트 원금 ($)", value=10000)
        bt_start = bc2.date_input("시작일", value=date(2010, 2, 12)); bt_end = bc3.date_input("종료일", value=date.today())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        bt_params = st.session_state.settings.copy(); bt_params['initial_capital'] = bt_cap
        b_df, b_logs = run_engine(df_weekly[df_weekly.index <= pd.to_datetime(bt_end)], bt_start.strftime('%Y-%m-%d'), bt_params)
        if not b_df.empty:
            final_v = b_df.iloc[-1]['Asset']; cagr = ((final_v / bt_cap) ** (365 / max(1, (b_df.iloc[-1]['Date'] - b_df.iloc[0]['Date']).days)) - 1) * 100
            b_df['Peak'] = b_df['Asset'].cummax(); b_df['DD'] = (b_df['Asset'] / b_df['Peak'] - 1) * 100
            mdd = b_df['DD'].min(); calmar = cagr / abs(mdd) if mdd != 0 else 0
            w_ret = b_df['Asset'].pct_change().dropna(); sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52) if not w_ret[w_ret<0].empty else 0
            
            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("최종수익률", f"{(final_v/bt_cap-1)*100:.1f}%"); i2.metric("CAGR", f"{cagr:.1f}%"); i3.metric("MDD", f"{mdd:.1f}%"); i4.metric("칼마", f"{calmar:.2f}"); i5.metric("소르티노", f"{sortino:.2f}")

            fig_bt, ax1_b = plt.subplots(figsize=(12, 5))
            ax1_b.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', lw=2); ax1_b.set_yscale('log'); ax2_b = ax1_b.twinx()
            ax2_b.fill_between(b_df['Date'], b_df['DD'], 0, color='#E53935', alpha=0.2); st.pyplot(fig_bt)
            
            b_df['Year'] = b_df['Date'].dt.year
            y_data = [{'연도': y, '수익률': f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD': f"{(g['Asset']/g['Asset'].cummax()-1).min()*100:.1f}%", '기말자산': f"${g.iloc[-1]['Asset']:,.0f}"} for y, g in b_df.groupby('Year')]
            st.table(pd.DataFrame(y_data).set_index('연도'))
            with st.expander("📜 상세 매매로그 보기", expanded=False):
                st.dataframe(pd.DataFrame(b_logs).sort_values('날짜', ascending=False), use_container_width=True)

# --- TAB 3: 매매전략 가이드 (기존 유지) ---
with tab3:
    st.markdown("""<div class="strategy-card"><h2>📘 Wedaeri Quantum T-Flow 매매전략 가이드</h2>...내용 생략(기존 유지)...</div>""", unsafe_allow_html=True)
