import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
import json
from datetime import datetime

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="TQQQ Wedaeri Pro v1.1", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .big-metric { font-size: 24px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 25px; border-radius: 12px; font-weight: bold; border: 2px solid #eee; margin-bottom: 20px; }
    .sub-text { font-size: 14px; color: #666; margin-bottom: 5px; }
    .strategy-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 및 데이터 관리
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_final.json'

# [용성님 요청 반영] 투자 시작일 기본값을 2025-01-01로 설정
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
            loaded = json.load(f)
            for k, v in default_settings.items():
                if k not in loaded: loaded[k] = v
            return loaded
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

settings = load_settings()

# -----------------------------------------------------------
# 2. 엔진 로직 (시뮬레이터와 100% 일치)
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

def run_engine(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
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
            shares = trade_val / price; cash -= trade_val
            action, is_first = "First Buy", False
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
        logs.append({'Date': date.strftime('%Y-%m-%d'), 'Tier': tier, 'Type': action, 'Price': round(price, 2), 'Trade_Val': round(trade_val, 0), 'Shares': round(shares, 2), 'Cash': round(cash, 0), 'Total_Asset': round(current_asset, 0)})
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 (실전 전용 글로벌 설정)
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.header("⚙️ 실전 대시보드 설정")
with st.sidebar.form("global_settings"):
    g_date = st.date_input("투자 시작일 (기본: 2025-01-01)", value=pd.to_datetime(settings['start_date']))
    g_cap = st.number_input("투자 원금 ($)", value=settings['initial_capital'], step=1000)
    g_max_c = st.slider("최대 현금 투입 한도 (%)", 10, 100, settings['max_cash_pct'])
    g_init_p = st.slider("초기 진입 비중 (%)", 0, 100, settings['initial_entry_pct'])
    save_btn = st.form_submit_button("🔄 설정 저장 및 대시보드 반영")

if save_btn:
    settings.update({'start_date': g_date.strftime('%Y-%m-%d'), 'initial_capital': g_cap, 'max_cash_pct': g_max_c, 'initial_entry_pct': g_init_p})
    save_settings(settings)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면 (탭 분리)
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["🔥 실전 대시보드", "📊 백테스트 분석"])

# --- TAB 1: 실전 대시보드 ---
with tab1:
    res_df, res_logs = run_engine(df_weekly, settings['start_date'], datetime.now(), settings)
    
    if not res_logs:
        st.warning(f"{settings['start_date']} 이후의 매매 데이터가 없습니다. 시작일을 확인해 주세요.")
    else:
        last_log = res_logs[-1]
        last_market = df_weekly.iloc[-1]
        
        st.subheader("💰 실전 계좌 현황")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="sub-text">보유 수량</div><div class="big-metric">{last_log["Shares"]:,.2f} 주</div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="sub-text">가용 현금</div><div class="big-metric">${last_log["Cash"]:,.0f}</div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="sub-text">총 자산 평가액</div><div class="big-metric">${last_log["Total_Asset"]:,.0f}</div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="sub-text">수익률 (원금대비)</div><div class="big-metric">{(last_log["Total_Asset"]/settings["initial_capital"]-1)*100:.1f}%</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📝 오늘 주문표 (Daily Order Generator)")
        st.info(f"현재 시장 모드: **{last_log['Tier']}** (QQQ 이격도: {last_market['Eval']*100:.1f}%)")
        
        c_p1, c_p2 = st.columns([1, 2])
        with c_p1:
            est_p = st.number_input("오늘 예상 종가 입력 ($)", value=float(last_market['TQQQ']), step=0.01)
        with c_p2:
            diff_p = est_p - last_market['TQQQ']
            order_text = "관망 (Hold)"
            b_color = "#f8f9fa"
            
            if diff_p > 0: # 상승 매도
                trade_v = (last_log['Shares'] * diff_p) * (settings['sell_ratios'][last_log['Tier']]/100)
                qty_to_order = trade_v / est_p
                order_text = f"📈 매도(SELL): ${trade_v:,.0f} ({qty_to_order:.2f} 주)"
                b_color = "#fff5f5"
            elif diff_p < 0: # 하락 매수
                max_usage = settings['initial_capital'] * (settings['max_cash_pct']/100)
                avail_limit = max_usage - (settings['initial_capital'] - last_log['Cash'])
                trade_v = min(last_log['Cash'], abs(last_log['Shares'] * diff_p) * (settings['buy_ratios'][last_log['Tier']]/100), max(0, avail_limit))
                qty_to_order = trade_v / est_p
                order_text = f"📉 매수(BUY): ${trade_v:,.0f} ({qty_to_order:.2f} 주)"
                b_color = "#f0fff4"
            
            st.markdown(f'<div class="order-box" style="background-color:{b_color}; color:#333;">{order_text}</div>', unsafe_allow_html=True)

        with st.expander("📋 실전 매매 히스토리 (상세)"):
            st.dataframe(pd.DataFrame(res_logs).sort_values('Date', ascending=False), use_container_width=True)

# --- TAB 2: 백테스트 분석 ---
with tab2:
    st.subheader("🔍 백테스트 기간 및 자본 설정")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("테스트 원금 ($)", value=settings['initial_capital'], step=1000)
        bt_start = bc2.date_input("백테스트 시작일", value=pd.to_datetime("2010-02-12"))
        bt_end = bc3.date_input("백테스트 종료일", value=datetime.now())
        run_bt = st.form_submit_button("🚀 백테스트 실행")
    
    if run_bt:
        bt_params = settings.copy()
        bt_params.update({'initial_capital': bt_cap})
        b_df, b_logs = run_engine(df_weekly, bt_start, bt_end, bt_params)
        
        if not b_df.empty:
            final_v = b_df.iloc[-1]['Asset']
            total_ret = (final_v / bt_cap - 1) * 100
            mdd = ((b_df['Asset'] / b_df['Asset'].cummax() - 1) * 100).min()
            
            st.success(f"백테스트 완료: {bt_start.strftime('%Y-%m-%d')} ~ {bt_end.strftime('%Y-%m-%d')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("최종 자산", f"${final_v:,.0f}")
            c2.metric("총 수익률", f"{total_ret:.1f}%")
            c3.metric("최대 낙폭(MDD)", f"{mdd:.1f}%")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', label='Growth Curve')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_title("백테스트 자산 성장 곡선 (Log Scale)")
            st.pyplot(fig)
            
            st.subheader("📅 연도별 성과 리포트")
            b_df['Year'] = b_df['Date'].dt.year
            y_perf = []
            for y, g in b_df.groupby('Year'):
                y_perf.append({'연도': y, '수익률': f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD': f"{((g['Asset']/g['Asset'].cummax()-1)*100).min():.1f}%"})
            st.table(pd.DataFrame(y_perf).set_index('연도').T)
