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
st.set_page_config(page_title="TQQQ Wedaeri Pro v1.1", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .big-metric { font-size: 24px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 25px; border-radius: 12px; font-weight: bold; border: 2px solid #eee; margin-bottom: 20px; }
    .sub-text { font-size: 14px; color: #666; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 관리 (투자시작일 2025-01-01 고정 로직 추가)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_final.json'

# 기본값 설정
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
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                # 누락된 키 보충
                for k, v in default_settings.items():
                    if k not in loaded: loaded[k] = v
                return loaded
        except:
            return default_settings
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

# 세션 상태를 사용하여 설정 유지
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

def run_engine(df, start_dt, end_dt, params):
    # 날짜 필터링 정밀화
    start_ts = pd.to_datetime(start_dt)
    end_ts = pd.to_datetime(end_dt)
    sim_data = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    
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

        history.append({'Date': date, 'Asset': cash + (shares * price)})
        logs.append({'Date': date.strftime('%Y-%m-%d'), 'Tier': tier, 'Type': action, 'Price': round(price, 2), 'Trade_Val': round(trade_val, 0), 'Shares': round(shares, 2), 'Cash': round(cash, 0), 'Total_Asset': round(cash + (shares * price), 0)})
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 (실전용 시작일 설정)
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.header("⚙️ 실전 대시보드 설정")
with st.sidebar.form("global_settings"):
    # session_state에서 날짜를 읽어와 기본값 설정
    current_start = datetime.strptime(st.session_state.settings['start_date'], '%Y-%m-%d').date()
    g_date = st.date_input("투자 시작일", value=current_start)
    g_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
    g_max_c = st.slider("최대 현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
    g_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
    save_btn = st.form_submit_button("🔄 설정 저장 및 대시보드 반영", type="primary")

if save_btn:
    st.session_state.settings.update({
        'start_date': g_date.strftime('%Y-%m-%d'), 
        'initial_capital': g_cap, 
        'max_cash_pct': g_max_c, 
        'initial_entry_pct': g_init_p
    })
    save_settings(st.session_state.settings)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["🔥 실전 대시보드", "📊 백테스트 분석"])

with tab1:
    # 실전 설정으로 엔진 가동
    res_df, res_logs = run_engine(df_weekly, st.session_state.settings['start_date'], datetime.now(), st.session_state.settings)
    
    if not res_logs:
        st.error(f"데이터 없음: {st.session_state.settings['start_date']} 이후의 금요일 데이터가 아직 없습니다. 시작일을 더 과거로 조정해 주세요.")
    else:
        last_log = res_logs[-1]
        last_mkt = df_weekly.iloc[-1]
        
        st.subheader(f"💰 실전 계좌 현황 (시작일: {st.session_state.settings['start_date']})")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="sub-text">보유 수량</div><div class="big-metric">{last_log["Shares"]:,.2f} 주</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="sub-text">가용 현금</div><div class="big-metric">${last_log["Cash"]:,.0f}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="sub-text">현재 평가액</div><div class="big-metric">${last_log["Total_Asset"]:,.0f}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="sub-text">수익률</div><div class="big-metric">{(last_log["Total_Asset"]/st.session_state.settings["initial_capital"]-1)*100:.1f}%</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📝 오늘 주문 생성기")
        st.write(f"현재 시장 모드: **{last_log['Tier']}** | TQQQ 현재가: **${last_mkt['TQQQ']:.2f}**")
        
        est_p = st.number_input("오늘 예상 종가($)", value=float(last_mkt['TQQQ']), step=0.01)
        diff_p = est_p - last_mkt['TQQQ']
        
        # 주문 계산 로직
        order_msg = "관망 (Hold)"
        if diff_p > 0:
            val = (last_log['Shares'] * diff_p) * (st.session_state.settings['sell_ratios'][last_log['Tier']]/100)
            order_msg = f"📈 매도(SELL): ${val:,.0f} ({val/est_p:.2f} 주)"
        elif diff_p < 0:
            max_u = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
            avail = max_u - (st.session_state.settings['initial_capital'] - last_log['Cash'])
            val = min(last_log['Cash'], abs(last_log['Shares'] * diff_p) * (st.session_state.settings['buy_ratios'][last_log['Tier']]/100), max(0, avail))
            order_msg = f"📉 매수(BUY): ${val:,.0f} ({val/est_p:.2f} 주)"
        
        st.markdown(f'<div class="order-box">{order_msg}</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📊 백테스트 (실전과 별개)")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("테스트 원금 ($)", value=10000, step=1000)
        bt_start = bc2.date_input("백테스트 시작", value=date(2010, 2, 12))
        bt_end = bc3.date_input("백테스트 종료", value=date.today())
        run_bt = st.form_submit_button("🚀 분석 실행")
    
    if run_bt:
        bt_params = st.session_state.settings.copy()
        bt_params.update({'initial_capital': bt_cap})
        b_df, _ = run_engine(df_weekly, bt_start.strftime('%Y-%m-%d'), bt_end.strftime('%Y-%m-%d'), bt_params)
        if not b_df.empty:
            st.line_chart(b_df.set_index('Date')['Asset'])
            st.metric("최종 자산", f"${b_df.iloc[-1]['Asset']:,.0f}")
