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
    .status-bar { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #1E88E5; }
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
# 2. 데이터 및 시뮬레이션 엔진
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
    # 2000년부터 가져와 정밀 추세선 유지
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
            shares = trade_val / price; cash -= trade_val
            action, is_first = "First Buy", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0: # 상승 매도
                trade_val = diff * s_r
                qty = min(trade_val / price, shares)
                shares -= qty; cash += (qty * price); action = "Sell"; trade_val = qty * price
            elif diff < 0: # 하락 매수
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_val = min(cash, abs(diff) * b_r, avail)
                    qty = trade_val / price
                    shares += qty; cash -= trade_val; action = "Buy"

        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({
            '날짜': date.strftime('%Y-%m-%d'), '시장티어': tier, '매매타입': action, 
            '현재가': round(price, 2), '거래금액': round(trade_val, 0), 
            '보유수량': round(shares, 2), '잔액': round(cash, 0), '총자산': round(current_asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 (설정 및 저장)
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.header("⚙️ 실전 대시보드 동기화")
with st.sidebar.form("sync_settings"):
    # 현재 저장된 날짜를 기본값으로 표시
    cur_date = datetime.strptime(st.session_state.settings['start_date'], '%Y-%m-%d').date()
    s_date = st.date_input("투자 시작일 (기본: 2025-01-01)", value=cur_date)
    s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
    s_max_c = st.slider("최대 현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
    s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
    
    submit = st.form_submit_button("🔄 설정 저장 및 대시보드 동기화", type="primary")

if submit:
    st.session_state.settings.update({
        'start_date': s_date.strftime('%Y-%m-%d'),
        'initial_capital': s_cap,
        'max_cash_pct': s_max_c,
        'initial_entry_pct': s_init_p
    })
    save_settings(st.session_state.settings)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["🔥 실전 대시보드", "📊 백테스트 분석"])

with tab1:
    # 실전 설정에 기반한 실시간 엔진 실행
    res_df, res_logs = run_engine(df_weekly, st.session_state.settings['start_date'], st.session_state.settings)
    
    st.markdown(f"""
        <div class="status-bar">
            <b>📍 현재 기준:</b> {st.session_state.settings['start_date']} 부터 오늘까지의 매매 데이터를 동기화 중입니다.
        </div>
    """, unsafe_allow_html=True)

    if not res_logs:
        st.error(f"⚠️ {st.session_state.settings['start_date']} 이후의 금요일 데이터가 없습니다. 투자 시작일을 더 과거 날짜로 변경해 주세요.")
    else:
        last = res_logs[-1]
        mkt_last = df_weekly.iloc[-1]
        
        # 상단 요약 지표
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="sub-text">현재 보유량</div><div class="big-metric">{last["보유수량"]:,.2f} 주</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="sub-text">가용 현금</div><div class="big-metric">${last["잔액"]:,.0f}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="sub-text">총 자산 평가액</div><div class="big-metric">${last["총자산"]:,.0f}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="sub-text">누적 수익률</div><div class="big-metric">{(last["총자산"]/st.session_state.settings["initial_capital"]-1)*100:.1f}%</div>', unsafe_allow_html=True)

        st.divider()
        
        # 오늘 주문 생성기
        st.subheader("📝 오늘 주문표 생성")
        col_o1, col_o2 = st.columns([1, 2])
        with col_o1:
            est_p = st.number_input("오늘 예상 종가($)", value=float(mkt_last['TQQQ']), step=0.01)
        with col_o2:
            diff = est_p - mkt_last['TQQQ']
            decision = "관망 (Hold)"
            if diff > 0:
                v = (last['보유수량'] * diff) * (st.session_state.settings['sell_ratios'][last['시장티어']]/100)
                decision = f"📈 매도(SELL): ${v:,.0f} ({v/est_p:.2f} 주)"
            elif diff < 0:
                max_u = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
                avail = max_u - (st.session_state.settings['initial_capital'] - last['잔액'])
                v = min(last['잔액'], abs(last['보유수량'] * diff) * (st.session_state.settings['buy_ratios'][last['시장티어']]/100), max(0, avail))
                decision = f"📉 매수(BUY): ${v:,.0f} ({v/est_p:.2f} 주)"
            st.markdown(f'<div class="order-box">{decision}</div>', unsafe_allow_html=True)

        # 상세 매매로그 (핵심 요청 사항)
        st.subheader("📜 투자 시작일 이후 상세 매매로그")
        log_df = pd.DataFrame(res_logs).sort_values('날짜', ascending=False)
        st.dataframe(log_df, use_container_width=True, height=400)

with tab2:
    st.subheader("📊 백테스트 (실전과 분리된 시뮬레이션)")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("시뮬레이션 투자금 ($)", value=10000, step=1000)
        bt_start = bc2.date_input("시뮬레이션 시작일", value=date(2010, 2, 12))
        bt_end = bc3.date_input("시뮬레이션 종료일", value=date.today())
        run_bt = st.form_submit_button("🚀 백테스트 실행")
    
    if run_bt:
        bt_params = st.session_state.settings.copy()
        bt_params['initial_capital'] = bt_cap
        b_df, b_logs = run_engine(df_weekly[df_weekly.index <= pd.to_datetime(bt_end)], bt_start.strftime('%Y-%m-%d'), bt_params)
        if not b_df.empty:
            st.line_chart(b_df.set_index('Date')['Asset'])
            st.dataframe(pd.DataFrame(b_logs).sort_values('날짜', ascending=False), use_container_width=True)
