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
st.set_page_config(page_title="TQQQ Wedaeri Trader", layout="wide", page_icon="🏆")

st.markdown("""
    <style>
    .big-metric { font-size: 24px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 10px; font-weight: bold; border: 1px solid #ddd; }
    .sub-text { font-size: 14px; color: #666; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 파일 관리 및 설정 (용성님 최적 파라미터 고정)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v3.json'
LOG_FILE = 'wedaeri_trade_log_v3.csv'
PROFIT_FILE = 'wedaeri_profit_log_v3.csv'

def load_json(file, default):
    if os.path.exists(file):
        with open(file, 'r') as f: return json.load(f)
    return default

def save_json(file, data):
    with open(file, 'w') as f: json.dump(data, f)

# 기본 설정 (25/01/01 시작)
default_conf = {
    'start_date': '2025-01-01',
    'initial_capital': 10000,
    'max_cash_pct': 100,
    'initial_entry_pct': 50
}
settings = load_json(SETTINGS_FILE, default_conf)

# -----------------------------------------------------------
# 2. 정밀 매매 엔진 (MDD 30% 복원 로직)
# -----------------------------------------------------------
def calculate_growth_curve(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]
        x_train = date_nums[i-window : i]
        if np.any(y_train <= 0): continue
        try:
            fit = np.polyfit(x_train, np.log(y_train), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=3600)
def fetch_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", progress=False, auto_adjust=True)['Close']
    df = df.dropna()
    df['Growth'] = calculate_growth_curve(df['QQQ'], df.index)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    df['Weekday'] = df.index.weekday
    weekly = df[df['Weekday'] == 4].copy()
    weekly['TQQQ_Prev'] = weekly['TQQQ'].shift(1)
    return weekly.dropna()

def run_wedaeri_engine(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame(), []

    initial_cap = params['initial_capital']
    max_cash_limit = initial_cap * (params['max_cash_pct'] / 100)
    cash, shares, is_first = initial_cap, 0, True
    history, logs = [], []

    for date, row in sim_data.iterrows():
        price, prev_p, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        
        # 용성님 최적 파라미터 (엔진 내장)
        if mkt_eval > 0.10: tier, s_r, b_r = 'UHIGH', 1.50, 0.30
        elif mkt_eval > 0.05: tier, s_r, b_r = 'HIGH', 1.00, 0.60
        elif mkt_eval < -0.10: tier, s_r, b_r = 'ULOW', 0.30, 2.00
        elif mkt_eval < -0.06: tier, s_r, b_r = 'LOW', 0.60, 1.20
        else: tier, s_r, b_r = 'MID', 0.60, 0.60

        action, trade_val = "Hold", 0
        if is_first:
            trade_val = min(initial_cap * (params['initial_entry_pct'] / 100), max_cash_limit)
            shares = trade_val / price
            cash -= trade_val
            action, is_first = "First Buy", False
        else:
            diff_val = (shares * price) - (shares * prev_p)
            if diff_val > 0: # 상승 매도
                trade_val = diff_val * s_r
                trade_val = min(trade_val, shares * price)
                shares -= (trade_val / price); cash += trade_val; action = "Sell"
            elif diff_val < 0: # 하락 매수
                trade_val = abs(diff_val) * b_r
                available_cash = max(0, max_cash_limit - (initial_cap - cash))
                trade_val = min(cash, trade_val, available_cash)
                shares += (trade_val / price); cash -= trade_val; action = "Buy"

        history.append({'Date': date, 'Asset': cash + (shares * price)})
        if action != "Hold":
            logs.append({'날짜': date.strftime('%Y-%m-%d'), '상태': tier, '매매': action, '가격': round(price, 2), '거래금액': round(trade_val, 0), '보유수량': round(shares, 2)})
            
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 자동 동기화
# -----------------------------------------------------------
# 앱 시작 시 데이터를 먼저 불러와서 NameError 방지
df_weekly = fetch_data()

st.sidebar.header("⚙️ 기본 설정")
with st.sidebar.form("main_settings"):
    set_date = st.date_input("투자 시작일", value=pd.to_datetime(settings['start_date']))
    set_cap = st.number_input("시작 원금 ($)", value=settings['initial_capital'], step=1000)
    set_max_cash = st.slider("최대 현금 투입 한도 (%)", 10, 100, settings['max_cash_pct'])
    set_init_pct = st.slider("초기 진입 비중 (%)", 0, 100, settings['initial_entry_pct'], step=5)
    sync_btn = st.form_submit_button("🔄 설정 저장 및 실전 동기화")

if sync_btn:
    settings.update({'start_date': set_date.strftime('%Y-%m-%d'), 'initial_capital': set_cap, 'max_cash_pct': set_max_cash, 'initial_entry_pct': set_init_pct})
    save_json(SETTINGS_FILE, settings)
    
    # 실전 로그 자동 생성
    _, res_logs = run_wedaeri_engine(df_weekly, set_date, datetime.now(), settings)
    if res_logs:
        converted = []
        temp_cash = set_cap
        for l in res_logs:
            if l['매매'] in ['First Buy', 'Buy']: temp_cash -= l['거래금액']
            else: temp_cash += l['거래금액']
            converted.append({'Date': l['날짜'], 'Type': l['매매'], 'Tier': l['상태'], 'Price': l['가격'], 'Value': l['거래금액'], 'Balance_Qty': l['보유수량'], 'Total_Cash': temp_cash})
        pd.DataFrame(converted).sort_values('Date', ascending=False).to_csv(LOG_FILE, index=False)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면 레이아웃
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 위대리 가이드"])

with tab1:
    last = df_weekly.iloc[-1]; prev = df_weekly.iloc[-2]
    eval_p = last['Eval']
    
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['Date', 'Type', 'Tier', 'Price', 'Value', 'Balance_Qty', 'Total_Cash']).to_csv(LOG_FILE, index=False)
    t_log = pd.read_csv(LOG_FILE)
    
    cash_now = t_log.iloc[0]['Total_Cash'] if not t_log.empty else set_cap
    shares_now = t_log.iloc[0]['Balance_Qty'] if not t_log.empty else 0
    week_idx = (datetime.now().date() - pd.to_datetime(settings['start_date']).date()).days // 7 + 1
    
    # 상단 정보바
    m_tier = 'MID'; m_col = 'gray'
    if eval_p > 0.10: m_tier = 'UHIGH'; m_col = 'red'
    elif eval_p > 0.05: m_tier = 'HIGH'; m_col = 'orange'
    elif eval_p < -0.10: m_tier = 'ULOW'; m_col = 'green'
    elif eval_p < -0.06: m_tier = 'LOW'; m_col = 'lightgreen'

    c_t1, c_t2, c_t3, c_t4 = st.columns(4)
    with c_t1: st.markdown(f'<div class="sub-text">시장모드</div><div class="big-metric" style="color:{m_col};">{m_tier} ({eval_p*100:.1f}%)</div>', unsafe_allow_html=True)
    with c_t2: st.markdown(f'<div class="sub-text">TQQQ 현재가</div><div class="big-metric">${last["TQQQ"]:.2f}</div>', unsafe_allow_html=True)
    with c_top3: st.markdown(f'<div class="sub-text">현금 비중</div><div class="big-metric">{ (cash_now/(cash_now+shares_now*last["TQQQ"])*100) if (cash_now+shares_now)>0 else 100:.1f}%</div>', unsafe_allow_html=True)
    with c_t4: st.markdown(f'<div class="sub-text">매매 주차</div><div class="big-metric">{week_idx}주차</div>', unsafe_allow_html=True)

    # 오늘 주문표
    st.subheader("📝 오늘 주문표 (Daily Order)")
    co1, co2 = st.columns([1, 2])
    with co1: est_p = st.number_input("예상 종가 입력 ($)", value=float(last['TQQQ']), step=0.01)
    with co2:
        diff_p = est_p - last['TQQQ']
        decision, b_c = "관망 (Hold)", "#f8f9fa"
        if diff_p > 0:
            v = (shares_now * diff_p) * (1.5 if m_tier=='UHIGH' else 1.0 if m_tier=='HIGH' else 0.6)
            decision = f"📈 매도 (SELL): 약 ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#fff5f5"
        elif diff_p < 0:
            v = abs(shares_now * diff_p) * (2.0 if m_tier=='ULOW' else 1.2 if m_tier=='LOW' else 0.6)
            decision = f"📉 매수 (BUY): 약 ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#f0fff4"
        st.markdown(f'<div class="order-box" style="background:{b_c};">{decision}</div>', unsafe_allow_html=True)

    # 계좌 현황
    st.divider()
    st.subheader("💰 내 계좌 현황")
    total_a = cash_now + (shares_now * est_p)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TQQQ 보유수량", f"{shares_now:,.2f} 주")
    k2.metric("예수금", f"${cash_now:,.0f}")
    k3.metric("총 평가손익", f"${total_a - set_cap:,.0f}", f"{(total_a/set_cap-1)*100:.1f}%")
    k4.metric("현재 총자산", f"${total_a:,.0f}")

    with st.expander("📋 매매 로그 및 수익 일지 수정"):
        ed_log = st.data_editor(t_log, num_rows="dynamic", use_container_width=True)
        if st.button("💾 로그 저장"): ed_log.to_csv(LOG_FILE, index=False); st.rerun()

# 

with tab2:
    st.subheader("📊 [위대리] 전략 정밀 백테스트")
    with st.form("bt_form"):
        c1, c2, c3 = st.columns(3)
        bt_cap = c1.number_input("검증 자본 ($)", 10000)
        bt_start = c2.date_input("검증 시작일", pd.to_datetime("2010-01-01"))
        bt_end = c3.date_input("검증 종료일", datetime.now())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        res, logs = run_wedaeri_engine(df_weekly, bt_start, bt_end, {'initial_capital': bt_cap, 'max_cash_pct': set_max_cash, 'initial_entry_pct': set_init_pct})
        if not res.empty:
            # 성과 지표 출력 및 그래프 (이전 로직 유지)
            st.success("백테스트 완료!")
            final_v = res.iloc[-1]['Asset']
            st.metric("최종 자산", f"${final_v:,.0f}", f"{(final_v/bt_cap-1)*100:.1f}%")
            
            fig_bt, ax_b1 = plt.subplots(figsize=(12, 5))
            ax_b1.plot(res['Date'], res['Asset'], color='#1E88E5')
            ax_b1.set_yscale('log')
            st.pyplot(fig_bt)

with tab3:
    st.markdown("### 📘 [위대리 v1.1] 정밀 매매 가이드")
    st.write("1. **UHIGH**: 매도 150% / 매수 30% | 2. **HIGH**: 매도 100% / 매수 60% | 3. **MID**: 매도 60% / 매수 60% | 4. **LOW**: 매도 60% / 매수 120% | 5. **ULOW**: 매도 30% / 매수 200%")
