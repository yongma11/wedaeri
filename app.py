import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
import json
from datetime import datetime, timedelta

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
# 1. 파일 관리 및 설정 (자동 저장 로직)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v2.json'
LOG_FILE = 'wedaeri_trade_log_v2.csv'
PROFIT_FILE = 'wedaeri_profit_log_v2.csv'

def load_json(file, default):
    if os.path.exists(file):
        with open(file, 'r') as f: return json.load(f)
    return default

def save_json(file, data):
    with open(file, 'w') as f: json.dump(data, f)

# 초기 설정값 (요청하신 25/1/1 세팅)
default_conf = {
    'start_date': '2025-01-01',
    'initial_capital': 10000,
    'max_cash_pct': 100,
    'initial_entry_pct': 50
}
settings = load_json(SETTINGS_FILE, default_conf)

# -----------------------------------------------------------
# 2. 정밀 매매 엔진 (이전 시뮬레이터 로직 완벽 복원)
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
    # 2010년 결과를 위해 2000년부터 데이터 예열
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", progress=False, auto_adjust=True)['Close']
    df = df.dropna()
    df['Growth'] = calculate_growth_curve(df['QQQ'], df.index)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    df['Weekday'] = df.index.weekday
    # 금요일 주간 데이터 생성
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
        
        # 고정된 시장 평가 기준 (매매 엔진 내장)
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
                trade_val = min(trade_val, shares * price) # 보유분 한도
                shares -= (trade_val / price); cash += trade_val; action = "Sell"
            elif diff_val < 0: # 하락 매수
                trade_val = abs(diff_val) * b_r
                available_cash = max(0, max_cash_limit - (initial_cap - cash))
                trade_val = min(cash, trade_val, available_cash)
                shares += (trade_val / price); cash -= trade_val; action = "Buy"

        asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': asset})
        if action != "Hold":
            logs.append({'날짜': date.strftime('%Y-%m-%d'), '상태': tier, '매매': action, '가격': round(price, 2), '거래금액': round(trade_val, 0), '보유수량': round(shares, 2)})
            
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 자동 동기화
# -----------------------------------------------------------
st.sidebar.header("⚙️ 기본 설정")
with st.sidebar.form("main_settings"):
    st.info("💡 동기화 버튼을 누르면 실전/백테스트가 즉시 갱신됩니다.")
    set_date = st.date_input("투자 시작일", value=pd.to_datetime(settings['start_date']))
    set_cap = st.number_input("시작 원금 ($)", value=settings['initial_capital'], step=1000)
    set_max_cash = st.slider("최대 현금 투입 한도 (%)", 10, 100, settings['max_cash_pct'])
    set_init_pct = st.slider("초기 진입 비중 (%)", 0, 100, settings['initial_entry_pct'], step=5)
    
    sync_btn = st.form_submit_button("🔄 설정 저장 및 자동 동기화")

if sync_btn:
    settings.update({'start_date': set_date.strftime('%Y-%m-%d'), 'initial_capital': set_cap, 'max_cash_pct': set_max_cash, 'initial_entry_pct': set_init_pct})
    save_json(SETTINGS_FILE, settings)
    st.rerun()

# 데이터 로드
df_weekly = fetch_data()

# -----------------------------------------------------------
# 4. 메인 대시보드 레이아웃
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 위대리 가이드"])

# ===========================================================
# TAB 1: 실전 트레이딩 (용성님 스타일 대시보드)
# ===========================================================
with tab1:
    last = df_weekly.iloc[-1]; prev = df_weekly.iloc[-2]
    eval_p = last['Eval']
    
    # 실시간 데이터 및 로그 로드
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['Date', 'Type', 'Tier', 'Price', 'Value', 'Qty', 'Balance_Qty', 'Total_Cash']).to_csv(LOG_FILE, index=False)
    t_log = pd.read_csv(LOG_FILE)
    
    cash_now = t_log.iloc[0]['Total_Cash'] if not t_log.empty else set_cap
    shares_now = t_log.iloc[0]['Balance_Qty'] if not t_log.empty else 0
    week_idx = (datetime.now().date() - set_date).days // 7 + 1
    
    # 상단 메트릭
    m_tier = 'MID'; m_col = 'gray'
    if eval_p > 0.10: m_tier = 'UHIGH'; m_col = 'red'
    elif eval_p > 0.05: m_tier = 'HIGH'; m_col = 'orange'
    elif eval_p < -0.10: m_tier = 'ULOW'; m_col = 'green'
    elif eval_p < -0.06: m_tier = 'LOW'; m_col = 'lightgreen'

    c_top1, c_top2, c_top3, c_top4 = st.columns(4)
    with c_top1: st.markdown(f'<div class="sub-text">시장모드</div><div class="big-metric" style="color:{m_col};">{m_tier} ({eval_p*100:.1f}%)</div>', unsafe_allow_html=True)
    with c_top2: st.markdown(f'<div class="sub-text">TQQQ 현재가</div><div class="big-metric">${last["TQQQ"]:.2f} <span style="font-size:14px; color:{"red" if last["TQQQ"]>prev["TQQQ"] else "blue"};">{(last["TQQQ"]/prev["TQQQ"]-1)*100:+.2f}%</span></div>', unsafe_allow_html=True)
    with c_top3: st.markdown(f'<div class="sub-text">현금 비중</div><div class="big-metric">{ (cash_now/(cash_now+shares_now*last["TQQQ"])*100) if (cash_now+shares_now)>0 else 100:.1f}%</div>', unsafe_allow_html=True)
    with c_top4: st.markdown(f'<div class="sub-text">매매 주차</div><div class="big-metric">{week_idx}주차</div>', unsafe_allow_html=True)

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
    eval_ret = (total_a / set_cap - 1) * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TQQQ 보유수량", f"{shares_now:,.2f} 주")
    k2.metric("예수금", f"${cash_now:,.0f}")
    k3.metric("총 평가손익", f"${total_a - set_cap:,.0f}", f"{eval_ret:.1f}%")
    k4.metric("현재 총자산", f"${total_a:,.0f}")

    # 로그 & 수익일지 (접기/수정)
    with st.expander("📋 매매 로그 및 수익 일지 수정"):
        ed_log = st.data_editor(t_log, num_rows="dynamic", use_container_width=True)
        if st.button("💾 로그 저장"): ed_log.to_csv(LOG_FILE, index=False); st.rerun()
        
        st.write("---")
        if not os.path.exists(PROFIT_FILE): pd.DataFrame(columns=['Date', 'Total_Asset', 'Profit', 'Return_Pct']).to_csv(PROFIT_FILE, index=False)
        p_log = pd.read_csv(PROFIT_FILE)
        ed_prof = st.data_editor(p_log, num_rows="dynamic", use_container_width=True)
        if st.button("💾 수익일지 저장"): ed_prof.to_csv(PROFIT_FILE, index=False); st.rerun()

    # 실전 자산 그래프
    if not p_log.empty:
        st.subheader("📈 자산 성장 히스토리")
        p_log['Date'] = pd.to_datetime(p_log['Date'])
        fig_real, ax_r1 = plt.subplots(figsize=(12, 4))
        ax_r1.plot(p_log['Date'], p_log['Total_Asset'], color='#1E88E5', label="Asset")
        ax_r1.set_yscale('log')
        ax_r2 = ax_r1.twinx()
        ax_r2.fill_between(p_log['Date'], (p_log['Total_Asset']/p_log['Total_Asset'].cummax()-1)*100, 0, color='red', alpha=0.1)
        st.pyplot(fig_real)

# ===========================================================
# TAB 2: 백테스트 분석 (이전 시뮬레이터 정밀 복원)
# ===========================================================
with tab2:
    st.subheader("📊 [위대리] 전략 정밀 백테스트")
    with st.form("bt_form"):
        c1, c2, c3 = st.columns(3)
        bt_cap = c1.number_input("검증 자본 ($)", 10000)
        bt_start = c2.date_input("검증 시작일", pd.to_datetime("2010-01-01"))
        bt_end = c3.date_input("검증 종료일", datetime.now())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        bt_params = settings.copy()
        bt_params.update({'initial_capital': bt_cap})
        res, logs = run_wedaeri_engine(df_weekly, bt_start, bt_end, bt_params)
        
        if not res.empty:
            final_v = res.iloc[-1]['Asset']; ret = (final_v/bt_cap-1)*100
            cagr = ((final_v/bt_cap)**(365/((pd.to_datetime(bt_end)-pd.to_datetime(bt_start)).days))-1)*100
            res['DD'] = (res['Asset']/res['Asset'].cummax()-1)*100; mdd = res['DD'].min()
            
            # 지수 계산
            calmar = cagr/abs(mdd); w_ret = res['Asset'].pct_change().dropna()
            sharpe = (w_ret.mean()/w_ret.std())*np.sqrt(52)
            sortino = (w_ret.mean()/w_ret[w_ret<0].std())*np.sqrt(52)

            m1, m2, m3, m4 = st.columns(4); m5, m6, m7, m8 = st.columns(4)
            m1.metric("최종 수익금", f"${final_v-bt_cap:,.0f}", f"{ret:.1f}%")
            m2.metric("CAGR", f"{cagr:.1f}%"); m3.metric("MDD", f"{mdd:.1f}%"); m4.metric("거래 횟수", f"{len(logs)}회")
            m5.metric("Calmar", f"{calmar:.2f}"); m6.metric("Sharpe", f"{sharpe:.2f}"); m7.metric("Sortino", f"{sortino:.2f}"); m8.metric("최종 자산", f"${final_v:,.0f}")

            # 백테스트 차트
            fig_bt, ax_b1 = plt.subplots(figsize=(12, 5))
            ax_b1.plot(res['Date'], res['Asset'], color='#1E88E5'); ax_b1.set_yscale('log')
            ax_b2 = ax_b1.twinx(); ax_b2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.1)
            st.pyplot(fig_bt)

            st.write("**📅 연도별 성과 요약**")
            res['Year'] = res['Date'].dt.year
            y_p = []
            for y, g in res.groupby('Year'):
                y_p.append({'연도':y, '수익률':f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD':f"{g['DD'].min():.1f}%"})
            st.table(pd.DataFrame(y_p).set_index('연도').T)

            st.write("**📋 상세 매매 로그**")
            st.dataframe(pd.DataFrame(logs).sort_values('날짜', ascending=False), use_container_width=True)

# ===========================================================
# TAB 3: 위대리 가이드 (용성님 최적 파라미터 내장)
# ===========================================================
with tab3:
    st.markdown("""
    ### 📘 [위대리 v1.1] 정밀 매매 가이드
    **위대리**는 QQQ의 5년 지수 회귀 추세선을 기반으로 TQQQ의 주간 변동성을 매매하는 시스템입니다.

    1. **시장 평가 기준 (Fixed)**: 
        * **UHIGH**: Eval > 10% | **HIGH**: Eval > 5% | **LOW**: Eval < -6% | **ULOW**: Eval < -10%
    2. **최적화된 매매율**:
        * **UHIGH**: 매도 150% / 매수 30%
        * **HIGH**: 매도 100% / 매수 60%
        * **MID**: 매도 60% / 매수 60%
        * **LOW**: 매도 60% / 매수 120%
        * **ULOW**: 매도 30% / 매수 200%
    3. **리스크 관리**: 초기 자본의 일정 비중만 진입하고, 하락장에서 설정한 현금 한도까지만 추가 투입하여 파산을 방지합니다.
    """)
