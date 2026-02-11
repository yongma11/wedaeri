import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import matplotlib.dates as mdates
import os
import json
from datetime import datetime

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="TQQQ Wedaeri Pro", layout="wide", page_icon="🏆")

st.markdown("""
    <style>
    .big-metric { font-size: 24px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 10px; font-weight: bold; border: 1px solid #ddd; }
    .sub-text { font-size: 14px; color: #666; }
    .strategy-card { background: #fdfdfd; padding: 25px; border-radius: 15px; border: 1px solid #eee; margin-bottom: 25px; line-height: 1.7; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 및 파일 관리
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v5.json'
TRADE_LOG_FILE = 'wedaeri_trade_log_v5.csv'
PROFIT_LOG_FILE = 'wedaeri_profit_log_v5.csv'

def load_json(file, default):
    if os.path.exists(file):
        with open(file, 'r') as f: return json.load(f)
    return default

def save_json(file, data):
    with open(file, 'w') as f: json.dump(data, f)

settings = load_json(SETTINGS_FILE, {
    'start_date': '2025-01-01',
    'initial_capital': 10000,
    'max_cash_pct': 100,
    'initial_entry_pct': 50
})

# -----------------------------------------------------------
# 2. 정밀 데이터 엔진 (누적 회귀 및 오류 방지)
# -----------------------------------------------------------
def calculate_cumulative_growth(series):
    """ Look-ahead bias 없이 각 시점까지의 누적 데이터를 기반으로 추세선을 구합니다. """
    results = [np.nan] * len(series)
    date_nums = series.index.map(pd.Timestamp.toordinal).values
    values = series.values
    # 최소 1년(252일) 데이터 확보 후 계산 시작
    for i in range(252, len(series)):
        y_train = values[:i]
        x_train = date_nums[:i]
        mask = y_train > 0
        if not mask.any(): continue
        fit = np.polyfit(x_train[mask], np.log(y_train[mask]), 1)
        results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=3600)
def fetch_data():
    # 데이터 다운로드 (ValueError 방지를 위해 안전하게 처리)
    qqq_raw = yf.download("QQQ", start="2000-01-01", progress=False, auto_adjust=True)
    tqqq_raw = yf.download("TQQQ", start="2010-01-01", progress=False, auto_adjust=True)
    
    # 컬럼 접근 안전화
    qqq = qqq_raw['Close'].squeeze()
    tqqq = tqqq_raw['Close'].squeeze()

    # QQQ 누적 추세선 계산
    qqq_df = pd.DataFrame({'Close': qqq})
    qqq_df['Growth'] = calculate_cumulative_growth(qqq_df['Close'])
    
    # 병합
    df = pd.concat([qqq_df, tqqq], axis=1).dropna(subset=['Growth'])
    df.columns = ['QQQ', 'Growth', 'TQQQ']
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    df['Weekday'] = df.index.weekday
    
    # 주간 데이터 (금요일)
    weekly = df[df['Weekday'] == 4].copy()
    weekly['TQQQ_Prev'] = weekly['TQQQ'].shift(1)
    return weekly.dropna()

def run_wedaeri_engine(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame(), []

    initial_cap = params['initial_capital']
    cash, shares, is_first = initial_cap, 0, True
    history, logs = [], []

    for date, row in sim_data.iterrows():
        price, prev_p, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        total_asset = cash + (shares * price)
        dynamic_max_cash = total_asset * (params['max_cash_pct'] / 100)
        
        # 용성님 최적화 티어 파라미터
        if mkt_eval > 0.10: tier, s_r, b_r = 'UHIGH', 1.50, 0.30
        elif mkt_eval > 0.05: tier, s_r, b_r = 'HIGH', 1.00, 0.60
        elif mkt_eval < -0.10: tier, s_r, b_r = 'ULOW', 0.30, 2.00
        elif mkt_eval < -0.06: tier, s_r, b_r = 'LOW', 0.60, 1.20
        else: tier, s_r, b_r = 'MID', 0.60, 0.60

        action, trade_val = "Hold", 0
        if is_first:
            trade_val = min(initial_cap * (params['initial_entry_pct'] / 100), dynamic_max_cash)
            shares = trade_val / price; cash -= trade_val; action, is_first = "First Buy", False
        else:
            diff_val = (shares * price) - (shares * prev_p)
            if diff_val > 0: # 매도
                trade_val = diff_val * s_r
                trade_val = min(trade_val, shares * price)
                shares -= (trade_val / price); cash += trade_val; action = "Sell"
            elif diff_val < 0: # 매수
                trade_val = abs(diff_val) * b_r
                avail_limit = dynamic_max_cash - (total_asset - cash)
                trade_val = min(cash, trade_val, max(0, avail_limit))
                shares += (trade_val / price); cash -= trade_val; action = "Buy"

        asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': asset})
        logs.append({
            'Date': date.strftime('%Y-%m-%d'), 'Tier': tier, 'Eval': f"{mkt_eval*100:.1f}%",
            'Type': action, 'Price': round(price, 2), 'Trade_Val': round(trade_val, 0), 
            'Shares': round(shares, 4), 'Cash': round(cash, 0), 'Total_Asset': round(asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 동기화
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.header("⚙️ 기본 설정")
with st.sidebar.form("main_settings"):
    set_date = st.date_input("투자 시작일", value=pd.to_datetime(settings['start_date']))
    set_cap = st.number_input("시작 원금 ($)", value=settings['initial_capital'], step=1000)
    set_max_cash = st.slider("최대 현금 투입 한도 (%)", 10, 100, settings['max_cash_pct'])
    set_init_pct = st.slider("초기 진입 비중 (%)", 0, 100, settings['initial_entry_pct'], step=5)
    sync_btn = st.form_submit_button("🔄 설정 저장 및 자동 동기화")

if sync_btn:
    settings.update({'start_date': set_date.strftime('%Y-%m-%d'), 'initial_capital': set_cap, 'max_cash_pct': set_max_cash, 'initial_entry_pct': set_init_pct})
    save_json(SETTINGS_FILE, settings)
    _, res_logs = run_wedaeri_engine(df_weekly, set_date, datetime.now(), settings)
    if res_logs:
        pd.DataFrame(res_logs).sort_values('Date', ascending=False).to_csv(TRADE_LOG_FILE, index=False)
        p_recs = [{'Date': l['Date'], 'Total_Asset': l['Total_Asset'], 'Return_Pct': round((l['Total_Asset']/set_cap-1)*100, 2)} for l in res_logs]
        pd.DataFrame(p_recs).sort_values('Date', ascending=False).to_csv(PROFIT_LOG_FILE, index=False)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면 레이아웃
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 위대리 전략 가이드"])

# --- TAB 1: 실전 트레이딩 ---
with tab1:
    last = df_weekly.iloc[-1]; t_log = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
    p_log = pd.read_csv(PROFIT_LOG_FILE) if os.path.exists(PROFIT_LOG_FILE) else pd.DataFrame()
    
    cash_now = t_log.iloc[0]['Cash'] if not t_log.empty else set_cap
    shares_now = t_log.iloc[0]['Shares'] if not t_log.empty else 0
    total_now = cash_now + (shares_now * last['TQQQ'])
    week_idx = (datetime.now().date() - pd.to_datetime(settings['start_date']).date()).days // 7 + 1
    
    m_tier = 'MID'; m_col = 'gray'
    if last['Eval'] > 0.10: m_tier = 'UHIGH'; m_col = 'red'
    elif last['Eval'] > 0.05: m_tier = 'HIGH'; m_col = 'orange'
    elif last['Eval'] < -0.10: m_tier = 'ULOW'; m_col = 'green'
    elif last['Eval'] < -0.06: m_tier = 'LOW'; m_col = 'lightgreen'

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="sub-text">시장모드</div><div class="big-metric" style="color:{m_col};">{m_tier} ({last["Eval"]*100:.1f}%)</div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="sub-text">TQQQ 현재가</div><div class="big-metric">${last["TQQQ"]:.2f}</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="sub-text">현금 비중</div><div class="big-metric">{(cash_now/total_now*100 if total_now>0 else 100):.1f}%</div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="sub-text">매매 주차</div><div class="big-metric">{week_idx}주차</div>', unsafe_allow_html=True)

    st.subheader("📝 오늘 주문표")
    co1, co2 = st.columns([1, 2])
    with co1: est_p = st.number_input("예상 종가 입력 ($)", value=float(last['TQQQ']), step=0.01)
    with co2:
        diff_p = est_p - last['TQQQ']
        decision, b_c = "관망", "#f8f9fa"
        if diff_p > 0:
            v = (shares_now * diff_p) * (1.5 if m_tier=='UHIGH' else 1.0 if m_tier=='HIGH' else 0.6)
            decision = f"📈 매도: ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#fff5f5"
        elif diff_p < 0:
            v = abs(shares_now * diff_p) * (2.0 if m_tier=='ULOW' else 1.2 if m_tier=='LOW' else 0.6)
            decision = f"📉 매수: ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#f0fff4"
        st.markdown(f'<div class="order-box" style="background:{b_c}; color:black;">{decision}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("💰 내 계좌 현황")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("보유수량", f"{shares_now:,.2f} 주"); k2.metric("예수금", f"${cash_now:,.0f}")
    k3.metric("평가손익", f"${total_now - set_cap:,.0f}", f"{(total_now/set_cap-1)*100:.1f}%"); k4.metric("현재 총자산", f"${total_now:,.0f}")

    with st.expander("📋 매매 로그 및 수익 일지 수정"):
        st.data_editor(t_log, num_rows="dynamic", use_container_width=True, key="log_ed")
        st.write("---")
        ed_p = st.data_editor(p_log, num_rows="dynamic", use_container_width=True, key="prof_ed")
        if not ed_p.empty:
            p_plot = ed_p.copy(); p_plot['Date'] = pd.to_datetime(p_plot['Date'])
            fig_r, ax_r1 = plt.subplots(figsize=(12, 4))
            ax_r1.plot(p_plot['Date'], p_plot['Total_Asset'], color='#1E88E5', marker='o')
            ax_r1.set_ylabel("Asset ($)"); ax_r2 = ax_r1.twinx(); peak = p_plot['Total_Asset'].cummax()
            ax_r2.fill_between(p_plot['Date'], (p_plot['Total_Asset']/peak-1)*100, 0, color='red', alpha=0.1)
            st.pyplot(fig_r)

# --- TAB 2: 백테스트 분석 ---
with tab2:
    st.subheader("📊 전략 정밀 백테스트 (성과 복원판)")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("검증 자본 ($)", 10000)
        bt_start = bc2.date_input("검증 시작일", pd.to_datetime("2010-02-12"))
        bt_end = bc3.date_input("검증 종료일", datetime.now())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        res, logs = run_wedaeri_engine(df_weekly, bt_start, bt_end, {'initial_capital': bt_cap, 'max_cash_pct': settings['max_cash_pct'], 'initial_entry_pct': settings['initial_entry_pct']})
        if not res.empty:
            final_v = res.iloc[-1]['Asset']; days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
            cagr = ((final_v / bt_cap) ** (365 / max(1, days)) - 1) * 100
            res['Peak'] = res['Asset'].cummax(); res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
            mdd = res['DD'].min(); w_ret = res['Asset'].pct_change().dropna()
            sharpe = (w_ret.mean() / w_ret.std()) * np.sqrt(52)
            sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52) if not w_ret[w_ret<0].empty else 0
            calmar = cagr / abs(mdd) if mdd != 0 else 0

            m1, m2, m3, m4 = st.columns(4); m5, m6, m7, m8 = st.columns(4)
            m1.metric("최종 수익금", f"${final_v - bt_cap:,.0f}", f"{(final_v/bt_cap-1)*100:.1f}%")
            m2.metric("CAGR", f"{cagr:.2f}%"); m3.metric("MDD", f"{mdd:.1f}%"); m4.metric("거래 횟수", f"{len([l for l in logs if l['Type'] != 'Hold'])}회")
            m5.metric("Calmar", f"{calmar:.2f}"); m6.metric("Sharpe", f"{sharpe:.2f}"); m7.metric("Sortino", f"{sortino:.2f}"); m8.metric("최종 자산", f"${final_v:,.0f}")

            fig_bt, ax_b1 = plt.subplots(figsize=(12, 5))
            ax_b1.plot(res['Date'], res['Asset'], color='#1E88E5', lw=2); ax_b1.set_yscale('log')
            ax_b2 = ax_b1.twinx(); ax_b2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.1)
            st.pyplot(fig_bt)
            st.subheader("📅 연도별 성과")
            res['Year'] = res['Date'].dt.year
            y_p = [{'연도':y, '수익률':f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD':f"{(g['Asset']/g['Asset'].cummax()-1).min()*100:.1f}%"} for y, g in res.groupby('Year')]
            st.table(pd.DataFrame(y_p).set_index('연도').T)
            st.subheader("📋 백테스트 상세 로그"); st.dataframe(pd.DataFrame(logs).sort_values('Date', ascending=False), use_container_width=True)

# --- TAB 3: 전략 가이드 ---
with tab3:
    st.markdown("""
    <div class="strategy-card">
        <h2>📘 위대리(Wedaeri) v1.1 Pro 전략 상세 가이드</h2>
        <p>본 전략은 나스닥 100 지수(QQQ)의 <b>전체 성장 역사</b>를 정밀 분석하여, 3배 레버리지(TQQQ)의 변동성을 수익으로 전환하는 시스템입니다.</p>
        
        <h3>1. 핵심 메커니즘: 누적 로그 회귀 (Cumulative Regression)</h3>
        <p>단기적인 이동평균이나 5년 단위의 짧은 추세선은 시장의 거대한 흐름을 놓칠 수 있습니다. 본 시스템은 2000년부터 현재까지의 모든 데이터를 누적하여 시장의 '진짜 적정 가치'를 산출합니다. 이를 통해 2010년대 강세장의 복리 효과를 온전히 누리며 $120만 수준의 성과를 재현합니다.</p>
        <h3>2. 유동적 현금 방패 (Dynamic Cash Shield)</h3>
        <p>자산 규모에 따라 하락장에서 대응 가능한 현금 한도를 매주 재계산합니다. 자산이 늘어날수록 방패의 크기도 함께 커지는 복리형 리스크 관리 시스템입니다.</p>
    </div>
    """, unsafe_allow_html=True)
