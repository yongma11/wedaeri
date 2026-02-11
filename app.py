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
    .order-box { text-align: center; padding: 20px; border-radius: 10px; font-weight: bold; border: 1px solid #ddd; }
    .sub-text { font-size: 14px; color: #666; }
    .strategy-card { background-color: #fdfdfd; padding: 25px; border-radius: 15px; border: 1px solid #eee; margin-bottom: 25px; line-height: 1.8; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 파일 및 설정 관리
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_final.json'
TRADE_LOG_FILE = 'wedaeri_trade_log_final.csv'

# [수정] 기본값 정의 (KeyError 방지용)
default_settings = {
    'start_date': '2010-01-01',
    'initial_capital': 10000,
    'max_cash_pct': 100,
    'initial_entry_pct': 50,
    'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
    'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
    'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
}

def load_json():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            loaded = json.load(f)
            # [핵심] 기존 파일에 누락된 키가 있으면 기본값에서 보충
            for k, v in default_settings.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
    return default_settings

def save_json(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

settings = load_json()

# -----------------------------------------------------------
# 2. 정밀 엔진 및 데이터 로직
# -----------------------------------------------------------
def calculate_growth_curve_precise(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]
        x_train = date_nums[i-window : i]
        if np.any(y_train <= 0) or np.isnan(y_train).any(): continue
        try:
            fit = np.polyfit(x_train, np.log(y_train), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=3600)
def fetch_data():
    start_fetch = "2000-01-01"
    end_fetch = datetime.now().strftime('%Y-%m-%d')
    qqq = yf.download("QQQ", start=start_fetch, end=end_fetch, progress=False, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start_fetch, end=end_fetch, progress=False, auto_adjust=True)
    
    df = pd.concat([qqq['Close'], tqqq['Close']], axis=1).dropna()
    df.columns = ['QQQ', 'TQQQ']
    df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df.index, window=1260)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    
    df['Weekday'] = df.index.weekday
    weekly = df[df['Weekday'] == 4].copy()
    weekly['TQQQ_Prev'] = weekly['TQQQ'].shift(1)
    return weekly

def run_wedaeri_engine(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame(), []

    initial_cap = params['initial_capital']
    cash, shares, is_first = initial_cap, 0, True
    history, logs = [], []
    max_cash_usage = initial_cap * (params['max_cash_pct'] / 100)

    for date, row in sim_data.iterrows():
        price, prev_p, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        if np.isnan(mkt_eval): mkt_eval = 0.0

        tier = 'MID'
        if mkt_eval > params['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < params['low_cut']/100: tier = 'LOW'

        action, trade_val = "Hold", 0
        s_r = params['sell_ratios'][tier] / 100
        b_r = params['buy_ratios'][tier] / 100

        if is_first:
            trade_val = min(initial_cap * (params['initial_entry_pct'] / 100), max_cash_usage)
            shares = trade_val / price
            cash -= trade_val
            action, is_first = "First Buy", False
        else:
            if np.isnan(prev_p): continue
            diff_val = (shares * price) - (shares * prev_p)
            
            if diff_val > 0:
                trade_val = diff_val * s_r
                qty_to_sell = trade_val / price
                actual_sell_qty = min(qty_to_sell, shares)
                shares -= actual_sell_qty; cash += (actual_sell_qty * price)
                action = "Sell"; trade_val = actual_sell_qty * price
            elif diff_val < 0:
                available = max_cash_usage - (initial_cap - cash)
                if available > 0:
                    trade_val = min(cash, abs(diff_val) * b_r, available)
                    qty_to_buy = trade_val / price
                    shares += qty_to_buy; cash -= trade_val
                    action = "Buy"

        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({
            'Date': date.strftime('%Y-%m-%d'), 'Tier': tier, 'Eval': f"{mkt_eval*100:.1f}%",
            'Type': action, 'Price': round(price, 2), 'Trade_Val': round(trade_val, 0), 
            'Shares': round(shares, 2), 'Cash': round(cash, 0), 'Total_Asset': round(current_asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 (파라미터 설정 및 Submit 버튼 수정)
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.header("⚙️ 전략 파라미터 설정")
with st.sidebar.form("main_settings"):
    s_date = st.date_input("시작일", value=pd.to_datetime(settings['start_date']))
    s_cap = st.number_input("초기 자본 ($)", value=settings['initial_capital'], step=1000)
    s_max_c = st.slider("최대 현금 투입 한도 (%)", 10, 100, settings['max_cash_pct'])
    s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, settings['initial_entry_pct'])
    
    st.divider()
    st.write("**[티어 임계값 설정]**")
    c_uh = st.number_input("UHIGH Cut (%)", value=float(settings['uhigh_cut']), step=0.5)
    c_h = st.number_input("HIGH Cut (%)", value=float(settings['high_cut']), step=0.5)
    c_l = st.number_input("LOW Cut (%)", value=float(settings['low_cut']), step=0.5)
    c_ul = st.number_input("ULOW Cut (%)", value=float(settings['ulow_cut']), step=0.5)
    
    # [수정] 폼 제출 버튼 위치 확인
    sync_btn = st.form_submit_button("🔄 설정 저장 및 데이터 동기화", type="primary")

if sync_btn:
    settings.update({
        'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 
        'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p,
        'uhigh_cut': c_uh, 'high_cut': c_h, 'low_cut': c_l, 'ulow_cut': c_ul
    })
    save_json(settings)
    _, res_logs = run_wedaeri_engine(df_weekly, s_date, datetime.now(), settings)
    if res_logs:
        pd.DataFrame(res_logs).sort_values('Date', ascending=False).to_csv(TRADE_LOG_FILE, index=False)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 대시보드
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 가이드"])

with tab1:
    last = df_weekly.iloc[-1]
    t_log = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
    
    # 실시간 현금/수량 계산
    cash_now = t_log.iloc[0]['Cash'] if not t_log.empty else settings['initial_capital']
    shares_now = t_log.iloc[0]['Shares'] if not t_log.empty else 0
    total_now = cash_now + (shares_now * last['TQQQ'])
    
    eval_val = last['Eval']
    m_tier = 'MID'
    if eval_val > settings['uhigh_cut']/100: m_tier = 'UHIGH'
    elif eval_val > settings['high_cut']/100: m_tier = 'HIGH'
    elif eval_val < settings['ulow_cut']/100: m_tier = 'ULOW'
    elif eval_val < settings['low_cut']/100: m_tier = 'LOW'

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="sub-text">시장모드</div><div class="big-metric">{m_tier} ({eval_val*100:.1f}%)</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="sub-text">TQQQ 현재가</div><div class="big-metric">${last["TQQQ"]:.2f}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="sub-text">현금 비중</div><div class="big-metric">{(cash_now/total_now*100 if total_now>0 else 100):.1f}%</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="sub-text">현재 총자산</div><div class="big-metric">${total_now:,.0f}</div>', unsafe_allow_html=True)

    st.subheader("📝 오늘 주문표 (Daily Order)")
    co1, co2 = st.columns([1, 2])
    with co1: est_p = st.number_input("예상 종가 입력 ($)", value=float(last['TQQQ']), step=0.01)
    with co2:
        diff_p = est_p - last['TQQQ']
        decision, b_c = "관망 (Hold)", "#f8f9fa"
        if diff_p > 0:
            v = (shares_now * diff_p) * (settings['sell_ratios'][m_tier] / 100)
            decision = f"📈 매도 (SELL): ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#fff5f5"
        elif diff_p < 0:
            max_c_usage = settings['initial_capital'] * (settings['max_cash_pct'] / 100)
            avail = max_c_usage - (settings['initial_capital'] - cash_now)
            v = min(cash_now, abs(shares_now * diff_p) * (settings['buy_ratios'][m_tier] / 100), max(0, avail))
            decision = f"📉 매수 (BUY): ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#f0fff4"
        st.markdown(f'<div class="order-box" style="background:{b_c}; color:black;">{decision}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📋 실전 매매 기록 (최근 로그)")
    st.dataframe(t_log, use_container_width=True)

with tab2:
    st.subheader("📊 전략 정밀 백테스트 (시뮬레이터 로직)")
    if st.button("🚀 백테스트 실행"):
        res, b_logs = run_wedaeri_engine(df_weekly, s_date, datetime.now(), settings)
        if not res.empty:
            final_v = res.iloc[-1]['Asset']
            days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
            cagr = ((final_v / settings['initial_capital']) ** (365 / max(1, days)) - 1) * 100
            res['Peak'] = res['Asset'].cummax()
            res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
            mdd = res['DD'].min()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("최종 자산", f"${final_v:,.0f}")
            m2.metric("CAGR", f"{cagr:.2f}%")
            m3.metric("MDD", f"{mdd:.1f}%")
            m4.metric("수익률", f"{(final_v/settings['initial_capital']-1)*100:.1f}%")

            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(res['Date'], res['Asset'], color='#1E88E5', lw=2)
            ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.1)
            st.pyplot(fig)
            
            st.subheader("📅 연도별 성과 요약")
            res['Year'] = res['Date'].dt.year
            y_p = [{'연도':y, '수익률':f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD':f"{(g['Asset']/g['Asset'].cummax()-1).min()*100:.1f}%"} for y, g in res.groupby('Year')]
            st.table(pd.DataFrame(y_p).set_index('연도').T)

with tab3:
    st.markdown("""
    <div class="strategy-card">
        <h3>📘 시스템 업데이트 노트</h3>
        <p><b>1. KeyError 방지:</b> 설정 파일 로드 시 <code>uhigh_cut</code> 등 누락된 키가 있으면 기본값으로 자동 보정합니다.</p>
        <p><b>2. Submit 버튼 수정:</b> 폼 제출 버튼이 누락되지 않도록 사이드바 구조를 최적화했습니다.</p>
        <p><b>3. 실전 연동:</b> 시뮬레이터와 동일한 엔진을 사용하여 백테스트 결과가 실제 매매 신호와 정확히 일치합니다.</p>
    </div>
    """, unsafe_allow_html=True)
