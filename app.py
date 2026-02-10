import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import matplotlib.dates as mdates
import os
import json
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="TQQQ Wedaeri Trader", layout="wide", page_icon="🏆")

st.markdown("""
    <style>
    .big-metric { font-size: 26px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 10px; color: white; font-weight: bold; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    .sub-text { font-size: 14px; color: #666; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 파일 입출력 및 설정 관리 (영구 저장)
# -----------------------------------------------------------
LOG_FILE = 'wedaeri_trade_log.csv'
PROFIT_FILE = 'wedaeri_profit_log.csv'
SETTINGS_FILE = 'wedaeri_settings.json'

def load_settings():
    default_settings = {
        'start_date': '2010-01-01',
        'initial_capital': 10000,
        'max_cash_pct': 100,
        'initial_entry_pct': 50,
        'uh_c': 10.0, 'h_c': 5.0, 'l_c': -6.0, 'ul_c': -10.0,
        'vals': {
            's_UHIGH': 150, 'b_UHIGH': 30, 's_HIGH': 100, 'b_HIGH': 60,
            's_MID': 60, 'b_MID': 60, 's_LOW': 60, 'b_LOW': 120, 's_ULOW': 30, 'b_ULOW': 200
        }
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: return json.load(f)
        except: return default_settings
    return default_settings

def save_settings(new_settings):
    with open(SETTINGS_FILE, 'w') as f: json.dump(new_settings, f)

def load_trade_log():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                return df.sort_values('Date', ascending=False)
        except: pass
    return pd.DataFrame(columns=['Date', 'Type', 'Tier', 'Price', 'Value', 'Qty', 'Balance_Qty', 'Total_Asset'])

def load_profit_log():
    if os.path.exists(PROFIT_FILE):
        try: return pd.read_csv(PROFIT_FILE)
        except: pass
    return pd.DataFrame(columns=['Date', 'Total_Asset', 'Profit', 'Return_Pct'])

# -----------------------------------------------------------
# 2. 데이터 및 로직 함수
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
def get_market_data():
    start_fetch = "2000-01-01"
    df = yf.download(["QQQ", "TQQQ"], start=start_fetch, progress=False, auto_adjust=True)['Close']
    df = df.dropna()
    df['Growth'] = calculate_growth_curve_precise(df['QQQ'], df.index)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    df['TQQQ_Prev'] = df['TQQQ'].shift(1)
    df['Weekday'] = df.index.weekday
    return df

def run_simulation(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    sim_data = sim_data[sim_data['Weekday'] == 4] # 금요일 주간 거래
    if sim_data.empty: return pd.DataFrame(), []
    
    initial_cap = params['initial_capital']
    cash, shares, is_first = initial_cap, 0, True
    history, trade_logs = [], []
    max_cash_usage = initial_cap * (params['max_cash_pct'] / 100)
    
    for date, row in sim_data.iterrows():
        price, prev_price, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        if np.isnan(mkt_eval): mkt_eval = 0.0
        
        tier = 'MID'
        if mkt_eval > params['uh_c']/100: tier = 'UHIGH'
        elif mkt_eval > params['h_c']/100: tier = 'HIGH'
        elif mkt_eval < params['ul_c']/100: tier = 'ULOW'
        elif mkt_eval < params['l_c']/100: tier = 'LOW'
        
        action, trade_val = "Hold", 0
        if is_first:
            trade_val = min(initial_cap * (params['initial_entry_pct'] / 100), max_cash_usage)
            shares = trade_val / price
            cash -= trade_val
            action, is_first = "First Buy", False
        else:
            diff = (shares * price) - (shares * prev_price)
            if diff > 0: # 상승 매도
                trade_val = diff * (params['vals'][f's_{tier}'] / 100)
                qty = trade_val / price
                shares -= qty; cash += trade_val; action = "Sell"
            elif diff < 0: # 하락 매수
                available = max_cash_usage - (initial_cap - cash)
                if available > 0:
                    trade_val = min(cash, abs(diff) * (params['vals'][f'b_{tier}'] / 100), available)
                    qty = trade_val / price
                    shares += qty; cash -= trade_val; action = "Buy"
        
        asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': asset})
        if action != "Hold":
            trade_logs.append({'날짜': date.strftime('%Y-%m-%d'), '상태': tier, '매매': action, '가격': round(price, 2), '거래금액': round(trade_val, 0)})
            
    return pd.DataFrame(history), trade_logs

# -----------------------------------------------------------
# 3. 사이드바 설정 (자동저장)
# -----------------------------------------------------------
settings = load_settings()
st.sidebar.header("⚙️ 기본 설정 (자동저장)")
with st.sidebar.form("settings_form"):
    s_date = st.date_input("투자 시작일", value=pd.to_datetime(settings['start_date']))
    s_cap = st.number_input("시작 원금 ($)", value=settings['initial_capital'], step=1000)
    s_max_cash = st.slider("최대 현금 투입 한도 (%)", 10, 100, settings['max_cash_pct'])
    s_init_entry = st.slider("초기 진입 비중 (%)", 0, 100, settings['initial_entry_pct'], step=5)
    
    st.markdown("---")
    st.write("**시장 평가 기준 (%)**")
    uh_c = st.number_input("UHIGH >", value=settings['uh_c'], step=0.5)
    h_c = st.number_input("HIGH >", value=settings['h_c'], step=0.5)
    l_c = st.number_input("LOW <", value=settings['l_c'], step=0.5)
    ul_c = st.number_input("ULOW <", value=settings['ul_c'], step=0.5)
    
    submit = st.form_submit_button("🔄 설정 저장 및 갱신")

if submit:
    settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_cash, 'initial_entry_pct': s_init_entry, 'uh_c': uh_c, 'h_c': h_c, 'l_c': l_c, 'ul_c': ul_c})
    save_settings(settings); st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면 레이아웃
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 위대리 가이드"])
df_market = get_market_data()

# ===========================================================
# TAB 1: 실전 트레이딩
# ===========================================================
with tab1:
    if df_market is not None:
        last = df_market.iloc[-1]; prev = df_market.iloc[-2]
        eval_val = last['Eval']
        tier = 'MID'
        if eval_val > uh_c/100: tier = 'UHIGH'; color = 'red'
        elif eval_val > h_c/100: tier = 'HIGH'; color = 'orange'
        elif eval_val < ul_c/100: tier = 'ULOW'; color = 'green'
        elif eval_val < l_c/100: tier = 'LOW'; color = 'lightgreen'
        else: color = 'gray'
        
        price_chg = last['TQQQ'] - prev['TQQQ']
        price_pct = (price_chg / prev['TQQQ']) * 100
        
        trade_log = load_trade_log()
        cash_now = trade_log.iloc[0]['Total_Cash'] if not trade_log.empty else s_cap
        shares_now = trade_log.iloc[0]['Balance_Qty'] if not trade_log.empty else 0
        cash_ratio = (cash_now / (cash_now + shares_now * last['TQQQ'])) * 100 if (cash_now + shares_now) > 0 else 100
        week_num = (datetime.now().date() - s_date).days // 7 + 1

        # 상단 정보바
        st.markdown(f"""
            <div style="display:flex; justify-content:space-between; background:#f8f9fa; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:20px;">
                <div style="text-align:center; width:25%;">
                    <div class="sub-text">시장 모드</div>
                    <div style="font-size:22px; font-weight:bold; color:{color};">{tier} ({eval_val*100:.1f}%)</div>
                </div>
                <div style="text-align:center; width:25%; border-left:1px solid #ddd;">
                    <div class="sub-text">TQQQ 현재가</div>
                    <div style="font-size:22px; font-weight:bold;">${last['TQQQ']:.2f} <span style="font-size:14px; color:{'red' if price_chg>0 else 'blue'};">{price_pct:+.2f}%</span></div>
                </div>
                <div style="text-align:center; width:25%; border-left:1px solid #ddd;">
                    <div class="sub-text">현금 비중</div>
                    <div style="font-size:22px; font-weight:bold; color:#1E88E5;">{cash_ratio:.1f}%</div>
                </div>
                <div style="text-align:center; width:25%; border-left:1px solid #ddd;">
                    <div class="sub-text">매매 주차</div>
                    <div style="font-size:22px; font-weight:bold;">{week_num}주차</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 오늘 주문표
        st.subheader("📝 오늘 주문표 (Wedaeri Order)")
        c_ord1, c_ord2 = st.columns([1, 2])
        with c_ord1: manual_p = st.number_input("예상 종가 입력 ($)", value=float(last['TQQQ']), step=0.01)
        
        with c_ord2:
            diff_p = manual_p - last['TQQQ']
            decision, box_c = "관망 (Hold)", "#eee"
            if diff_p > 0:
                sell_val = (shares_now * diff_p) * (settings['vals'][f's_{tier}']/100)
                decision = f"📈 매도 (SELL): 약 ${sell_val:,.0f} ({int(sell_val/manual_p)}주)"; box_c = "#f8d7da"
            elif diff_p < 0:
                buy_val = abs(shares_now * diff_p) * (settings['vals'][f'b_{tier}']/100)
                decision = f"📉 매수 (BUY): 약 ${buy_val:,.0f} ({int(buy_val/manual_p)}주)"; box_c = "#d1e7dd"
            st.markdown(f'<div class="order-box" style="background:{box_c}; color:black;">{decision}</div>', unsafe_allow_html=True)

        # 계좌 현황
        st.divider()
        st.subheader("💰 내 계좌 현황")
        cur_asset = cash_now + (shares_now * manual_p)
        eval_pnl = cur_asset - s_cap
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("보유 수량", f"{shares_now:,.2f} 주")
        k2.metric("예수금", f"${cash_now:,.0f}")
        k3.metric("총 평가손익", f"${eval_pnl:,.0f}", f"{(eval_pnl/s_cap)*100:.1f}%")
        k4.metric("현재 총자산", f"${cur_asset:,.0f}")

        # 매매 로그 & 수익 일지 (수정 가능/접기)
        with st.expander("📋 매매 로그 및 수익 일지 수정", expanded=False):
            edited_log = st.data_editor(trade_log, num_rows="dynamic", use_container_width=True, key="log_editor")
            if st.button("💾 로그 저장"): edited_log.to_csv(LOG_FILE, index=False); st.rerun()
            
            st.write("---")
            prof_log = load_profit_log()
            edited_prof = st.data_editor(prof_log, num_rows="dynamic", use_container_width=True, key="prof_editor")
            if st.button("💾 수익일지 저장"): edited_prof.to_csv(PROFIT_FILE, index=False); st.rerun()

        # 자산 성장 그래프 (통합)
        st.subheader("📈 자산 성장 및 하락폭(MDD)")
        prof_data = load_profit_log()
        if not prof_data.empty:
            prof_data['Date'] = pd.to_datetime(prof_data['Date'])
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(prof_data['Date'], prof_data['Total_Asset'], color='#1E88E5', lw=2)
            ax1.set_yscale('log'); ax1.set_ylabel("Asset ($)")
            ax2 = ax1.twinx()
            peak = prof_data['Total_Asset'].cummax()
            dd = (prof_data['Total_Asset'] / peak - 1) * 100
            ax2.fill_between(prof_data['Date'], dd, 0, color='red', alpha=0.1)
            ax2.set_ylim(-100, 5); ax2.set_ylabel("MDD (%)")
            st.pyplot(fig)

# ===========================================================
# TAB 2: 백테스트 분석
# ===========================================================
with tab2:
    st.subheader("📊 [위대리] 전략 백테스트 분석")
    with st.form("bt_form"):
        c1, c2, c3 = st.columns(3)
        b_cap = c1.number_input("초기 자본 ($)", 10000)
        b_start = c2.date_input("검증 시작일", pd.to_datetime("2010-01-01"))
        b_end = c3.date_input("검증 종료일", datetime.now())
        run_bt = st.form_submit_button("🔄 분석 실행")

    if run_bt:
        res, logs = run_simulation(df_market, b_start, b_end, settings)
        if not res.empty:
            final_a = res.iloc[-1]['Asset']; total_r = (final_a/b_cap-1)*100
            days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
            cagr = ((final_a/b_cap)**(365/max(1,days))-1)*100
            res['Peak'] = res['Asset'].cummax(); res['DD'] = (res['Asset']/res['Peak']-1)*100
            mdd = res['DD'].min(); calmar = cagr/abs(mdd) if mdd!=0 else 0
            
            w_ret = res['Asset'].pct_change().dropna()
            sharpe = (w_ret.mean() / w_ret.std()) * np.sqrt(52)
            sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52)

            m1, m2, m3, m4 = st.columns(4); m5, m6, m7, m8 = st.columns(4)
            m1.metric("최종 수익금", f"${final_a-b_cap:,.0f}", f"{total_r:.1f}%")
            m2.metric("CAGR", f"{cagr:.1f}%"); m3.metric("MDD", f"{mdd:.1f}%")
            m4.metric("거래 횟수", f"{len(logs)}회")
            m5.metric("Calmar", f"{calmar:.2f}"); m6.metric("Sharpe", f"{sharpe:.2f}")
            m7.metric("Sortino", f"{sortino:.2f}"); m8.metric("최종 자산", f"${final_a:,.0f}")

            # 차트 & 연도별 성과 & 로그 (생략 없이 통합)
            fig_bt, ax_bt1 = plt.subplots(figsize=(12, 5))
            ax_bt1.plot(res['Date'], res['Asset'], color='#1E88E5'); ax_bt1.set_yscale('log')
            ax_bt2 = ax_bt1.twinx(); ax_bt2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.1)
            st.pyplot(fig_bt)

            st.write("**📅 연도별 성과 요약**")
            res['Year'] = res['Date'].dt.year
            y_p = []
            for y, g in res.groupby('Year'):
                r = (g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100
                y_p.append({'연도':y, '수익률':f"{r:.1f}%", 'MDD':f"{g['DD'].min():.1f}%"})
            st.table(pd.DataFrame(y_p).set_index('연도').T)
            
            st.write("**📋 상세 매매 로그**")
            st.dataframe(pd.DataFrame(logs).sort_values('날짜', ascending=False), use_container_width=True)

# ===========================================================
# TAB 3: 위대리 가이드
# ===========================================================
with tab3:
    st.markdown(f"""
    ### 📘 [위대리 v1.1] 매매 전략 가이드
    **위대리**는 TQQQ의 주간 변동성을 활용하여 장기 우상향의 복리 효과를 극대화하는 전략입니다.
    
    1. **핵심 원칙**: 매주 금요일 종가 기준, 전주 대비 **자산 변동분**의 특정 비율만큼 매매합니다.
    2. **시장 평가 (Tier)**: QQQ의 5년 지수 회귀 추세선을 기준으로 시장을 5단계로 나눕니다.
    3. **최적 파라미터 (용성님 Pick)**:
        * **UHIGH (Eval > {uh_c}%)**: 매도 150% / 매수 30% (적극적 과열 해소)
        * **HIGH (Eval > {h_c}%)**: 매도 100% / 매수 60%
        * **MID (중립)**: 매도 60% / 매수 60% (변동성 드래그 방지)
        * **LOW (Eval < {l_c}%)**: 매도 60% / 매수 120%
        * **ULOW (Eval < {ul_c}%)**: 매도 30% / 매수 200% (공격적 수량 확보)
    4. **리스크 관리**: 초기 투입 비중과 최대 현금 사용 한도를 통해Tail Risk를 방어합니다.
    """)
