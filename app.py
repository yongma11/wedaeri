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
SETTINGS_FILE = 'wedaeri_settings_v11.json'
TRADE_LOG_FILE = 'wedaeri_trade_log_v11.csv'

def load_json():
    default = {
        'start_date': '2010-01-01',
        'initial_capital': 10000,
        'max_cash_pct': 100,
        'initial_entry_pct': 50,
        'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
        'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
        'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
            default.update(data)
    return default

def save_json(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

settings = load_json()

# -----------------------------------------------------------
# 2. 핵심 엔진 (오전 8시 자동 업데이트 캐시 적용)
# -----------------------------------------------------------
def get_ttl():
    """오전 8시가 되면 캐시가 만료되도록 설정"""
    now = datetime.now()
    next_8am = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now >= next_8am:
        next_8am += timedelta(days=1)
    return (next_8am - now).total_seconds()

def calculate_growth_curve(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        y_train = values[i-window : i]
        x_train = date_nums[i-window : i]
        try:
            fit = np.polyfit(x_train, np.log(y_train), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        except: pass
    return pd.Series(results, index=series.index)

@st.cache_data(ttl=get_ttl())
def fetch_data():
    # 용성님, 데이터는 2000년부터 가져와서 2010년 이전 데이터를 예열합니다.
    df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close']
    df = df_raw.dropna(subset=['QQQ'])
    df['Growth'] = calculate_growth_curve(df['QQQ'], df.index)
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    
    # 금요일 종가 기준 주간 데이터 생성
    weekly = df[df.index.weekday == 4].copy()
    weekly['TQQQ_Prev'] = weekly['TQQQ'].shift(1)
    return weekly

def run_engine(df, start_dt, end_dt, params):
    sim_data = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
    if sim_data.empty: return pd.DataFrame(), []

    cash, shares, is_first = params['initial_capital'], 0, True
    history, logs = [], []
    max_cash_limit = params['initial_capital'] * (params['max_cash_pct'] / 100)
    total_invested_cost = 0 # 평단가 계산용

    for date, row in sim_data.iterrows():
        price, prev_p, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        
        tier = 'MID'
        if mkt_eval > params['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < params['low_cut']/100: tier = 'LOW'

        action, trade_val = "Hold", 0
        if is_first:
            trade_val = min(params['initial_capital'] * (params['initial_entry_pct'] / 100), max_cash_limit)
            shares = trade_val / price
            cash -= trade_val
            total_invested_cost = trade_val
            action, is_first = "First Buy", False
        else:
            if np.isnan(prev_p): continue
            diff_val = (shares * price) - (shares * prev_p)
            if diff_val > 0: # 수익 시 매도
                trade_val = diff_val * (params['sell_ratios'][tier] / 100)
                actual_sell = min(trade_val / price, shares)
                # 매도 시 평단가는 유지, 투자원금 비중 축소
                avg_p = total_invested_cost / shares if shares > 0 else 0
                total_invested_cost -= (actual_sell * avg_p)
                shares -= actual_sell; cash += (actual_sell * price)
                action = "Sell"; trade_val = actual_sell * price
            elif diff_val < 0: # 손실 시 매수
                available = max_cash_limit - (params['initial_capital'] - cash)
                if available > 0:
                    trade_val = min(cash, abs(diff_val) * (params['buy_ratios'][tier] / 100), available)
                    actual_buy = trade_val / price
                    shares += actual_buy; cash -= trade_val
                    total_invested_cost += trade_val
                    action = "Buy"

        curr_asset = cash + (shares * price)
        avg_price = total_invested_cost / shares if shares > 0 else 0
        history.append({'Date': date, 'Asset': curr_asset})
        logs.append({
            'Date': date.strftime('%Y-%m-%d'), 'Tier': tier, 'Type': action, 
            'Price': round(price, 2), 'Trade_Val': round(trade_val, 0),
            'Shares': round(shares, 2), 'Avg_Price': round(avg_price, 2),
            'Cash': round(cash, 0), 'Total_Asset': round(curr_asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 동기화
# -----------------------------------------------------------
df_weekly = fetch_data()

st.sidebar.header("👤 용성님 계좌 설정")
with st.sidebar.form("user_config"):
    set_date = st.date_input("투자 시작일", value=pd.to_datetime(settings['start_date']))
    set_cap = st.number_input("초기 투자금 ($)", value=settings['initial_capital'], step=1000)
    sync_btn = st.form_submit_button("🔄 설정 저장 및 동기화")

if sync_btn:
    settings.update({'start_date': set_date.strftime('%Y-%m-%d'), 'initial_capital': set_cap})
    save_json(settings)
    _, real_logs = run_engine(df_weekly, set_date, datetime.now(), settings)
    pd.DataFrame(real_logs).sort_values('Date', ascending=False).to_csv(TRADE_LOG_FILE, index=False)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 대시보드
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 실전 대시보드", "📉 백테스트 분석", "📘 위대리 가이드북"])

with tab1:
    t_log = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
    last_price = df_weekly.iloc[-1]['TQQQ']
    
    if not t_log.empty:
        curr = t_log.iloc[0]
        st.subheader("🏦 현재 계좌 상태")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("주식 잔고", f"{curr['Shares']:,} 주")
        m2.metric("보유 평단가", f"${curr['Avg_Price']:.2f}")
        m3.metric("예수금 (Cash)", f"${curr['Cash']:,.0f}")
        m4.metric("총 자산", f"${curr['Total_Asset']:,.0f}", f"{(curr['Total_Asset']/settings['initial_capital']-1)*100:.1f}%")

        st.divider()
        st.subheader("📋 실전 매매 기록 및 성과")
        with st.expander("매매 기록 상세보기", expanded=False):
            st.dataframe(t_log, use_container_width=True)
        
        # 수익 그래프
        fig_real, ax_real = plt.subplots(figsize=(12, 4))
        plot_df = t_log.sort_values('Date')
        ax_real.plot(pd.to_datetime(plot_df['Date']), plot_df['Total_Asset'], color='#1E88E5', lw=2)
        ax_real.fill_between(pd.to_datetime(plot_df['Date']), plot_df['Total_Asset'], settings['initial_capital'], alpha=0.1, color='blue')
        ax_real.set_title("실전 자산 성장 곡선"); ax_real.grid(True, alpha=0.2)
        st.pyplot(fig_real)
    else:
        st.info("사이드바에서 동기화를 먼저 진행해 주세요.")

with tab2:
    st.subheader("🧪 기간별 백테스트 실행")
    c1, c2, c3 = st.columns(3)
    bt_s = c1.date_input("백테스트 시작", value=pd.to_datetime("2010-02-12"))
    bt_e = c2.date_input("백테스트 종료", value=datetime.now())
    run_bt = c3.button("🚀 분석 시작", use_container_width=True)

    if run_bt:
        res, b_logs = run_engine(df_weekly, bt_s, bt_e, settings)
        if not res.empty:
            final_v = res.iloc[-1]['Asset']
            ret = (final_v / settings['initial_capital'] - 1) * 100
            days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
            cagr = ((final_v / settings['initial_capital']) ** (365 / max(1, days)) - 1) * 100
            res['Peak'] = res['Asset'].cummax(); res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
            mdd = res['DD'].min()
            
            w_ret = res['Asset'].pct_change().dropna()
            sharpe = (w_ret.mean() / w_ret.std()) * np.sqrt(52) if w_ret.std() != 0 else 0
            sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52) if not w_ret[w_ret<0].empty else 0
            calmar = cagr / abs(mdd) if mdd != 0 else 0
            
            # 손익비 계산
            trades = pd.DataFrame(b_logs)
            plus = trades[trades['Type'] == 'Sell']['Trade_Val'].sum()
            minus = trades[trades['Type'] == 'Buy']['Trade_Val'].sum()
            profit_factor = plus / minus if minus != 0 else 0

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("최종자산", f"${final_v:,.0f}", f"{ret:.1f}%")
            k2.metric("CAGR", f"{cagr:.2f}%")
            k3.metric("MDD", f"{mdd:.1f}%")
            k4.metric("Calmar / Sortino", f"{calmar:.2f} / {sortino:.2f}")
            k5.metric("손익비 (PF)", f"{profit_factor:.2f}")

            st.pyplot(plt.figure(figsize=(10,3))) # 간격용
            st.subheader("📅 연도별 성과 요약")
            res['Year'] = res['Date'].dt.year
            y_p = [{'연도':y, '수익률':f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD':f"{(g['Asset']/g['Asset'].cummax()-1).min()*100:.1f}%"} for y, g in res.groupby('Year')]
            st.table(pd.DataFrame(y_p).set_index('연도').T)
            
            st.subheader("📜 상세 매매 로그 (백테스트 기간 내)")
            st.dataframe(pd.DataFrame(b_logs).sort_values('Date', ascending=False), use_container_width=True)

with tab3:
    st.markdown(f"""
    <div class="strategy-card">
        <h2>📘 위대리(Wedaeri) v1.1 Pro 매매법 가이드</h2>
        <p>용성님, 이 매매법은 <b>'시장은 결국 우상향하지만, 그 과정의 변동성을 수익으로 바꾼다'</b>는 철학을 담고 있습니다.</p>
        
        <h3>1. 핵심 원리: QQQ 로그 회귀</h3>
        <p>나스닥(QQQ)의 지난 25년간의 성장 궤적을 수학적으로 계산합니다. 현재 주가가 이 궤적보다 <b>높으면(과열)</b> 팔고, <b>낮으면(저평가)</b> 더 사는 방식입니다. 이 기준점 덕분에 폭락장에서도 공포에 질리지 않고 기계적으로 매수할 수 있습니다.</p>
        
        <h3>2. TQQQ를 쓰는 이유</h3>
        <p>TQQQ는 나스닥 지수를 3배로 추종합니다. 변동성이 매우 크지만, 위대리 전략의 <b>'티어별 리밸런싱'</b>을 통해 하락장에서는 수량을 폭발적으로 늘리고, 상승장에서는 이익을 실현하여 복리 효과를 극대화합니다.</p>

        <h3>3. 티어별 대응 공식 (자동 적용)</h3>
        <ul>
            <li><b>UHIGH (초과열):</b> 시장이 너무 뜨겁습니다. 상승 시 수익금의 150%를 매도해 현금을 두둑히 챙깁니다.</li>
            <li><b>MID (평범):</b> 추세에 몸을 맡깁니다. 변동폭의 60% 정도만 리밸런싱합니다.</li>
            <li><b>ULOW (대폭락):</b> 기회입니다! 하락 시 손실액의 200%를 추가 매수하여 수량을 공격적으로 모읍니다.</li>
        </ul>

        <h3>4. 운용 팁</h3>
        <ul>
            <li><b>금요일 밤:</b> 매주 금요일 종가(한국시간 토요일 새벽)를 기준으로 한 번만 확인하시면 됩니다.</li>
            <li><b>오전 8시 업데이트:</b> 본 대시보드는 매일 오전 8시에 최신 시장 데이터를 자동으로 가져와 분석합니다.</li>
            <li><b>인내심:</b> 백테스트 결과에서 보듯, MDD(최대 하락폭)를 견뎌내면 CAGR(연복리 수익률)이 보답할 것입니다.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
