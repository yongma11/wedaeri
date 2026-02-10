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
st.set_page_config(page_title="TQQQ Wedaeri Pro Trader", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .big-metric { font-size: 24px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 10px; font-weight: bold; border: 1px solid #ddd; }
    .sub-text { font-size: 14px; color: #666; }
    .strategy-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 파일 관리 및 설정 관리
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v4.json'
TRADE_LOG_FILE = 'wedaeri_trade_log_v4.csv'
PROFIT_LOG_FILE = 'wedaeri_profit_log_v4.csv'

def load_json(file, default):
    if os.path.exists(file):
        try:
            with open(file, 'r') as f: return json.load(f)
        except: return default
    return default

def save_json(file, data):
    with open(file, 'w') as f: json.dump(data, f)

# 초기 설정값 (용성님 요청: 25/01/01 시작)
default_conf = {
    'start_date': '2025-01-01',
    'initial_capital': 10000,
    'max_cash_pct': 100,
    'initial_entry_pct': 50
}
settings = load_json(SETTINGS_FILE, default_conf)

# -----------------------------------------------------------
# 2. 정밀 매매 엔진 및 데이터 로직
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
    # 2010년 결과를 위해 QQQ는 2000년부터 불러와서 Growth 미리 계산
    qqq = yf.download("QQQ", start="2000-01-01", progress=False, auto_adjust=True)['Close']
    tqqq = yf.download("TQQQ", start="2010-01-01", progress=False, auto_adjust=True)['Close']
    
    qqq_df = pd.DataFrame(qqq)
    qqq_df['Growth'] = calculate_growth_curve(qqq_df['Close'], qqq_df.index)
    
    # 병합 (TQQQ 시작일인 2010년부터 데이터 보장)
    df = pd.concat([qqq_df, tqqq], axis=1).dropna()
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
        
        # 실시간 자산 리밸런싱을 위한 총자산 및 현금 한도 계산
        total_asset = cash + (shares * price)
        dynamic_max_cash = total_asset * (params['max_cash_pct'] / 100)
        
        # 고정 파라미터 (용성님 최적화 수치)
        if mkt_eval > 0.10: tier, s_r, b_r = 'UHIGH', 1.50, 0.30
        elif mkt_eval > 0.05: tier, s_r, b_r = 'HIGH', 1.00, 0.60
        elif mkt_eval < -0.10: tier, s_r, b_r = 'ULOW', 0.30, 2.00
        elif mkt_eval < -0.06: tier, s_r, b_r = 'LOW', 0.60, 1.20
        else: tier, s_r, b_r = 'MID', 0.60, 0.60

        action, trade_val = "Hold", 0
        if is_first:
            trade_val = min(initial_cap * (params['initial_entry_pct'] / 100), dynamic_max_cash)
            shares = trade_val / price; cash -= trade_val
            action, is_first = "First Buy", False
        else:
            diff_val = (shares * price) - (shares * prev_p)
            if diff_val > 0: # 상승 매도
                trade_val = diff_val * s_r
                trade_val = min(trade_val, shares * price)
                shares -= (trade_val / price); cash += trade_val; action = "Sell"
            elif diff_val < 0: # 하락 매수
                trade_val = abs(diff_val) * b_r
                available_cash_limit = dynamic_max_cash - (total_asset - cash)
                trade_val = min(cash, trade_val, max(0, available_cash_limit))
                shares += (trade_val / price); cash -= trade_val; action = "Buy"

        asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': asset})
        logs.append({
            'Date': date.strftime('%Y-%m-%d'), 'Tier': tier, 'Eval': f"{mkt_eval*100:.1f}%",
            'Type': action, 'Price': round(price, 2), 'Trade_Val': round(trade_val, 0), 
            'Balance_Qty': round(shares, 4), 'Cash': round(cash, 0), 'Total_Asset': round(asset, 0)
        })
            
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 자동 동기화
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
    
    # [핵심] 실전 로그 및 수익 지표 자동 생성
    _, res_logs = run_wedaeri_engine(df_weekly, set_date, datetime.now(), settings)
    if res_logs:
        # 매매 로그 저장
        pd.DataFrame(res_logs).sort_values('Date', ascending=False).to_csv(TRADE_LOG_FILE, index=False)
        # 수익 일지 데이터 생성 및 저장
        p_records = []
        for l in res_logs:
            p_records.append({
                'Date': l['Date'], 'Total_Asset': l['Total_Asset'],
                'Profit': l['Total_Asset'] - set_cap,
                'Return_Pct': round((l['Total_Asset'] / set_cap - 1) * 100, 2)
            })
        pd.DataFrame(p_records).sort_values('Date', ascending=False).to_csv(PROFIT_LOG_FILE, index=False)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면 레이아웃
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 위대리 전략 가이드"])

# ===========================================================
# TAB 1: 실전 트레이딩
# ===========================================================
with tab1:
    last_row = df_weekly.iloc[-1]; prev_row = df_weekly.iloc[-2]
    eval_val = last_row['Eval']
    
    # 데이터 로드
    t_log = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
    p_log = pd.read_csv(PROFIT_LOG_FILE) if os.path.exists(PROFIT_LOG_FILE) else pd.DataFrame()
    
    cash_now = t_log.iloc[0]['Cash'] if not t_log.empty else set_cap
    shares_now = t_log.iloc[0]['Balance_Qty'] if not t_log.empty else 0
    total_now = cash_now + (shares_now * last_row['TQQQ'])
    week_idx = (datetime.now().date() - pd.to_datetime(settings['start_date']).date()).days // 7 + 1
    
    # 1. 상단 정보바
    m_tier = 'MID'; m_col = 'gray'
    if eval_val > 0.10: m_tier = 'UHIGH'; m_col = 'red'
    elif eval_val > 0.05: m_tier = 'HIGH'; m_col = 'orange'
    elif eval_val < -0.10: m_tier = 'ULOW'; m_col = 'green'
    elif eval_val < -0.06: m_tier = 'LOW'; m_col = 'lightgreen'

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="sub-text">시장모드 (평가율)</div><div class="big-metric" style="color:{m_col};">{m_tier} ({eval_val*100:.1f}%)</div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="sub-text">TQQQ 현재가</div><div class="big-metric">${last_row["TQQQ"]:.2f} <span style="font-size:14px; color:{"red" if last_row["TQQQ"]>prev_row["TQQQ"] else "blue"};">{(last_row["TQQQ"]/prev_row["TQQQ"]-1)*100:+.2f}%</span></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="sub-text">현금 비중</div><div class="big-metric">{(cash_now/total_now*100 if total_now>0 else 100):.1f}%</div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="sub-text">매매 주차</div><div class="big-metric">{week_idx}주차</div>', unsafe_allow_html=True)

    # 2. 오늘 주문표
    st.subheader("📝 오늘 주문표 (Daily Order)")
    co1, co2 = st.columns([1, 2])
    with co1: est_p = st.number_input("예상 종가 입력 ($)", value=float(last_row['TQQQ']), step=0.01)
    with co2:
        diff_p = est_p - last_row['TQQQ']
        decision, b_c = "관망 (Hold)", "#f8f9fa"
        if diff_p > 0:
            v = (shares_now * diff_p) * (1.5 if m_tier=='UHIGH' else 1.0 if m_tier=='HIGH' else 0.6)
            decision = f"📈 매도 (SELL): 약 ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#fff5f5"
        elif diff_p < 0:
            v = abs(shares_now * diff_p) * (2.0 if m_tier=='ULOW' else 1.2 if m_tier=='LOW' else 0.6)
            decision = f"📉 매수 (BUY): 약 ${v:,.0f} ({v/est_p:.2f}주)"; b_c = "#f0fff4"
        st.markdown(f'<div class="order-box" style="background:{b_c}; color:black;">{decision}</div>', unsafe_allow_html=True)

    # 3. 계좌 현황
    st.divider()
    st.subheader("💰 내 계좌 현황")
    eval_ret = (total_now / set_cap - 1) * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TQQQ 보유수량", f"{shares_now:,.2f} 주")
    k2.metric("예수금", f"${cash_now:,.0f}")
    k3.metric("총 평가손익", f"${total_now - set_cap:,.0f}", f"{eval_ret:.1f}%")
    k4.metric("현재 총자산", f"${total_now:,.0f}")

    # 4. 매매로그 및 수익일지 (수정가능)
    with st.expander("📋 매매 로그 및 수익 일지 기록", expanded=False):
        st.write("**실전 매매 로그**")
        ed_log = st.data_editor(t_log, num_rows="dynamic", use_container_width=True, key="real_trade_editor")
        if st.button("💾 매매로그 저장"): ed_log.to_csv(TRADE_LOG_FILE, index=False); st.rerun()
        
        st.write("---")
        st.write("**수익 일지 (날짜 / 총자산 / 수익금 / 수익률)**")
        ed_prof = st.data_editor(p_log, num_rows="dynamic", use_container_width=True, key="real_profit_editor")
        if st.button("💾 수익일지 저장"): ed_prof.to_csv(PROFIT_LOG_FILE, index=False); st.rerun()

    # 5. 자산 성장 그래프
    if not p_log.empty:
        st.subheader("📈 실전 자산 성장 및 하락폭")
        p_log['Date'] = pd.to_datetime(p_log['Date'])
        p_plot = p_log.sort_values('Date')
        fig_r, ax_r1 = plt.subplots(figsize=(12, 5))
        ax_r1.plot(p_plot['Date'], p_plot['Total_Asset'], color='#1E88E5', lw=2, label="총자산 ($)")
        ax_r1.set_ylabel("Asset Value ($)")
        ax_r1.grid(True, alpha=0.2)
        ax_r2 = ax_r1.twinx()
        peak = p_plot['Total_Asset'].cummax()
        dd = (p_plot['Total_Asset'] / peak - 1) * 100
        ax_r2.fill_between(p_plot['Date'], dd, 0, color='red', alpha=0.1, label="Drawdown (%)")
        ax_r2.set_ylim(-100, 5)
        st.pyplot(fig_r)

# ===========================================================
# TAB 2: 백테스트 분석
# ===========================================================
with tab2:
    st.subheader("📊 전략 정밀 백테스트 (2010~현재)")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("검증 자본 ($)", 10000)
        bt_start = bc2.date_input("검증 시작일", pd.to_datetime("2010-02-12")) # TQQQ 탄생일
        bt_end = bc3.date_input("검증 종료일", datetime.now())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        res, logs = run_wedaeri_engine(df_weekly, bt_start, bt_end, {'initial_capital': bt_cap, 'max_cash_pct': settings['max_cash_pct'], 'initial_entry_pct': settings['initial_entry_pct']})
        
        if not res.empty:
            # 지표 계산
            final_v = res.iloc[-1]['Asset']
            total_r = (final_v / bt_cap - 1) * 100
            days = (res.iloc[-1]['Date'] - res.iloc[0]['Date']).days
            cagr = ((final_v / bt_cap) ** (365 / max(1, days)) - 1) * 100
            res['Peak'] = res['Asset'].cummax()
            res['DD'] = (res['Asset'] / res['Peak'] - 1) * 100
            mdd = res['DD'].min()
            
            # Sharpe, Sortino, Calmar
            w_ret = res['Asset'].pct_change().dropna()
            sharpe = (w_ret.mean() / w_ret.std()) * np.sqrt(52)
            down_ret = w_ret[w_ret < 0]
            sortino = (w_ret.mean() / down_ret.std()) * np.sqrt(52) if not down_ret.empty else 0
            calmar = cagr / abs(mdd) if mdd != 0 else 0

            # 1. 종합 성과 리포트
            m1, m2, m3, m4 = st.columns(4); m5, m6, m7, m8 = st.columns(4)
            m1.metric("최종 수익금", f"${final_v - bt_cap:,.0f}", f"{total_r:.1f}%")
            m2.metric("CAGR (연평균)", f"{cagr:.2f}%")
            m3.metric("MDD (최대낙폭)", f"{mdd:.1f}%")
            m4.metric("거래 횟수", f"{len([l for l in logs if l['Type'] != 'Hold'])}회")
            m5.metric("Calmar 지수", f"{calmar:.2f}")
            m6.metric("샤프 지수", f"{sharpe:.2f}")
            m7.metric("소르티노 지수", f"{sortino:.2f}")
            m8.metric("최종 자산", f"${final_v:,.0f}")

            # 2. 성과 차트
            fig_bt, ax_b1 = plt.subplots(figsize=(12, 5))
            ax_b1.plot(res['Date'], res['Asset'], color='#1E88E5', lw=2)
            ax_b1.set_yscale('log')
            ax_b1.set_title("자산 성장 곡선 (로그 스케일)")
            ax_b2 = ax_b1.twinx()
            ax_b2.fill_between(res['Date'], res['DD'], 0, color='red', alpha=0.1)
            ax_b2.set_ylim(-100, 5)
            st.pyplot(fig_bt)

            # 3. 연도별 성과
            st.subheader("📅 연도별 성과 요약")
            res['Year'] = res['Date'].dt.year
            y_p = []
            for y, g in res.groupby('Year'):
                y_p.append({'연도':y, '수익률':f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD':f"{(g['Asset']/g['Asset'].cummax()-1).min()*100:.1f}%"})
            st.table(pd.DataFrame(y_p).set_index('연도').T)

            # 4. 상세 매매 로그
            st.subheader("📋 백테스트 상세 매매 로그")
            st.dataframe(pd.DataFrame(logs).sort_values('날짜' if '날짜' in pd.DataFrame(logs).columns else 'Date', ascending=False), use_container_width=True)

# ===========================================================
# TAB 3: 위대리 전략 가이드 (상세 설명)
# ===========================================================
with tab3:
    st.markdown("""
    <div class="strategy-card">
        <h2>📘 위대리(Wedaeri) v1.1 매매 전략 상세 가이드</h2>
        <p><b>위대리</b>는 나스닥 3배 레버리지 ETF인 <b>TQQQ</b>를 대상으로, 시장의 장기 추세와 단기 변동성을 동시에 공략하는 <b>정량적 리밸런싱 전략</b>입니다.</p>
    </div>
    
    <h3>1. 핵심 철학: "변동성을 수익으로 전환"</h3>
    <ul>
        <li><b>레버리지 드래그 방어:</b> 하락장에서 무지성 홀딩 시 발생하는 레버리지 침식 현상을 방어하기 위해, 매주 정해진 로직에 따라 주식과 현금의 비중을 조절합니다.</li>
        <li><b>유동적 현금 방패:</b> 자산이 커지면 그에 비례하여 현금 보유 한도를 늘려, 대공황급 하락장에서도 계좌가 녹지 않도록 설계되었습니다.</li>
    </ul>

    <h3>2. 시장 평가 (Tier System)</h3>
    <p>QQQ의 <b>5년(1260영업일) 지수 회귀 추세선(Log-Linear Growth)</b>을 기준으로 현재 가격이 얼마나 과열되었거나 저평가되었는지 판단합니다.</p>
    <ul>
        <li><b>UHIGH (초고평가 > 10%):</b> 시장이 매우 과열됨. 수익 실현 비중을 최대화하여 현금을 대량 확보합니다.</li>
        <li><b>HIGH (고평가 > 5%):</b> 완만한 고점권. 점진적으로 수익을 챙깁니다.</li>
        <li><b>MID (중립):</b> 추세에 순응하며 기본 비중 유지.</li>
        <li><b>LOW (저평가 < -6%):</b> 하락세 진입. 현금을 투입하여 수량을 늘리기 시작합니다.</li>
        <li><b>ULOW (초저평가 < -10%):</b> 바닥권 기회. 강력한 현금 투입으로 평단가를 낮추고 반등을 준비합니다.</li>
    </ul>

    <h3>3. 매매 로직 (The Engine)</h3>
    <p>매주 금요일 종가 기준, <b>전주 대비 자산 변동분($)</b>에 티어별 매매율을 곱해 거래합니다.</p>
    <ul>
        <li><b>상승 시 (매도):</b> (보유 주식 가치 상승분 × 티어별 매도율) 만큼 팔아 현금화.</li>
        <li><b>하락 시 (매수):</b> (보유 주식 가치 하락분 × 티어별 매수율) 만큼 현금을 투입해 추가 매수.</li>
    </ul>

    <h3>4. 리스크 관리 (Safety First)</h3>
    <ul>
        <li><b>최대 현금 투입 한도:</b> 계좌가 감당할 수 있는 현금 사용량을 설정하여 예측 불가능한 하락에 대비합니다.</li>
        <li><b>유동적 리밸런싱:</b> 매주 현재 총자산을 기준으로 현금 한도를 재계산하므로, 복리 성장의 혜택을 온전히 누리면서도 방어력은 유지됩니다.</li>
    </ul>
    """, unsafe_allow_html=True)
