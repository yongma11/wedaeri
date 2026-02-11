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
st.set_page_config(page_title="Wedaeri Quantum T-Flow v1.5", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .program-title { font-size: 38px; font-weight: 900; color: #0D47A1; text-align: center; margin-bottom: 20px; letter-spacing: -1px; }
    .big-metric { font-size: 26px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 12px; font-weight: bold; border: 2px solid #1E88E5; background-color: #f0f7ff; margin-top: 10px; }
    .status-bar { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 6px solid #1E88E5; }
    .strategy-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #e0e0e0; line-height: 1.8; margin-bottom: 20px; }
    .tier-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# [요청 반영] 멋진 제목 상단 배치
st.markdown('<div class="program-title">Wedaeri Quantum T-Flow</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 및 데이터 관리 (기존 로직 유지)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v15.json'
default_settings = {
    'start_date': '2025-01-01', 'initial_capital': 10000, 'max_cash_pct': 100, 'initial_entry_pct': 50,
    'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
    'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
    'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try: return json.load(f)
            except: return default_settings
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f: json.dump(data, f)

if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

@st.cache_data(ttl=600)
def fetch_weekly_data():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
    window = 1260
    qqq_vals, date_nums = df['QQQ'].values, df.index.map(pd.Timestamp.toordinal).values
    growth = [np.nan] * len(df)
    for i in range(window, len(df)):
        fit = np.polyfit(date_nums[i-window : i], np.log(qqq_vals[i-window : i]), 1)
        growth[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    df['Growth'], df['Eval'] = growth, (df['QQQ'] / growth) - 1
    weekly_df = df[df.index.weekday == 4].copy()
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df

# -----------------------------------------------------------
# 2. 엔진 로직 (명칭 변경 반영)
# -----------------------------------------------------------
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
        tier = 'MID'
        if mkt_eval > params['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < params['low_cut']/100: tier = 'LOW'

        action, trade_val, trade_qty = "관망", 0, 0
        s_r, b_r = params['sell_ratios'][tier]/100, params['buy_ratios'][tier]/100

        if is_first:
            trade_val = min(cap * (params['initial_entry_pct'] / 100), max_c_usage)
            trade_qty = trade_val / price
            shares = trade_qty; cash -= trade_val; action, is_first = "매수", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0:
                trade_val = diff * s_r
                trade_qty = min(trade_val / price, shares)
                shares -= trade_qty; cash += (trade_qty * price); action = "매도"
                trade_qty = -trade_qty # 매도는 마이너스 표시
            elif diff < 0:
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_val = min(cash, abs(diff) * b_r, avail)
                    trade_qty = trade_val / price
                    shares += trade_qty; cash -= trade_val; action = "매수"
        
        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        # [요청 반영] 명칭 변경 및 컬럼 재배치
        logs.append({
            '날짜': date.strftime('%Y-%m-%d'), '시장평가': tier, '매매': action, 
            '가격 ($)': f"${price:.2f}", '거래수량': f"{trade_qty:+.2f}",
            '보유수량': round(shares, 2), '평가금 ($)': round(shares * price, 0),
            '예수금 ($)': round(cash, 0), '총자산 ($)': round(current_asset, 0)
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 메인 화면 레이아웃
# -----------------------------------------------------------
df_weekly = fetch_weekly_data()

with st.sidebar:
    st.header("⚙️ System Config")
    with st.form("settings_form"):
        s_date = st.date_input("투자 시작일", value=pd.to_datetime(st.session_state.settings['start_date']))
        s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
        s_max_c = st.slider("현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
        s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
        save_btn = st.form_submit_button("💾 설정 저장 및 동기화", type="primary")

if save_btn:
    st.session_state.settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p})
    save_settings(st.session_state.settings); st.rerun()

tab1, tab2, tab3 = st.tabs(["🚀 실전 대시보드", "📊 백테스트 분석", "📘 매매전략 가이드"])

# --- TAB 1: 실전 대시보드 ---
with tab1:
    res_df, res_logs = run_engine(df_weekly, st.session_state.settings['start_date'], st.session_state.settings)
    last_mkt = df_weekly.iloc[-1]
    st.markdown(f'<div class="status-bar"><b>📅 분석 기준일:</b> {df_weekly.index[-1].strftime("%Y-%m-%d")} | <b>💎 TQQQ 종가:</b> ${last_mkt["TQQQ"]:.2f}</div>', unsafe_allow_html=True)
    
    if res_logs:
        last = res_logs[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("보유수량", f"{last['보유수량']:,.2f} 주")
        c2.metric("평가금", f"${last['평가금 ($)']:,.0f}")
        c3.metric("예수금", f"${last['예수금 ($)']:,.0f}")
        c4.metric("총자산", f"${last['총자산 ($)']:,.0f}")
        
        st.divider()
        st.subheader("🎯 이번 주 예상 주문")
        col_in, col_out = st.columns([1, 1.5])
        with col_in: est_p = st.number_input("예상 종가 ($)", value=float(last_mkt['TQQQ']), step=0.01)
        with col_out:
            diff = est_p - last_mkt['TQQQ']
            order_msg = "관망"
            if diff > 0:
                val = (last['보유수량'] * diff) * (st.session_state.settings['sell_ratios'][last['시장평가']]/100)
                order_msg = f"매도: ${val:,.0f} ({val/est_p:.2f} 주)"
            elif diff < 0:
                max_u = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
                avail = max_u - (st.session_state.settings['initial_capital'] - last['예수금 ($)'])
                val = min(last['예수금 ($)'], abs(last['보유수량'] * diff) * (st.session_state.settings['buy_ratios'][last['시장평가']]/100), max(0, avail))
                order_msg = f"매수: ${val:,.0f} ({val/est_p:.2f} 주)"
            st.markdown(f'<div class="order-box">{order_msg}</div>', unsafe_allow_html=True)

        st.subheader("📜 상세 매매로그")
        st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)

# --- TAB 2: 백테스트 분석 ---
with tab2:
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("테스트 원금 ($)", value=10000)
        bt_start = bc2.date_input("시작일", value=date(2010, 2, 12))
        bt_end = bc3.date_input("종료일", value=date.today())
        run_bt = st.form_submit_button("🚀 분석 실행")

    if run_bt:
        bt_params = st.session_state.settings.copy(); bt_params['initial_capital'] = bt_cap
        b_df, b_logs = run_engine(df_weekly[df_weekly.index <= pd.to_datetime(bt_end)], bt_start.strftime('%Y-%m-%d'), bt_params)
        if not b_df.empty:
            final_v = b_df.iloc[-1]['Asset']; ret = (final_v / bt_cap - 1) * 100
            days = (b_df.iloc[-1]['Date'] - b_df.iloc[0]['Date']).days
            cagr = ((final_v / bt_cap) ** (365 / max(1, days)) - 1) * 100
            b_df['Peak'] = b_df['Asset'].cummax(); b_df['DD'] = (b_df['Asset'] / b_df['Peak'] - 1) * 100
            mdd = b_df['DD'].min(); calmar = cagr / abs(mdd) if mdd != 0 else 0
            w_ret = b_df['Asset'].pct_change().dropna()
            sortino = (w_ret.mean() / w_ret[w_ret<0].std()) * np.sqrt(52) if not w_ret[w_ret<0].empty else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("수익률", f"{ret:.1f}%"); m2.metric("CAGR", f"{cagr:.1f}%"); m3.metric("MDD", f"{mdd:.1f}%"); m4.metric("칼마", f"{calmar:.2f}"); m5.metric("소르티노", f"{sortino:.2f}")

            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', lw=2, label='자산 (Log)')
            ax1.set_yscale('log'); ax2 = ax1.twinx()
            ax2.fill_between(b_df['Date'], b_df['DD'], 0, color='#E53935', alpha=0.2, label='MDD')
            ax2.set_ylim(-100, 5); st.pyplot(fig)
            
            # 연도별 성과
            b_df['Year'] = b_df['Date'].dt.year
            y_data = [{'연도': y, '수익률': f"{(g.iloc[-1]['Asset']/g.iloc[0]['Asset']-1)*100:.1f}%", 'MDD': f"{(g['Asset']/g['Asset'].cummax()-1).min()*100:.1f}%"} for y, g in b_df.groupby('Year')]
            st.table(pd.DataFrame(y_data).set_index('연도'))
            st.dataframe(pd.DataFrame(b_logs).sort_values('날짜', ascending=False), use_container_width=True)

# --- TAB 3: [요청 반영] 매매전략 상세 가이드 ---
with tab3:
    st.markdown("""
    <div class="strategy-card">
        <h2>📘 Wedaeri Quantum T-Flow 전략 가이드</h2>
        <p>본 전략은 나스닥 100 지수의 3배 레버리지인 <b>TQQQ</b>의 극심한 변동성을 <b>역사적 로그 회귀선</b>을 통해 수익으로 전환하는 정량적 리밸런싱 시스템입니다.</p>
        
        <h3>1. 시장 평가 지표 (Market Evaluation)</h3>
        <p>나스닥 지수(QQQ)의 지난 25년간의 성장 궤적을 분석하여 현재 주가가 역사적 추세 대비 어느 위치에 있는지를 측정합니다.</p>
        <ul>
            <li><b>UHIGH / HIGH (고평가):</b> 시장이 과열된 상태입니다. 가격 상승 시 적극적으로 이익을 실현하여 현금을 확보합니다.</li>
            <li><b>MID (중립):</b> 시장이 추세선 근처에 있습니다. 완만한 리밸런싱을 통해 비중을 조절합니다.</li>
            <li><b>LOW / ULOW (저평가):</b> 시장이 공포에 빠진 상태입니다. 가격 하락 시 확보해둔 현금으로 저가 매수를 단행하여 수량을 극대화합니다.</li>
        </ul>
        
        <h3>2. 주간 리밸런싱 원칙</h3>
        <p>매일의 시장 소음(Noise)에 휘둘리지 않기 위해 <b>매주 금요일 종가</b>를 기준으로만 매매를 검토합니다.</p>
        
        <h3>3. 변동성 매매 로직 (Volatility Trading)</h3>
        <p>단순히 사고 버티는(Buy & Hold) 전략이 아니라, 주가 변동에 따른 <b>평가금액의 변화분</b>을 티어별 비율로 매수/매도합니다.</p>
        <ul>
            <li><b>상승 시:</b> 증가한 평가금의 일정 비율만큼 매도하여 수익을 확정합니다.</li>
            <li><b>하락 시:</b> 감소한 평가금의 일정 비율만큼 추가 매수하여 평균 단가를 낮추고 수량을 늘립니다.</li>
        </ul>
        
        <h3>4. 현금 방패 (Cash Shield)</h3>
        <p>설정된 <b>현금 투입 한도</b>를 통해 폭락장에서도 마지막까지 대응할 수 있는 현금을 보존합니다. 이는 MDD(최대 낙폭)를 획기적으로 줄여주는 심리적, 기술적 방어선입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    # 시각적 이해를 돕기 위한 이미지 태그 추가
    st.write("### 💡 이해를 돕기 위한 보조 자료")
