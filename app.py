import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
import json
from datetime import datetime, date, timedelta

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri Quantum T-Flow v1.3", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .big-metric { font-size: 26px !important; font-weight: bold; color: #1E88E5; }
    .order-box { text-align: center; padding: 20px; border-radius: 12px; font-weight: bold; border: 2px solid #1E88E5; background-color: #f0f7ff; margin-top: 10px; }
    .status-bar { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 6px solid #1E88E5; }
    .program-title { font-size: 28px; font-weight: 800; color: #0D47A1; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 관리 (저장 및 동기화 기능 강화)
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v13.json'

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
                return json.load(f)
            except: return default_settings
    return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

# -----------------------------------------------------------
# 2. 데이터 엔진 (최신 날짜 반영 로직)
# -----------------------------------------------------------
@st.cache_data(ttl=600) # 10분마다 갱신
def fetch_data():
    # TQQQ와 QQQ 최신 데이터를 오늘 날짜까지 가져옴
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
    
    # 추세선 계산 (이격도 분석용)
    window = 1260
    results = [np.nan] * len(df)
    date_nums = df.index.map(pd.Timestamp.toordinal).values
    qqq_vals = df['QQQ'].values
    for i in range(window, len(df)):
        y_train = qqq_vals[i-window : i]
        x_train = date_nums[i-window : i]
        fit = np.polyfit(x_train, np.log(y_train), 1)
        results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    
    df['Growth'] = results
    df['Eval'] = (df['QQQ'] / df['Growth']) - 1
    
    # 매매 기준일 설정 (매일 업데이트를 위해 일간 데이터 유지하되, 전략상 전일 종가 비교)
    df['TQQQ_Prev'] = df['TQQQ'].shift(1)
    return df

def run_engine(df, start_dt, params, is_backtest=False):
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
        
        # 티어 판별
        tier = 'MID'
        if mkt_eval > params['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > params['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < params['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < params['low_cut']/100: tier = 'LOW'

        action, trade_val = "관망", 0
        s_r, b_r = params['sell_ratios'][tier]/100, params['buy_ratios'][tier]/100

        if is_first:
            trade_val = min(cap * (params['initial_entry_pct'] / 100), max_c_usage)
            shares = trade_val / price; cash -= trade_val; action, is_first = "초기진입", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0: # 수익분 발생 -> 매도
                trade_val = diff * s_r
                qty = min(trade_val / price, shares)
                shares -= qty; cash += (qty * price); action = "분할매도"; trade_val = qty * price
            elif diff < 0: # 손실분 발생 -> 매수
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_val = min(cash, abs(diff) * b_r, avail)
                    qty = trade_val / price
                    shares += qty; cash -= trade_val; action = "분할매수"

        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({'날짜': date.strftime('%Y-%m-%d'), '티어': tier, '타입': action, '가격': round(price, 2), '거래금액': round(trade_val, 0), '수량': round(shares, 2), '잔액': round(cash, 0), '총자산': round(current_asset, 0)})
    
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 (저장 버튼 클릭 시 파일 영구 저장)
# -----------------------------------------------------------
full_df = fetch_data()

st.sidebar.markdown('<div class="program-title">Wedaeri Quantum T-Flow</div>', unsafe_allow_html=True)
with st.sidebar.form("settings_form"):
    s_date = st.date_input("투자 시작일", value=pd.to_datetime(st.session_state.settings['start_date']))
    s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
    s_max_c = st.slider("현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
    s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
    save_btn = st.form_submit_button("💾 설정 저장 및 동기화", type="primary")

if save_btn:
    new_settings = st.session_state.settings.copy()
    new_settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p})
    st.session_state.settings = new_settings
    save_settings(new_settings)
    st.rerun()

# -----------------------------------------------------------
# 4. 메인 화면
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 실전 대시보드", "📊 백테스트 분석"])

with tab1:
    # 실전 대시보드는 오늘(최신 데이터)까지 가동
    res_df, res_logs = run_engine(full_df, st.session_state.settings['start_date'], st.session_state.settings)
    
    today_now = datetime.now().strftime('%Y-%m-%d %H:%M')
    last_mkt = full_df.iloc[-1]
    
    st.markdown(f"""
        <div class="status-bar">
            <b>🕒 데이터 갱신:</b> {today_now} | <b>📉 TQQQ 현재가:</b> ${last_mkt['TQQQ']:.2f} (기준일: {full_df.index[-1].strftime('%Y-%m-%d')})
        </div>
    """, unsafe_allow_html=True)

    if not res_logs:
        st.warning("설정된 시작일 이후의 데이터가 없습니다.")
    else:
        last = res_logs[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재 보유 수량", f"{last['수량']:,.2f} 주")
        c2.metric("가용 현금 잔액", f"${last['잔액']:,.0f}")
        c3.metric("총 자산 평가액", f"${last['총자산']:,.0f}")
        c4.metric("누적 수익률", f"{(last['총자산']/st.session_state.settings['initial_capital']-1)*100:.1f}%")

        st.divider()
        
        # [요청 반영] 당일 매매 예상 주문표
        st.subheader("🎯 오늘 매매 전략 (예상가 입력)")
        col_in, col_out = st.columns([1, 1.5])
        with col_in:
            est_p = st.number_input("실시간/예상 종가 ($)", value=float(last_mkt['TQQQ']), step=0.01)
        with col_out:
            diff_p = est_p - last_mkt['TQQQ']
            order_msg = "시세 변동 대기 중..."
            if diff_p > 0:
                val = (last['수량'] * diff_p) * (st.session_state.settings['sell_ratios'][last['티어']]/100)
                order_msg = f"SELL: 약 ${val:,.0f} 매도 주문 ({val/est_p:.2f} 주)"
            elif diff_p < 0:
                max_u = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
                avail = max_u - (st.session_state.settings['initial_capital'] - last['잔액'])
                val = min(last['잔액'], abs(last['수량'] * diff_p) * (st.session_state.settings['buy_ratios'][last['티어']]/100), max(0, avail))
                order_msg = f"BUY: 약 ${val:,.0f} 매수 주문 ({val/est_p:.2f} 주)"
            st.markdown(f'<div class="order-box">{order_msg}</div>', unsafe_allow_html=True)

        with st.expander("🧾 실전 상세 매매 로그 (최근순)"):
            st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)

with tab2:
    st.subheader("📊 백테스트 정밀 분석 리포트")
    with st.form("bt_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("시뮬레이션 원금 ($)", value=10000)
        bt_start = bc2.date_input("분석 시작일", value=date(2010, 2, 12))
        bt_end = bc3.date_input("분석 종료일", value=date.today())
        run_bt = st.form_submit_button("🚀 백테스트 실행")

    if run_bt:
        bt_params = st.session_state.settings.copy()
        bt_params['initial_capital'] = bt_cap
        b_df, b_logs = run_engine(full_df[full_df.index <= pd.to_datetime(bt_end)], bt_start.strftime('%Y-%m-%d'), bt_params)
        
        if not b_df.empty:
            # 지표 계산
            final_v = b_df.iloc[-1]['Asset']
            total_ret = (final_v / bt_cap - 1) * 100
            days = (b_df.iloc[-1]['Date'] - b_df.iloc[0]['Date']).days
            cagr = ((final_v / bt_cap) ** (365 / max(1, days)) - 1) * 100
            b_df['Peak'] = b_df['Asset'].cummax()
            b_df['DD'] = (b_df['Asset'] / b_df['Peak'] - 1) * 100
            mdd = b_df['DD'].min()
            calmar = cagr / abs(mdd) if mdd != 0 else 0
            
            # 지표 출력
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("최종 수익률", f"{total_ret:.1f}%")
            m2.metric("CAGR", f"{cagr:.1f}%")
            m3.metric("MDD", f"{mdd:.1f}%")
            m4.metric("칼마 지수", f"{calmar:.2f}")
            m5.metric("최종 자산", f"${final_v:,.0f}")

            # [요청 반영] 수익률 + MDD 통합 그래프
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', lw=2, label='Asset (Log)')
            ax1.set_yscale('log'); ax1.set_title("자산 성장 및 하락 분석"); ax1.grid(True, alpha=0.2); ax1.legend()
            ax2.fill_between(b_df['Date'], b_df['DD'], 0, color='#E53935', alpha=0.3, label='MDD')
            ax2.set_ylabel("Drawdown (%)"); ax2.set_ylim(-100, 5); ax2.grid(True, alpha=0.2); ax2.legend()
            st.pyplot(fig)

            # [요청 반영] 년도별 성과표
            b_df['Year'] = b_df['Date'].dt.year
            y_data = []
            for y, g in b_df.groupby('Year'):
                y_ret = (g.iloc[-1]['Asset'] / g.iloc[0]['Asset'] - 1) * 100
                y_mdd = (g['Asset'] / g['Asset'].cummax() - 1).min() * 100
                y_data.append({'연도': y, '수익률': f"{y_ret:.1f}%", 'MDD': f"{y_mdd:.1f}%", '기말자산': f"${g.iloc[-1]['Asset']:,.0f}"})
            st.subheader("📅 연도별 성과 리포트")
            st.table(pd.DataFrame(y_data).set_index('연도'))

            st.subheader("📜 백테스트 상세 매매 로그")
            st.dataframe(pd.DataFrame(b_logs).sort_values('날짜', ascending=False), use_container_width=True)
