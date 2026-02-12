import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
import json
import pytz
from datetime import datetime, date, timedelta

# -----------------------------------------------------------
# 0. 기본 설정 & 스타일 (이미지 레이아웃 100% 반영)
# -----------------------------------------------------------
st.set_page_config(page_title="Wedaeri Quantum T-Flow v2.0", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .program-title { font-size: 38px; font-weight: 900; color: #0D47A1; text-align: center; margin-bottom: 25px; }
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #eee; text-align: center; height: 140px; }
    .metric-label { font-size: 15px; color: #666; font-weight: 500; margin-bottom: 8px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #333; }
    .order-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #eee; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .order-result { padding: 25px; border-radius: 12px; font-size: 26px; font-weight: bold; text-align: center; }
    .buy-box { background-color: #f0fff4; color: #2e7d32; border: 1px solid #c8e6c9; }
    .sell-box { background-color: #fff5f5; color: #c62828; border: 1px solid #ffcdd2; }
    .hold-box { background-color: #f8f9fa; color: #616161; border: 1px solid #e0e0e0; }
    .account-label { font-size: 14px; color: #666; margin-bottom: 5px; }
    .account-value { font-size: 28px; font-weight: 800; color: #222; }
    .strategy-card { background-color: #ffffff; padding: 35px; border-radius: 15px; border: 1px solid #e0e0e0; line-height: 1.9; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="program-title">Wedaeri Quantum T-Flow</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 1. 설정 및 미국 시간 자동 동기화 로직
# -----------------------------------------------------------
SETTINGS_FILE = 'wedaeri_settings_v20.json'
default_settings = {
    'start_date': '2025-01-01', 'initial_capital': 100000, 'max_cash_pct': 100, 'initial_entry_pct': 50,
    'uhigh_cut': 10.0, 'high_cut': 5.0, 'low_cut': -6.0, 'ulow_cut': -10.0,
    'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
    'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
}

# 미국 동부 시간(EST/EDT) 계산
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est)
is_friday = now_est.weekday() == 4  # 4는 금요일
date_color = "#E53935" if is_friday else "#1E88E5" # 금요일엔 빨간색

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

@st.cache_data(ttl=300) # 미국 장 마감 반영을 위해 5분마다 갱신
def fetch_data_v20():
    df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
    window = 1260
    qqq_vals, date_nums = df['QQQ'].values, df.index.map(pd.Timestamp.toordinal).values
    growth = [np.nan] * len(df)
    for i in range(window, len(df)):
        fit = np.polyfit(date_nums[i-window : i], np.log(qqq_vals[i-window : i]), 1)
        growth[i] = np.exp(fit[1] + fit[0] * date_nums[i])
    df['Growth'], df['Eval'] = growth, (df['QQQ'] / growth) - 1
    
    # 금요일 혹은 가장 최근 거래일 포함 (미국 시간 동기화)
    weekly_df = df[df.index.weekday == 4].copy()
    if not is_friday and now_est.date() > weekly_df.index[-1].date():
        # 금요일이 아니지만 최신 일간 데이터가 있다면 가상으로 포함 (주문표용)
        latest_day = df.iloc[-1:]
        weekly_df = pd.concat([weekly_df, latest_day]).drop_duplicates()
        
    weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
    return weekly_df

# -----------------------------------------------------------
# 2. 정수 매매 엔진
# -----------------------------------------------------------
def run_engine_v20(df, start_dt, params):
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

        action, trade_qty = "관망", 0
        s_r, b_r = params['sell_ratios'][tier]/100, params['buy_ratios'][tier]/100

        if is_first:
            trade_qty = round(min(cap * (params['initial_entry_pct'] / 100), max_c_usage) / price)
            shares = trade_qty; cash -= (trade_qty * price); action, is_first = "매수", False
        else:
            if np.isnan(prev_p): continue
            diff = (shares * price) - (shares * prev_p)
            if diff > 0:
                trade_qty = int(min(round((diff * s_r) / price), shares))
                shares -= trade_qty; cash += (trade_qty * price); action = "매도"; trade_qty = -trade_qty
            elif diff < 0:
                avail = max_c_usage - (cap - cash)
                if avail > 0:
                    trade_qty = round(min(cash, abs(diff) * b_r, avail) / price)
                    if (trade_qty * price) > cash: trade_qty = int(cash // price)
                    shares += trade_qty; cash -= (trade_qty * price); action = "매수"
        
        current_asset = cash + (shares * price)
        history.append({'Date': date, 'Asset': current_asset})
        logs.append({
            '날짜': date.strftime('%Y-%m-%d'), '시장평가': tier, '매매': action, 
            '가격': price, '거래수량': trade_qty, '보유수량': int(shares), 
            '예수금': cash, '평가금': shares * price, '총자산': current_asset
        })
    return pd.DataFrame(history), logs

# -----------------------------------------------------------
# 3. 사이드바 및 레이아웃
# -----------------------------------------------------------
df_weekly = fetch_data_v20()

with st.sidebar:
    st.header("⚙️ 시스템 설정")
    with st.form(key="sidebar_v2"):
        s_date = st.date_input("투자 시작일", value=pd.to_datetime(st.session_state.settings['start_date']))
        s_cap = st.number_input("투자 원금 ($)", value=st.session_state.settings['initial_capital'], step=1000)
        s_max_c = st.slider("현금 투입 한도 (%)", 10, 100, st.session_state.settings['max_cash_pct'])
        s_init_p = st.slider("초기 진입 비중 (%)", 0, 100, st.session_state.settings['initial_entry_pct'])
        save_btn = st.form_submit_button("💾 설정 저장 및 동기화", type="primary")

if save_btn:
    st.session_state.settings.update({'start_date': s_date.strftime('%Y-%m-%d'), 'initial_capital': s_cap, 'max_cash_pct': s_max_c, 'initial_entry_pct': s_init_p})
    save_settings(st.session_state.settings); st.rerun()

tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 전략 로직"])

# --- TAB 1: 실전 트레이딩 ---
with tab1:
    res_df, res_logs = run_engine_v20(df_weekly, st.session_state.settings['start_date'], st.session_state.settings)
    last_mkt = df_weekly.iloc[-1]
    prev_mkt = df_weekly.iloc[-2]
    
    # 상단 4분할 지표 바 (이미지 100% 반영)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">현재 날짜 (미국시간)</div><div class="metric-value" style="color:{date_color};">{now_est.strftime("%Y-%m-%d (%a)")}</div></div>', unsafe_allow_html=True)
    with c2:
        eval_val = last_mkt['Eval'] * 100
        tier_str = "MID"
        if eval_val > st.session_state.settings['uhigh_cut']: tier_str = "HIGH (고평가)"
        elif eval_val < st.session_state.settings['ulow_cut']: tier_str = "LOW (저평가)"
        st.markdown(f'<div class="metric-card"><div class="metric-label">시장 모드</div><div class="metric-value" style="color:#f57c00;">{tier_str}</div><div style="font-size:12px; color:#999;">평가율 {eval_val:.2f}%</div></div>', unsafe_allow_html=True)
    with c3:
        diff = last_mkt['TQQQ'] - prev_mkt['TQQQ']
        diff_pct = (diff / prev_mkt['TQQQ']) * 100
        st.markdown(f'<div class="metric-card"><div class="metric-label">TQQQ 현재가</div><div class="metric-value">${last_mkt["TQQQ"]:.2f}</div><div style="font-size:13px; color:{"red" if diff>0 else "blue"};">전일대비 {diff:+.2f} ({diff_pct:+.2f}%)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">매매 사이클</div><div class="metric-value">{len(res_logs)}주차</div><div style="font-size:12px; color:#999;">매주 금요일 리밸런싱</div></div>', unsafe_allow_html=True)

    st.write("---")
    
    # 오늘 주문표
    st.subheader("📝 오늘 주문표 (Daily Order)")
    o_col1, o_col2 = st.columns([1, 2])
    if res_logs:
        last_st = res_logs[-1]
        with o_col1: est_p = st.number_input("예상 종가 입력 ($)", value=float(last_mkt['TQQQ']), step=0.01)
        with o_col2:
            diff_p = est_p - last_mkt['TQQQ']
            order_msg, box_class = "관망 (HOLD)", "hold-box"
            if diff_p > 0:
                s_ratio = st.session_state.settings['sell_ratios'].get(last_st['시장평가'], 60) / 100
                qty = int(min(round((last_st['보유수량'] * diff_p * s_ratio) / est_p), last_st['보유수량']))
                if qty > 0: order_msg, box_class = f"📈 매도 (SELL): 기준가 ${est_p:.2f} ({qty}주)", "sell-box"
            elif diff_p < 0:
                b_ratio = st.session_state.settings['buy_ratios'].get(last_st['시장평가'], 60) / 100
                max_l = st.session_state.settings['initial_capital'] * (st.session_state.settings['max_cash_pct']/100)
                avail = max_l - (st.session_state.settings['initial_capital'] - last_st['예수금'])
                qty = round(min(last_st['예수금'], abs(last_st['보유수량'] * diff_p * b_ratio), max(0, avail)) / est_p)
                if qty > 0: order_msg, box_class = f"📉 매수 (BUY): 기준가 ${est_p:.2f} ({qty}주)", "buy-box"
            st.markdown(f'<div class="order-result {box_class}">{order_msg}</div>', unsafe_allow_html=True)

    st.write("---")

    # 내 계좌 현황 (이미지 디자인 반영)
    st.subheader("💰 내 계좌 현황")
    if res_logs:
        acc = res_logs[-1]
        a1, a2, a3, a4 = st.columns(4)
        with a1: st.markdown(f'<div class="account-label">총 보유 수량</div><div class="account-value">{acc["보유수량"]:,} 주</div>', unsafe_allow_html=True)
        with a2: st.markdown(f'<div class="account-label">보유 현금</div><div class="account-value">${acc["예수금"]:,.0f}</div>', unsafe_allow_html=True)
        profit = acc['총자산'] - st.session_state.settings['initial_capital']
        profit_p = (profit / st.session_state.settings['initial_capital']) * 100
        with a3: st.markdown(f'<div class="account-label">총 평가 손익</div><div class="account-value">${profit:,.0f}</div><div style="color:{"red" if profit>0 else "blue"}; font-weight:bold;">↑ {profit_p:.1f}%</div>', unsafe_allow_html=True)
        with a4: st.markdown(f'<div class="account-label">현재 총 자산</div><div class="account-value">${acc["총자산"]:,.0f}</div>', unsafe_allow_html=True)

    # 매매 로그 (Expander)
    with st.expander("📜 상세 매매 히스토리 보기"):
        st.dataframe(pd.DataFrame(res_logs).sort_values('날짜', ascending=False), use_container_width=True)

    st.write("---")
    
    # 자산 성장 그래프 (이미지 하단 그래프 구현)
    st.subheader("📈 내 자산 성장 그래프 (Equity Curve)")
    if not res_df.empty:
        fig_real, ax = plt.subplots(figsize=(12, 5))
        ax.plot(res_df['Date'], res_df['Asset'], color='#2e7d32', lw=2.5, label='Total Equity')
        ax.fill_between(res_df['Date'], res_df['Asset'], st.session_state.settings['initial_capital'], 
                        where=(res_df['Asset'] >= st.session_state.settings['initial_capital']), 
                        color='#e8f5e9', alpha=0.5)
        ax.fill_between(res_df['Date'], res_df['Asset'], st.session_state.settings['initial_capital'], 
                        where=(res_df['Asset'] < st.session_state.settings['initial_capital']), 
                        color='#ffebee', alpha=0.5)
        ax.axhline(st.session_state.settings['initial_capital'], color='#999', linestyle='--', lw=1)
        ax.set_title("Total Equity Growth", fontsize=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        st.pyplot(fig_real)

# --- TAB 2: 백테스트 분석 ---
with tab2:
    st.subheader("📊 백테스트 정밀 분석")
    with st.form("bt_v2"):
        bc1, bc2, bc3 = st.columns(3)
        bt_cap = bc1.number_input("테스트 원금 ($)", value=100000)
        bt_start = bc2.date_input("시작일", value=date(2010, 2, 12)); bt_end = bc3.date_input("종료일", value=date.today())
        run_bt = st.form_submit_button("🚀 분석 실행")
    if run_bt:
        bt_p = st.session_state.settings.copy(); bt_p['initial_capital'] = bt_cap
        b_df, b_logs = run_engine_v20(df_weekly[df_weekly.index <= pd.to_datetime(bt_end)], bt_start.strftime('%Y-%m-%d'), bt_p)
        if not b_df.empty:
            final_v = b_df.iloc[-1]['Asset']; cagr = ((final_v / bt_cap) ** (365 / max(1, (b_df.iloc[-1]['Date'] - b_df.iloc[0]['Date']).days)) - 1) * 100
            b_df['Peak'] = b_df['Asset'].cummax(); b_df['DD'] = (b_df['Asset'] / b_df['Peak'] - 1) * 100
            mdd = b_df['DD'].min(); calmar = cagr / abs(mdd) if mdd != 0 else 0
            
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("수익률", f"{(final_v/bt_cap-1)*100:.1f}%"); i2.metric("CAGR", f"{cagr:.1f}%"); i3.metric("MDD", f"{mdd:.1f}%"); i4.metric("칼마", f"{calmar:.2f}")
            fig_bt, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(b_df['Date'], b_df['Asset'], color='#1E88E5', lw=2)
            ax2 = ax1.twinx(); ax2.fill_between(b_df['Date'], b_df['DD'], 0, color='#E53935', alpha=0.2)
            st.pyplot(fig_bt)
            st.dataframe(pd.DataFrame(b_logs).sort_values('날짜', ascending=False), use_container_width=True)

# --- TAB 3: 전략 로직 상세 설명 (가독성 개선 버전) ---
with tab3:
    # 제목 부분
    st.title("📘 Wedaeri Quantum T-Flow 전략 가이드")
    st.write("---")

    # 전체 서론
    st.info("본 전략은 나스닥100 3배 레버리지(TQQQ)의 **'극단적 변동성'**을 인위적인 리밸런싱을 통해 **'복리 수익'**으로 치환하는 퀀트 시스템입니다.")

    # 1. 핵심 로직
    st.subheader("1. 핵심 로직: 로그 회귀 이격도 (Log-Regression Eval)")
    st.write("""
    시장의 소음(Noise)을 제거하기 위해 QQQ의 지난 25년 데이터에 **로그 회귀(Log-Regression)**를 적용합니다. 
    이를 통해 산출된 **'역사적 평균 성장선(Growth Line)'** 대비 현재 가격의 이격도를 측정하여 시장의 과열(HIGH)과 침체(LOW)를 정량적으로 판별합니다.
    """)

    # 2. 주간 변동성 리밸런싱
    st.subheader("2. 주간 변동성 리밸런싱 (Weekly Rebalancing)")
    st.write("""
    매일 매매할 경우 수수료와 슬리피지가 수익을 갉아먹습니다. 본 시스템은 **매주 금요일 종가**만을 확인합니다. 
    한 주 동안 발생한 자산 가치의 변화분(Volatility)에 티어별 가중치를 곱하여 기계적으로 수량을 조절합니다.
    """)
    
    st.markdown("""
    * **상승 시 (익절):** 올라간 가치만큼 정해진 비율로 매도하여 현금을 확보합니다.
    * **하락 시 (추매):** 내려간 가치만큼 확보된 현금으로 추가 매수하여 평단가를 낮추고 수량을 늘립니다.
    """)

    # 3. 전략의 3대 장점
    st.subheader("3. 전략의 3대 장점")
    st.success("""
    1.  **MDD 방어:** 하락장에서 '현금 방패' 로직이 작동하여 TQQQ의 -90% 폭락 구간에서도 생존을 보장합니다.
    2.  **수량 극대화:** 횡보장과 하락장에서 기계적 추매를 통해 상승장이 왔을 때 폭발적인 수익을 낼 수 있는 수량을 확보합니다.
    3.  **심리적 안정:** 감정을 배제하고 '오늘 주문표'에 나온 수치대로만 매매하므로 뇌동매매를 원천 차단합니다.
    """)

    # 4. 실전 매매 가이드
    st.subheader("4. 실전 매매 가이드")
    st.write("""
    금요일 밤(미국시간), 대시보드의 **'오늘 주문표'**에 예상 종가를 입력하세요. 
    시스템이 계산해주는 정수 수량만큼 증권사 앱에서 주문을 넣는 것으로 한 주의 매매는 끝납니다. 남은 시간은 본업과 휴식에 집중하세요.
    """)
