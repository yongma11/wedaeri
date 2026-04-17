import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="위대리 Quantum T-Flow v3.1", layout="wide")

# [사용자 설정]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'

# 2. 데이터 엔진 (보안 및 에러 방어 강화)
@st.cache_data(ttl=3600)
def load_market_data():
    try:
        tickers = ["QQQ", "TQQQ"]
        df_raw = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False, timeout=20)
        if df_raw.empty: return pd.DataFrame()
        
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)
        
        # 25년 로그 회귀 엔진
        results = [np.nan] * len(df)
        date_nums = df['Date'].map(pd.Timestamp.toordinal).values
        values = df['QQQ'].values
        for i in range(1260, len(df)):
            fit = np.polyfit(date_nums[i-1260:i], np.log(values[i-1260:i]), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        
        df['Growth'] = results
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        return df
    except: return pd.DataFrame()

# 3. 시뮬레이션 엔진 (상세 로그 생성 포함)
def run_enhanced_sim(data, start_dt, end_dt, init_cap, cash_ratio, settings):
    sim_df = data[(data['Date'] >= pd.to_datetime(start_dt)) & (data['Date'] <= pd.to_datetime(end_dt))].copy()
    weekly = sim_df[sim_df['Date'].dt.weekday == 4].reset_index(drop=True)
    if weekly.empty: return pd.DataFrame(), 0, 0, 0, 0
    
    cash = init_cap * cash_ratio
    shares = int((init_cap * (1 - cash_ratio)) / weekly['TQQQ'].iloc[0])
    logs = []

    for i in range(len(weekly)):
        p, m_eval = weekly.loc[i, 'TQQQ'], weekly.loc[i, 'Eval']
        tier = 'MID'
        if m_eval >= settings['high_cut']: tier = 'HIGH'
        elif m_eval <= settings['low_cut']: tier = 'LOW'
        
        action, order_qty, trade_amt = "유지", 0, 0
        
        if i > 0:
            prev_p = weekly.loc[i-1, 'TQQQ']
            diff = (shares * p) - (shares * prev_p)
            if diff > 0: # 매도
                order_qty = int(min(round((diff * settings['sell_ratios'][tier]) / p), shares))
                action = "매도" if order_qty > 0 else "유지"
                shares -= order_qty; cash += order_qty * p
                trade_amt = order_qty * p
            elif diff < 0: # 매수
                order_qty = int(min(cash, abs(diff) * settings['buy_ratios'][tier]) / p)
                action = "매수" if order_qty > 0 else "유지"
                shares += order_qty; cash -= order_qty * p
                trade_amt = order_qty * p
        
        total_asset = cash + (shares * p)
        # MDD 계산을 위한 누적 최고치 기록
        running_max = max([l['총자산'] for l in logs] + [total_asset])
        current_mdd = (running_max - total_asset) / running_max * 100

        logs.append({
            '날짜': weekly.loc[i, 'Date'].strftime('%Y-%m-%d'),
            '시장평가': f"{m_eval:.2%}",
            '액션': action,
            '주문수량': order_qty,
            '매매대금': f"${trade_amt:,.0f}",
            '보유수량': shares,
            '평가금': f"${(shares*p):,.0f}",
            '현금': f"${cash:,.0f}",
            '총자산': total_asset,
            '현금비중': f"{(cash/total_asset)*100:.1f}%",
            'MDD': f"{current_mdd:.2f}%"
        })
    
    log_df = pd.DataFrame(logs)
    return log_df, cash, shares, total_asset, current_mdd

# --- 메인 앱 시작 ---
df = load_market_data()

if df.empty:
    st.error("데이터 로드 실패. 야후 파이낸스 상태를 확인하세요.")
else:
    with st.sidebar:
        st.title("🗂️ 메뉴")
        menu = st.radio("선택", ["📈 실전 대시보드", "🧪 자유 백테스트"])
        st.divider()
        st_start = st.date_input("실전 시작일", datetime.strptime("2025-12-26", "%Y-%m-%d"))
        st_cap = st.number_input("실전 원금 ($)", value=108000)
        
        st.subheader("⚙️ 전략 파라미터")
        h_c = st.slider("고평가(Eval >)", 0.0, 0.15, 0.055)
        l_c = st.slider("저평가(Eval <)", -0.15, 0.0, -0.07)
        params = {
            'high_cut': h_c, 'low_cut': l_c,
            'sell_ratios': {'HIGH': 1.5, 'MID': 0.6, 'LOW': 0.33},
            'buy_ratios': {'HIGH': 0.5, 'MID': 0.6, 'LOW': 2.0}
        }

    if menu == "📈 실전 대시보드":
        st.title("📈 실전 자산 현황 & 매매로그")
        log_df, f_cash, f_shares, f_asset, f_mdd = run_enhanced_sim(df, st_start, datetime.now(), st_cap, 0.40, params)
        
        if not log_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재 총 자산", f"${f_asset:,.2f}")
            c2.metric("누적 수익률", f"{(f_asset/st_cap-1)*100:.2f}%")
            c3.metric("현재 보유 수량", f"{f_shares} 주")
            c4.metric("현재 MDD", f"-{f_mdd:.2f}%")

            st.divider()
            st.subheader("📜 상세 매매 로그 (최근순)")
            # 총자산 포맷팅 및 로그 출력
            display_log = log_df.copy()
            display_log['총자산'] = display_log['총자산'].map(lambda x: f"${x:,.2f}")
            st.dataframe(display_log.iloc[::-1], use_container_width=True)

    elif menu == "🧪 자유 백테스트":
        st.title("🧪 백테스트 리포트")
        t_start = st.date_input("테스트 시작일", datetime.strptime("2010-01-01", "%Y-%m-%d"))
        t_end = st.date_input("테스트 종료일", datetime.now())
        t_cap = st.number_input("테스트 자본 ($)", value=100000)
        
        if st.button("백테스트 실행"):
            log_df, f_cash, f_shares, f_asset, f_mdd = run_enhanced_sim(df, t_start, t_end, t_cap, 0.40, params)
            
            if not log_df.empty:
                # 1. 연도별 성과 계산
                log_df['Year'] = pd.to_datetime(log_df['날짜']).dt.year
                yearly_stats = []
                for year, group in log_df.groupby('Year'):
                    y_start_asset = group['총자산'].iloc[0]
                    y_end_asset = group['총자산'].iloc[-1]
                    y_profit = (y_end_asset / y_start_asset - 1) * 100
                    # 연도 내 MDD
                    y_max = group['총자산'].cummax()
                    y_mdd = ((y_max - group['총자산']) / y_max).max() * 100
                    yearly_stats.append({'연도': year, '수익률': f"{y_profit:.2f}%", 'MDD': f"-{y_mdd:.2f}%"})
                
                st.subheader("📅 연도별 성과 요약 (CAGR/MDD)")
                st.table(pd.DataFrame(yearly_stats))
                
                # 2. 전체 통계
                total_years = (pd.to_datetime(t_end) - pd.to_datetime(t_start)).days / 365.25
                cagr = ((f_asset / t_cap) ** (1/total_years) - 1) * 100
                
                b1, b2, b3 = st.columns(3)
                b1.metric("최종 자산", f"${f_asset:,.2f}")
                b2.metric("CAGR (연평균 수익률)", f"{cagr:.2f}%")
                b3.metric("최대 낙폭 (MDD)", f"-{f_mdd:.2f}%")

                st.divider()
                st.subheader("📜 전체 매매 상세 로그")
                display_log = log_df.drop(columns=['Year']).copy()
                display_log['총자산'] = display_log['총자산'].map(lambda x: f"${x:,.2f}")
                st.dataframe(display_log, use_container_width=True)
