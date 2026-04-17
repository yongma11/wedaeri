import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="위대리 Quantum T-Flow v3.0", layout="wide")

# 2. 데이터 엔진 (캐싱 적용)
@st.cache_data(ttl=3600)
def load_market_data():
    try:
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)
        if df_raw.empty: return pd.DataFrame()
        
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)
        
        # 성장성 및 Eval 계산 (25년 로그 회귀)
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

# 3. 백테스트 시뮬레이션 핵심 함수
def run_simulation(data, start_dt, end_dt, init_cap, cash_ratio, settings):
    sim_df = data[(data['Date'] >= pd.to_datetime(start_dt)) & (data['Date'] <= pd.to_datetime(end_dt))].copy()
    weekly = sim_df[sim_df['Date'].dt.weekday == 4].reset_index(drop=True)
    
    if weekly.empty: return pd.DataFrame(), 0, 0, 0
    
    cash = init_cap * cash_ratio
    shares = int((init_cap * (1 - cash_ratio)) / weekly['TQQQ'].iloc[0])
    history = []
    
    for i in range(len(weekly)):
        p, m_eval = weekly.loc[i, 'TQQQ'], weekly.loc[i, 'Eval']
        tier = 'MID'
        if m_eval >= settings['high_cut']: tier = 'HIGH'
        elif m_eval <= settings['low_cut']: tier = 'LOW'
        
        s_ratio = settings['sell_ratios'][tier]
        b_ratio = settings['buy_ratios'][tier]
        
        if i > 0:
            diff = (shares * p) - (shares * weekly.loc[i-1, 'TQQQ'])
            if diff > 0: # 매도
                q = int(min(round((diff * s_ratio) / p), shares))
                shares -= q; cash += q * p
            elif diff < 0: # 매수
                q = int(min(cash, abs(diff) * b_ratio) / p)
                shares += q; cash -= q * p
        history.append(cash + (shares * p))
    
    weekly['Total_Asset'] = history
    return weekly, cash, shares, weekly['Total_Asset'].iloc[-1]

# --- 메인 앱 시작 ---
df = load_market_data()

if df.empty:
    st.error("데이터를 불러올 수 없습니다. 야후 파이낸스 연결을 확인하세요.")
else:
    # 4. 사이드바 메뉴 구성
    with st.sidebar:
        st.title("🗂️ 메뉴 선택")
        menu = st.radio("이동할 페이지", ["📈 실전 대시보드", "🧪 자유 백테스트"])
        
        st.divider()
        st.subheader("⚙️ 기본 설정")
        global_start = st.date_input("기본 시작일", datetime.strptime("2025-12-26", "%Y-%m-%d"))
        global_cap = st.number_input("기본 자본 ($)", value=108000)
        
        # 공통 파라미터
        st.divider()
        st.subheader("⚖️ 전략 파라미터")
        high_c = st.slider("고평가(Eval >)", 0.0, 0.15, 0.055)
        low_c = st.slider("저평가(Eval <)", -0.15, 0.0, -0.07)
        
        st_params = {
            'high_cut': high_c, 'low_cut': low_c,
            'sell_ratios': {'HIGH': 1.5, 'MID': 0.6, 'LOW': 0.33},
            'buy_ratios': {'HIGH': 0.5, 'MID': 0.6, 'LOW': 2.0}
        }

    # --- 페이지 1: 실전 대시보드 ---
    if menu == "📈 실전 대시보드":
        st.title("📈 위대리 Quantum T-Flow 실전 상황")
        
        results_df, final_cash, final_shares, final_asset = run_simulation(
            df, global_start, datetime.now(), global_cap, 0.40, st_params
        )
        
        if not results_df.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("현재 총 자산", f"${final_asset:,.2f}")
            m2.metric("누적 수익률", f"{(final_asset/global_cap-1)*100:.2f}%")
            m3.metric("현재 Eval", f"{df['Eval'].iloc[-1]:.2%}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Total_Asset'], name="자산", line=dict(color='#00FFCC')))
            fig.update_layout(title="현재 전략 자산 성장 추이", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(results_df.tail(10), use_container_width=True)

    # --- 페이지 2: 자유 백테스트 ---
    elif menu == "🧪 자유 백테스트":
        st.title("🧪 특정 기간 자유 백테스트")
        st.info("원하는 과거 기간을 설정하여 위대리 전략의 성과를 확인하세요.")
        
        c1, c2 = st.columns(2)
        with c1:
            test_start = st.date_input("테스트 시작일", datetime.strptime("2020-01-01", "%Y-%m-%d"))
        with c2:
            test_end = st.date_input("테스트 종료일", datetime.now())
            
        test_cap = st.number_input("테스트 자본 ($)", value=100000)
        test_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, 40) / 100
        
        if st.button("백테스트 실행"):
            res_df, f_cash, f_shares, f_asset = run_simulation(
                df, test_start, test_end, test_cap, test_cash_ratio, st_params
            )
            
            if not res_df.empty:
                st.divider()
                st.subheader("📊 백테스트 결과 리포트")
                
                # 성과 지표 계산
                total_ret = (f_asset / test_cap - 1) * 100
                mdd = ((res_df['Total_Asset'].cummax() - res_df['Total_Asset']) / res_df['Total_Asset'].cummax()).max() * 100
                
                b1, b2, b3 = st.columns(3)
                b1.metric("최종 자산", f"${f_asset:,.2f}")
                b2.metric("기간 수익률", f"{total_ret:.2f}%")
                b3.metric("최대 낙폭(MDD)", f"-{mdd:.2f}%")
                
                # 차트
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Total_Asset'], name="백테스트 자산", line=dict(color='#FFA500')))
                fig_test.update_layout(title=f"{test_start} ~ {test_end} 성과 추이", template="plotly_dark")
                st.plotly_chart(fig_test, use_container_width=True)
                
                st.write("📂 상세 데이터 (하단)")
                st.dataframe(res_df, use_container_width=True)
            else:
                st.warning("해당 기간에 데이터가 충분하지 않습니다.")
