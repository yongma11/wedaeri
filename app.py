import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
import warnings
from datetime import datetime, timedelta

# 1. 환경 설정 및 경고 무시
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# [사용자 핵심 파라미터 설정]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE = '2025-12-26'  # 정정된 시작일
INITIAL_CAP = 108000       # 정정된 초기자본 ($108,000)
INITIAL_CASH_RATIO = 0.40  # 정정된 초기현금비중 (40%)

# 매매 핵심 파라미터 (3단계 분류)
SETTINGS = {
    'high_cut': 0.055,   # 고평가 기준 (5.5% 이상)
    'low_cut': -0.07,    # 저평가 기준 (-7.0% 이하)
    'sell_ratios': {
        'HIGH': 1.5,     # 고평가 시 매도율 150%
        'MID': 0.6,      # 중간평가 시 매수/매도율 60%
        'LOW': 0.33      # 저평가 시 매도율 33%
    },
    'buy_ratios': {
        'HIGH': 0.5,     # 고평가 시 매수율 50%
        'MID': 0.6,      # 중간평가 시 매수/매도율 60%
        'LOW': 2.0       # 저평가 시 매수율 200%
    }
}

# 2. 장기 로그 회귀 추세선 계산 (핵심 엔진)
def calculate_longterm_growth(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        try:
            # 과거 5년(1260일) 데이터를 기반으로 로그 회귀 분석
            fit = np.polyfit(date_nums[i-window:i], np.log(values[i-window:i]), 1)
            pred_log = fit[1] + fit[0] * date_nums[i]
            results[i] = np.exp(pred_log)
        except: pass
    return pd.Series(results, index=series.index)

# 3. 텔레그램 전송 함수
def send_telegram(text):
    bot_token = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
    chat_id = "1442265681"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

# 4. 메인 실행 루틴
def main():
    print(f"🚀 [위대리 v3.0] {START_DATE} 기준 3단계 로직 가동...")
    
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw: return

    try:
        # --- A. 인증 및 구글 시트 연결 ---
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
        print("✅ 구글 시트 연결 성공!")

        # --- B. 실시간 데이터 수집 및 보정 (시그마2 방식 이식) ---
        end_dt = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", end=end_dt, auto_adjust=True, progress=False)
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)

        # 실시간가 포착
        t_ticker = yf.Ticker("TQQQ")
        live_p = float(t_ticker.fast_info['last_price'])
        live_d = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        # 날짜 비교 후 행 추가 또는 업데이트
        if live_d > pd.to_datetime(df['Date'].iloc[-1]):
            q_price = yf.Ticker("QQQ").fast_info['last_price']
            new_row = pd.DataFrame({'Date': [live_d], 'TQQQ': [live_p], 'QQQ': [float(q_price)]})
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[df.index[-1], 'TQQQ'] = live_p
        
        cur_p = round(float(df['TQQQ'].iloc[-1]), 2)
        print(f"📊 현재가 확정: TQQQ ${cur_p}")

        # --- C. 위대리 지표 계산 ---
        df['Growth'] = calculate_longterm_growth(df['QQQ'], df['Date'])
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        
        # 금요일 데이터만 추출하여 매매 시뮬레이션 준비
        weekly = df[df['Date'].dt.weekday == 4].copy()
        sim_data = weekly[weekly['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)

        # --- D. 잔고 및 수량 시뮬레이션 (복기 로직) ---
        # 초기 설정
        first_row = sim_data.iloc[0]
        cash = INITIAL_CAP * INITIAL_CASH_RATIO
        shares = int((INITIAL_CAP * (1 - INITIAL_CASH_RATIO)) / first_row['TQQQ'])
        avg_price = first_row['TQQQ']
        
        # 매주 금요일마다의 가격 변동에 따른 수량 복기 (마지막 주 직전까지)
        for i in range(1, len(sim_data)-1):
            p = sim_data.loc[i, 'TQQQ']
            prev_p = sim_data.loc[i-1, 'TQQQ']
            m_eval = sim_data.loc[i, 'Eval']

            # 3단계 티어 판정
            tier = 'MID'
            if m_eval >= SETTINGS['high_cut']: tier = 'HIGH'
            elif m_eval <= SETTINGS['low_cut']: tier = 'LOW'

            price_diff = (shares * p) - (shares * prev_p)
            
            if price_diff > 0 and shares > 0: # 수익 발생 시 매도
                q_sell = int(min(round((price_diff * SETTINGS['sell_ratios'][tier]) / p), shares))
                shares -= q_sell
                cash += (q_sell * p)
            elif price_diff < 0: # 하락 발생 시 매수
                q_buy = int(min(cash, abs(price_diff) * SETTINGS['buy_ratios'][tier]) / p)
                if q_buy > 0:
                    total_cost = (shares * avg_price) + (q_buy * p)
                    shares += q_buy
                    avg_price = total_cost / shares
                    cash -= (q_buy * p)

        # --- E. 금주 실전 주문량 계산 ---
        last_row = sim_data.iloc[-1]
        prev_row = sim_data.iloc[-2]
        last_eval = last_row['Eval']
        
        this_tier = 'MID'
        if last_eval >= SETTINGS['high_cut']: this_tier = 'HIGH'
        elif last_eval <= SETTINGS['low_cut']: this_tier = 'LOW'

        cur_diff = (shares * last_row['TQQQ']) - (shares * prev_row['TQQQ'])
        action, order_qty = "관망", 0
        
        if cur_diff > 0:
            action = "매도"
            order_qty = int(min(round((cur_diff * SETTINGS['sell_ratios'][this_tier]) / last_row['TQQQ']), shares))
        elif cur_diff < 0:
            action = "매수"
            order_qty = int(min(cash, abs(cur_diff) * SETTINGS['buy_ratios'][this_tier]) / last_row['TQQQ'])
        
        if order_qty == 0: action = "관망"

        # --- F. 결과 업데이트 및 리포트 전송 ---
        ws.update_acell('L4', action)
        ws.update_acell('M4', 'LOC' if action != "관망" else "-")
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', order_qty)

        total_asset = cash + (shares * cur_p)
        profit_rate = (total_asset / INITIAL_CAP - 1) * 100
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        msg = f"🚀 *[위대리 v3.0] 3단계 매매 리포트*\n\n" \
              f"💰 *자산 현황*\n" \
              f"- 총 자산: `${total_asset:,.2f}`\n" \
              f"- 누적 수익률: *{profit_rate:.2f}%*\n" \
              f"- 보유 현금: `${cash:,.2f}`\n\n" \
              f"📈 *보유 주식 정보*\n" \
              f"- 종목: *TQQQ*\n" \
              f"- 수량: *{shares}* 주\n" \
              f"- 평단가: `${avg_price:.2f}`\n" \
              f"- 현재가: `${cur_p}`\n\n" \
              f"🌡️ *시장 평가*: `{last_eval:.2%}` ({this_tier})\n" \
              f"🎯 *이번 주 주문*: *{action} {order_qty}주 (LOC)*\n\n" \
              f"💡 설정: 고평가 {SETTINGS['high_cut']:.1%}, 저평가 {SETTINGS['low_cut']:.1%}"
        
        send_telegram(msg)
        print(f"✅ 리포트 전송 완료: {action} {order_qty}주")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
