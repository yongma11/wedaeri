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

# [사용자 개인 설정]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE = '2025-12-26'  # 매매 시작일 정정
INITIAL_CAP = 108000       # 시작 원금 ($108,000)
INITIAL_CASH_RATIO = 0.40  # 초기 현금 비중 (40%)

def calculate_longterm_growth(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    for i in range(window, len(series)):
        try:
            fit = np.polyfit(date_nums[window:i+1], np.log(values[window:i+1]), 1)
            pred_log = fit[1] + fit[0] * date_nums[i]
            results[i] = np.exp(pred_log)
        except: pass
    return pd.Series(results, index=series.index)

def send_telegram(text):
    bot_token = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
    chat_id = "1442265681"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

def main():
    print("🚀 [위대리 v3.0] 3단계 로직 적용 오토봇 가동...")
    
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw: return

    try:
        # --- A. 인증 및 시트 연결 ---
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")

        # --- B. 데이터 수집 및 현재가 확정 ---
        end_dt = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", end=end_dt, auto_adjust=True, progress=False)
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)

        t_ticker = yf.Ticker("TQQQ")
        live_p = float(t_ticker.fast_info['last_price'])
        live_d = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        if live_d > pd.to_datetime(df['Date'].iloc[-1]):
            new_row = pd.DataFrame({'Date': [live_d], 'TQQQ': [live_p], 'QQQ': [yf.Ticker("QQQ").fast_info['last_price']]})
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[df.index[-1], 'TQQQ'] = live_p
        
        cur_p = round(float(df['TQQQ'].iloc[-1]), 2)

        # --- C. 지표 계산 ---
        df['Growth'] = calculate_longterm_growth(df['QQQ'], df['Date'])
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        
        weekly = df[df['Date'].dt.weekday == 4].copy()
        sim_data = weekly[weekly['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)

        # --- D. 정정된 3단계 잔고 시뮬레이션 ---
        # 매매 설정 (정정된 파라미터)
        settings = {
            'high_cut': 0.055, 'low_cut': -0.07,
            'sell_ratios': {'HIGH': 1.5, 'MID': 0.6, 'LOW': 0.33},
            'buy_ratios': {'HIGH': 0.5, 'MID': 0.6, 'LOW': 2.0}
        }

        # 초기 진입 (25년 12월 26일 기준)
        first_row = sim_data.iloc[0]
        cash = INITIAL_CAP * INITIAL_CASH_RATIO
        shares = int((INITIAL_CAP * (1 - INITIAL_CASH_RATIO)) / first_row['TQQQ'])
        avg_price = first_row['TQQQ']
        
        # 매매 복기
        for i in range(1, len(sim_data)-1):
            p = sim_data.loc[i, 'TQQQ']
            prev_p = sim_data.loc[i-1, 'TQQQ']
            m_eval = sim_data.loc[i, 'Eval']

            # 3단계 티어 판정
            tier = 'MID'
            if m_eval >= settings['high_cut']: tier = 'HIGH'
            elif m_eval <= settings['low_cut']: tier = 'LOW'

            price_diff = (shares * p) - (shares * prev_p)
            if price_diff > 0 and shares > 0: # 매도
                q_sell = int(min(round((price_diff * settings['sell_ratios'][tier]) / p), shares))
                shares -= q_sell
                cash += (q_sell * p)
            elif price_diff < 0: # 매수
                avail = INITIAL_CAP - (INITIAL_CAP - cash) # 가용 범위
                q_buy = int(min(cash, abs(price_diff) * settings['buy_ratios'][tier]) / p)
                shares += q_buy
                cash -= (q_buy * p)

        # --- E. 이번 주 주문량 계산 ---
        last_row = sim_data.iloc[-1]
        prev_row = sim_data.iloc[-2]
        last_eval = last_row['Eval']
        
        this_tier = 'MID'
        if last_eval >= settings['high_cut']: this_tier = 'HIGH'
        elif last_eval <= settings['low_cut']: this_tier = 'LOW'

        cur_diff = (shares * last_row['TQQQ']) - (shares * prev_row['TQQQ'])
        action, order_qty = "관망", 0
        
        if cur_diff > 0:
            action = "매도"
            order_qty = int(min(round((cur_diff * settings['sell_ratios'][this_tier]) / last_row['TQQQ']), shares))
        elif cur_diff < 0:
            action = "매수"
            order_qty = int(min(cash, abs(cur_diff) * settings['buy_ratios'][this_tier]) / last_row['TQQQ'])

        if order_qty == 0: action = "관망"

        # --- F. 업데이트 및 전송 ---
        ws.update_acell('L4', action)
        ws.update_acell('M4', 'LOC' if action != "관망" else "-")
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', order_qty)

        total_asset = cash + (shares * cur_p)
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        msg = f"🚀 *[위대리 v3.0] 3단계 매매 리포트*\n\n" \
              f"💰 *자산 현황*\n" \
              f"- 총 자산: `${total_asset:,.2f}`\n" \
              f"- 보유 주식: *{shares}* 주\n" \
              f"- 보유 현금: `${cash:,.2f}`\n\n" \
              f"🌡️ *시장 평가*: `{last_eval:.2%}` ({this_tier})\n" \
              f"🎯 *이번 주 주문*: *{action} {order_qty}주 (LOC)*\n\n" \
              f"📅 시작일: `2025-12-26` (원금 $108,000)"
        
        send_telegram(msg)
        print(f"✅ 완료: {action} {order_qty}주")

    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()
