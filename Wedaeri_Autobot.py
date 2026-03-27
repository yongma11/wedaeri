import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import requests
from datetime import datetime

# ==========================================
# [설정 구간] 용성님, 여기만 수정하세요!
# ==========================================
JSON_KEY_FILE = 'wedaeri-key.json'  # 발급받은 JSON 파일명 (같은 폴더에 있어야 함)
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI' # 구글 시트 주소의 ID
START_DATE = '2026-02-02'          # 위대리 매매 시작일 (잔고 복구용)
INITIAL_CAP = 100000               # 투자 원금 ($)
# ==========================================

# 1. 위대리 장기 추세선 엔진
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

# 2. 시장 데이터 수집 및 실시간가 반영
def get_market_data():
    try:
        df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
        
        # 장마감 15분 전 실시간 가격 (예상종가)
        tqqq_ticker = yf.Ticker("TQQQ")
        qqq_ticker = yf.Ticker("QQQ")
        live_tqqq = tqqq_ticker.fast_info['last_price']
        live_qqq = qqq_ticker.fast_info['last_price']
        
        df.iloc[-1, df.columns.get_loc('TQQQ')] = live_tqqq
        df.iloc[-1, df.columns.get_loc('QQQ')] = live_qqq

        df_reset = df.reset_index()
        df_reset['Growth'] = calculate_longterm_growth(df_reset['QQQ'], df_reset['Date'])
        df_reset['Eval'] = (df_reset['QQQ'] / df_reset['Growth']) - 1
        
        weekly_df = df_reset[df_reset['Date'].dt.weekday == 4].copy()
        weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
        
        return weekly_df.dropna(subset=['Growth', 'TQQQ_Prev']).reset_index(drop=True)
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None

# 3. 위대리 로직 기반 현재 상태 복원
def get_current_state(df, settings):
    df_sim = df.iloc[:-1].copy() # 오늘 데이터 제외하고 과거만 시뮬레이션
    sim_data = df_sim[df_sim['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)
    
    cash = INITIAL_CAP
    shares = 0
    is_first = True
    max_c = INITIAL_CAP * (settings['max_cash_pct'] / 100)

    for i, row in sim_data.iterrows():
        price, prev_p, mkt_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
        tier = 'MID'
        if mkt_eval > settings['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > settings['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < settings['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < settings['low_cut']/100: tier = 'LOW'

        if is_first:
            qty = round(min(INITIAL_CAP * (settings['initial_entry_pct']/100), max_c) / price)
            shares = qty; cash -= (qty * price); is_first = False
        else:
            diff = (shares * price) - (shares * prev_p)
            if diff > 0:
                qty = int(min(round((diff * (settings['sell_ratios'][tier]/100)) / price), shares))
                shares -= qty; cash += (qty * price)
            elif diff < 0:
                avail = max_c - (INITIAL_CAP - cash)
                if avail > 0:
                    qty = round(min(cash, abs(diff) * (settings['buy_ratios'][tier]/100), avail) / price)
                    if (qty * price) > cash: qty = int(cash // price)
                    shares += qty; cash -= (qty * price)
    return cash, shares

# 4. 텔레그램 알림
def send_telegram(text):
    bot_token = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
    chat_id = "1442265681"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try: requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

# 5. 메인 실행 함수
def main():
    print("📈 [위대리 오토봇] 매매 시그널 판독 시작...")
    df_market = get_market_data()
    if df_market is None: return

    settings = {
        'max_cash_pct': 100, 'initial_entry_pct': 50,
        'uhigh_cut': 10, 'high_cut': 5, 'low_cut': -6, 'ulow_cut': -10,
        'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
        'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
    }
    
    cash_now, shares_now = get_current_state(df_market, settings)
    last_row = df_market.iloc[-1]
    cur_p = round(last_row['TQQQ'], 2)
    mkt_eval = last_row['Eval']
    
    tier = 'MID'
    if mkt_eval > 0.10: tier = 'UHIGH'
    elif mkt_eval > 0.05: tier = 'HIGH'
    elif mkt_eval < -0.10: tier = 'ULOW'
    elif mkt_eval < -0.06: tier = 'LOW'
    
    diff = (shares_now * cur_p) - (shares_now * last_row['TQQQ_Prev'])
    action, qty = "관망", 0
    
    if diff > 0:
        action = "매도"
        qty = int(min(round((diff * (settings['sell_ratios'][tier]/100)) / cur_p), shares_now))
    elif diff < 0:
        action = "매수"
        max_c = INITIAL_CAP * (settings['max_cash_pct']/100)
        avail = max_c - (INITIAL_CAP - cash_now)
        qty = round(min(cash_now, abs(diff) * (settings['buy_ratios'][tier]/100), avail) / cur_p)
        if (qty * cur_p) > cash_now: qty = int(cash_now // cur_p)

    # 구글 시트 업데이트
    try:
        gc = gspread.service_account(filename=JSON_KEY_FILE)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
        ws.update('L4:O4', [[action, "LOC", cur_p, qty if qty > 0 else 0]])
        print("📤 구글 시트 업데이트 완료!")
    except Exception as e:
        print(f"❌ 구글 시트 연동 실패: {e}")

    # 텔레그램 전송
    msg = f"🚀 *[위대리] 주간 매매 시그널*\n\n" \
          f"📊 시장모드: `{tier}`\n" \
          f"💰 자산: `${cash_now + (shares_now * cur_p):,.0f}`\n\n" \
          f"🎯 *주문 (LOC)*\n" \
          f"- 종류: *{action}*\n" \
          f"- 가격: `${cur_p}`\n" \
          f"- 수량: *{qty}* 주"
    send_telegram(msg)

if __name__ == "__main__":
    main()
