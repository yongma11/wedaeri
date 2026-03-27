import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 1. 위대리 핵심 엔진: 25년 장기 로그 회귀 추세선
# -----------------------------------------------------------
def calculate_longterm_growth(series, dates, window=1260):
    results = [np.nan] * len(series)
    date_nums = dates.map(pd.Timestamp.toordinal).values
    values = series.values
    # 위대리 v2.5 원칙: 전체 기간 데이터를 활용한 정밀 로그 회귀
    for i in range(window, len(series)):
        try:
            # 시작점부터 현재(i)까지의 데이터를 모두 사용하여 추세선 산출
            fit = np.polyfit(date_nums[window:i+1], np.log(values[window:i+1]), 1)
            pred_log = fit[1] + fit[0] * date_nums[i]
            results[i] = np.exp(pred_log)
        except: pass
    return pd.Series(results, index=series.index)

def get_market_data():
    try:
        # 데이터 호출 (2000년부터 호출하여 장기 추세선 신뢰도 확보)
        df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
        
        # [실시간가 반영] 장 마감 15분 전 현재가를 '오늘의 종가'로 간주
        tqqq_ticker = yf.Ticker("TQQQ")
        qqq_ticker = yf.Ticker("QQQ")
        live_tqqq = tqqq_ticker.fast_info['last_price']
        live_qqq = qqq_ticker.fast_info['last_price']
        
        # 마지막 행의 가격을 실시간 데이터로 교체 (또는 새로운 행으로 추가)
        df.iloc[-1, df.columns.get_loc('TQQQ')] = live_tqqq
        df.iloc[-1, df.columns.get_loc('QQQ')] = live_qqq

        df_reset = df.reset_index()
        df_reset['Growth'] = calculate_longterm_growth(df_reset['QQQ'], df_reset['Date'])
        df_reset['Eval'] = (df_reset['QQQ'] / df_reset['Growth']) - 1
        
        # 주간 데이터(금요일) 추출 및 전주 종가 생성
        weekly_df = df_reset[df_reset['Date'].dt.weekday == 4].copy()
        weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
        
        return weekly_df.dropna(subset=['Growth', 'TQQQ_Prev']).reset_index(drop=True)
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None

# -----------------------------------------------------------
# 2. 위대리 상태 복원 (시뮬레이션 엔진)
# -----------------------------------------------------------
def get_current_state(df, start_date, initial_capital, settings):
    # 실제 매매 직전(마지막 줄 제외)까지의 히스토리를 시뮬레이션하여 현재 잔고 복원
    df_sim = df.iloc[:-1].copy()
    sim_data = df_sim[df_sim['Date'] >= pd.to_datetime(start_date)].copy().reset_index(drop=True)
    
    cash = initial_capital
    shares = 0
    is_first = True
    max_c_usage = initial_capital * (settings['max_cash_pct'] / 100)

    for i, row in sim_data.iterrows():
        price = row['TQQQ']
        prev_p = row['TQQQ_Prev']
        mkt_eval = row['Eval']
        
        tier = 'MID'
        if mkt_eval > settings['uhigh_cut']/100: tier = 'UHIGH'
        elif mkt_eval > settings['high_cut']/100: tier = 'HIGH'
        elif mkt_eval < settings['ulow_cut']/100: tier = 'ULOW'
        elif mkt_eval < settings['low_cut']/100: tier = 'LOW'

        if is_first:
            qty = round(min(initial_capital * (settings['initial_entry_pct'] / 100), max_c_usage) / price)
            shares = qty; cash -= (qty * price); is_first = False
        else:
            diff = (shares * price) - (shares * prev_p)
            if diff > 0: # 수익 시 매도
                ratio = settings['sell_ratios'][tier] / 100
                qty = int(min(round((diff * ratio) / price), shares))
                shares -= qty; cash += (qty * price)
            elif diff < 0: # 손실 시 매수
                ratio = settings['buy_ratios'][tier] / 100
                avail = max_c_usage - (initial_capital - cash)
                if avail > 0:
                    qty = round(min(cash, abs(diff) * ratio, avail) / price)
                    if (qty * price) > cash: qty = int(cash // price)
                    shares += qty; cash -= (qty * price)
                    
    return cash, shares

def send_telegram_message(text):
    bot_token = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
    chat_id = "1442265681"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: print("❌ 텔레그램 전송 실패")

# -----------------------------------------------------------
# 3. 메인 실행 루틴
# -----------------------------------------------------------
def main():
    print("📈 [위대리 오토봇] 매매 판독 시작...")
    
    # 1. 시장 데이터 및 지표 계산
    df_market = get_market_data()
    if df_market is None: return

    # 2. 위대리 v2.5 설정값
    settings = {
        'max_cash_pct': 100, 'initial_entry_pct': 50,
        'uhigh_cut': 10, 'high_cut': 5, 'low_cut': -6, 'ulow_cut': -10,
        'sell_ratios': {'UHIGH': 150, 'HIGH': 100, 'MID': 60, 'LOW': 60, 'ULOW': 30},
        'buy_ratios': {'UHIGH': 30, 'HIGH': 60, 'MID': 60, 'LOW': 120, 'ULOW': 200}
    }
    initial_cap = 100000 # 원금 설정
    start_date = '2025-01-01' # 투자 시작일

    # 3. 현재 잔고 상태 복원
    cash_now, shares_now = get_current_state(df_market, start_date, initial_cap, settings)
    
    # 4. 오늘의 주문 계산 (마지막 행 데이터 사용)
    last_row = df_market.iloc[-1]
    cur_price = round(last_row['TQQQ'], 2)
    mkt_eval = last_row['Eval']
    
    tier = 'MID'
    if mkt_eval > 0.10: tier = 'UHIGH'
    elif mkt_eval > 0.05: tier = 'HIGH'
    elif mkt_eval < -0.10: tier = 'ULOW'
    elif mkt_eval < -0.06: tier = 'LOW'
    
    diff = (shares_now * cur_price) - (shares_now * last_row['TQQQ_Prev'])
    order_action, order_qty = "관망", 0
    
    if diff > 0: # 수익 구간 -> 매도
        ratio = settings['sell_ratios'][tier] / 100
        order_qty = int(min(round((diff * ratio) / cur_price), shares_now))
        if order_qty > 0: order_action = "매도"
    elif diff < 0: # 손실 구간 -> 매수
        ratio = settings['buy_ratios'][tier] / 100
        max_c = initial_cap * (settings['max_cash_pct'] / 100)
        avail = max_c - (initial_cap - cash_now)
        if avail > 0:
            order_qty = round(min(cash_now, abs(diff) * ratio, avail) / cur_price)
            if (order_qty * cur_price) > cash_now: order_qty = int(cash_now // cur_price)
            if order_qty > 0: order_action = "매수"

    # 5. 구글 시트 전송 (환경 변수에 저장된 JSON 키 사용)
    creds_json = os.environ.get('GCP_CREDENTIALS')
    if creds_json:
        try:
            credentials = json.loads(creds_json)
            gc = gspread.service_account_from_dict(credentials)
            sh = gc.open_by_key("1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI")
            worksheet = sh.worksheet("위대리")
            
            # L4: 매수/매도, M4: 방법, N4: 가격, O4: 수량 업데이트
            worksheet.update('L4:O4', [[order_action, "LOC", cur_price, order_qty]])
            print("📤 구글 시트 업데이트 완료!")
        except Exception as e:
            print(f"❌ 구글 시트 에러: {e}")

    # 6. 텔레그램 알림 전송
    total_asset = cash_now + (shares_now * cur_price)
    msg = f"🚀 *[위대리] 주간 매매 시그널*\n\n" \
          f"📊 시장모드: `{tier}` (평가율: {mkt_eval*100:.2f}%)\n" \
          f"💰 현재 총 자산: `${total_asset:,.0f}`\n\n" \
          f"🎯 *주문 요약 (LOC)*\n" \
          f"- 종류: *{order_action}*\n" \
          f"- 가격: `${cur_price}`\n" \
          f"- 수량: *{order_qty}* 주"
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
