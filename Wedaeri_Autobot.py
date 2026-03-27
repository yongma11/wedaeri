import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
import warnings
from datetime import datetime, timedelta  # timedelta를 추가해야 합니다!

# 경고 무시
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# [설정 정보]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE = '2026-02-02'
INITIAL_CAP = 100000

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
        res = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        if res.status_code == 200:
            print("✅ 텔레그램 메시지 전송 성공!")
        else:
            print(f"❌ 텔레그램 전송 실패 (코드: {res.status_code})")
    except Exception as e:
        print(f"❌ 텔레그램 에러: {e}")

def main():
    print("🚀 [위대리] 잔고 관리 강화 버전 가동...")
    
    # [사용자 설정 파라미터]
    START_DATE = '2026-02-02'
    INITIAL_CAP = 100000  # 10만 불
    INITIAL_RATIO = 0.59  # 초기 진입 비중 59%

    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw: return

    try:
        # 1. 인증 및 시트 연결
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")

        # 2. 데이터 수집 및 현재가 확정 (시그마2 정밀 로직)
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", end=end_date, auto_adjust=True, progress=False)
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        
        # 실시간가 보정
        tqqq_ticker = yf.Ticker("TQQQ")
        live_tqqq = float(tqqq_ticker.fast_info['last_price'])
        live_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        if live_date > pd.to_datetime(df['Date'].iloc[-1]):
            new_row = pd.DataFrame({'Date': [live_date], 'TQQQ': [live_tqqq], 'QQQ': [yf.Ticker("QQQ").fast_info['last_price']]})
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[df.index[-1], 'TQQQ'] = live_tqqq

        cur_p = round(float(df['TQQQ'].iloc[-1]), 2)

        # 3. 위대리 엔진 계산 (Growth & Eval)
        df['Growth'] = calculate_longterm_growth(df['QQQ'], df['Date'])
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        
        # 주간 단위 필터링 (금요일 기준)
        weekly = df[df['Date'].dt.weekday == 4].copy()
        sim_data = weekly[weekly['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)

        # 4. 잔고 시뮬레이션 (26년 2월 2일 기준)
        cash = INITIAL_CAP * (1 - INITIAL_RATIO)
        first_price = sim_data['TQQQ'].iloc[0]
        shares = int((INITIAL_CAP * INITIAL_RATIO) / first_price)
        avg_price = first_price
        
        # 매주 시뮬레이션 진행 (실제 매매 로직 반영 가능)
        for i in range(1, len(sim_data)):
            # 여기에 매주 action에 따른 cash, shares 변동 로직을 추가할 수 있습니다.
            # 현재는 보유 상태를 유지하며 자산 가치만 계산합니다.
            pass

        total_asset = cash + (shares * cur_p)
        profit_rate = (total_asset / INITIAL_CAP - 1) * 100

        # 5. 구글 시트 업데이트
        action, qty = "관망", 0 # 판독 로직 결과값
        ws.update_acell('L4', action)
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', qty)

        # 6. 텔레그램 전송 (요청하신 정보 포함)
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        msg = f"🚀 *[위대리] 주간 매매 리포트* ({today_str})\n\n" \
              f"💰 *자산 현황*\n" \
              f"- 예상 총 자산: `${total_asset:,.2f}`\n" \
              f"- 수익률: *{profit_rate:.2f}%*\n" \
              f"- 보유 현금: `${cash:,.2f}`\n\n" \
              f"📈 *보유 주식 정보*\n" \
              f"- 종목: *TQQQ*\n" \
              f"- 보유 수량: *{shares}* 주\n" \
              f"- 현재가: `${cur_p}`\n" \
              f"- 평단가: `${avg_price:.2f}`\n\n" \
              f"🎯 *이번 주 주문 (LOC)*\n" \
              f"- 상태: *{action}*\n" \
              f"- 수량: *{qty}* 주\n\n" \
              f"📅 시작일: `{START_DATE}` (초기비중 59%)"
        send_telegram(msg)
        print("✅ 텔레그램 리포트 전송 완료!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
