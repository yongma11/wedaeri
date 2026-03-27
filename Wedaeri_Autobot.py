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
START_DATE = '2026-02-02'  # 매매 시작일
INITIAL_CAP = 100000       # 시작 원금 ($)
INITIAL_RATIO = 0.59       # 초기 진입 비중 (59%)

# 2. 장기 추세선(Growth) 계산 함수
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

# 3. 텔레그램 전송 함수
def send_telegram(text):
    bot_token = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
    chat_id = "1442265681"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

# 4. 메인 실행 함수
def main():
    print("🚀 [위대리 v2.7] 오토봇 가동 시작...")
    
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw:
        print("❌ 에러: GCP_CREDENTIALS 환경변수가 없습니다.")
        return

    try:
        # --- A. 인증 및 시트 연결 ---
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
        print("✅ 구글 시트 연결 성공!")

        # --- B. 데이터 수집 및 현재가 확정 (시그마2 정밀 로직) ---
        end_dt = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", end=end_dt, auto_adjust=True, progress=False)
        
        # Multi-Index 대응 및 정리
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)

        # 실시간가 포착 (fast_info 활용)
        t_ticker = yf.Ticker("TQQQ")
        live_p = float(t_ticker.fast_info['last_price'])
        live_d = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        # 날짜 비교 후 데이터 프레임 업데이트
        if live_d > pd.to_datetime(df['Date'].iloc[-1]):
            new_row = pd.DataFrame({'Date': [live_d], 'TQQQ': [live_p], 'QQQ': [yf.Ticker("QQQ").fast_info['last_price']]})
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[df.index[-1], 'TQQQ'] = live_p
        
        cur_p = round(float(df['TQQQ'].iloc[-1]), 2)
        print(f"📊 현재가 포착: TQQQ ${cur_p}")

        # --- C. 위대리 지표 계산 ---
        df['Growth'] = calculate_longterm_growth(df['QQQ'], df['Date'])
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        
        # 주간(금요일) 데이터 필터링
        weekly = df[df['Date'].dt.weekday == 4].copy()
        sim_data = weekly[weekly['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)

        if sim_data.empty:
            print("⚠️ 시작일 이후의 데이터가 아직 없습니다.")
            return

        # --- D. 잔고 및 수익률 시뮬레이션 ---
        # 초기 진입 설정 (26년 2월 2일)
        first_price = sim_data['TQQQ'].iloc[0]
        entry_cash = INITIAL_CAP * INITIAL_RATIO
        shares = int(entry_cash / first_price)
        cash = INITIAL_CAP - (shares * first_price)
        avg_price = first_price  # 초기 평단가
        
        # (여기에 주별 매매 로직 반영 가능 - 현재는 보유 유지 기준 계산)
        total_asset = cash + (shares * cur_p)
        profit_pnl = total_asset - INITIAL_CAP
        profit_rate = (total_asset / INITIAL_CAP - 1) * 100

        # --- E. 구글 시트 업데이트 (L4:O4) ---
        # 매매 판독 로직 (예시: 관망)
        action, qty = "관망", 0 
        ws.update_acell('L4', action)
        ws.update_acell('M4', 'LOC' if action != "관망" else "-")
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', qty)
        print("📤 구글 시트 업데이트 완료!")

        # --- F. 텔레그램 리포트 전송 ---
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        mkt_eval = df['Eval'].iloc[-1]
        
        msg = f"🚀 *[위대리] 주간 매매 리포트* ({today_str})\n\n" \
              f"💰 *자산 현황*\n" \
              f"- 예상 총 자산: `${total_asset:,.2f}`\n" \
              f"- 누적 수익: `${profit_pnl:,.2f}` ({profit_rate:.2f}%)\n" \
              f"- 보유 현금: `${cash:,.2f}`\n\n" \
              f"📈 *보유 주식 정보*\n" \
              f"- 종목: *TQQQ*\n" \
              f"- 수량: *{shares}* 주\n" \
              f"- 평단가: `${avg_price:.2f}`\n" \
              f"- 현재가: `${cur_p}`\n\n" \
              f"🌡️ *시장 평가*: `{mkt_eval:.2%}`\n" \
              f"🎯 *이번 주 주문*: *{action} {qty}주*\n\n" \
              f"📅 기준일: `{START_DATE}` (초기비중 59%)"
        
        send_telegram(msg)
        print("✅ 텔레그램 리포트 전송 완료!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
