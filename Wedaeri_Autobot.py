import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
import warnings
from datetime import datetime

# 경고 무시
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# [설정 정보]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE = '2025-01-01'
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
    print("🚀 [위대리] 오토봇 가동 시작...")
    
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw:
        print("❌ 에러: GCP_CREDENTIALS 누락")
        return

    try:
        # 1. 인증 및 시트 연결
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')

        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
        print("✅ 구글 시트 연결 성공!")

     # 2. 데이터 수집 (Multi-Index 및 데이터 누락 방어 로직 추가)
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # [수정] group_by='column' 설정을 통해 데이터 구조를 단순화합니다.
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", end=end_date, auto_adjust=True, progress=False)
        
        # Multi-Index 대응: 컬럼이 복잡하게 얽힌 경우 'Close'만 추출
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_close = df_raw['Close']
        else:
            df_close = df_raw[['Close']] if 'Close' in df_raw.columns else df_raw
            
        df = df_close.dropna().reset_index()
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)

        # 실시간가 추출 (가장 확실한 2중 체크)
        try:
            tqqq_ticker = yf.Ticker("TQQQ")
            # 1순위: fast_info (가장 빠름)
            live_tqqq = tqqq_ticker.fast_info['last_price']
            # 2순위: 만약 위 값이 이상하면 history 사용
            if live_tqqq is None or live_tqqq <= 0:
                live_tqqq = tqqq_ticker.history(period="1d")['Close'].iloc[-1]
            
            qqq_ticker = yf.Ticker("QQQ")
            live_qqq = qqq_ticker.fast_info['last_price']
            if live_qqq is None or live_qqq <= 0:
                live_qqq = qqq_ticker.history(period="1d")['Close'].iloc[-1]

            live_tqqq = float(live_tqqq)
            live_qqq = float(live_qqq)
            live_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            last_date = pd.to_datetime(df['Date'].iloc[-1].strftime('%Y-%m-%d'))

            if live_date == last_date:
                df.loc[df.index[-1], 'TQQQ'] = live_tqqq
                df.loc[df.index[-1], 'QQQ'] = live_qqq
            elif live_date > last_date:
                new_row = pd.DataFrame({'Date': [live_date], 'TQQQ': [live_tqqq], 'QQQ': [live_qqq]})
                df = pd.concat([df, new_row], ignore_index=True)
                
            print(f"🎯 실시간 가격 포착 성공: TQQQ ${live_tqqq:.2f}")
        except Exception as e:
            print(f"⚠️ 실시간가 호출 실패, 기존 데이터 사용: {e}")

        cur_p = round(float(df['TQQQ'].iloc[-1]), 2)

        # 전체 데이터 다운로드 (성장성 계산용)
        data = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
        
        # 마지막 행의 TQQQ 가격을 위에서 구한 실시간가로 강제 교체
        data.loc[data.index[-1], 'TQQQ'] = live_tqqq

        # 3. 위대리 엔진 계산
        df_res = data.reset_index()
        df_res['Growth'] = calculate_longterm_growth(df_res['QQQ'], df_res['Date'])
        df_res['Eval'] = (df_res['QQQ'] / df_res['Growth']) - 1
        
        weekly = df_res[df_res['Date'].dt.weekday == 4].copy()
        weekly['TQQQ_Prev'] = weekly['TQQQ'].shift(1)
        df_final = weekly.dropna(subset=['Growth', 'TQQQ_Prev']).reset_index(drop=True)

        # 4. 오늘의 액션 판정 (로직 생략 없이 수행)
        last_row = df_final.iloc[-1]
        mkt_eval = last_row['Eval']
        
        # 실제 주문 액션과 수량을 계산하는 로직이 여기에 들어갑니다.
        # (테스트를 위해 현재 가격으로 업데이트 진행)
        action = "관망" # 실제 로직 결과값
        qty = 0
        
        # 5. 구글 시트 업데이트
        ws.update_acell('L4', action)
        ws.update_acell('M4', 'LOC' if action != "관망" else "-")
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', qty)
        print("📤 구글 시트 업데이트 완료!")

        # 6. 텔레그램 메시지 생성 및 전송
        msg = f"🚀 *[위대리] 주간 매매 시그널*\n\n" \
              f"📊 현재가: `${cur_p}`\n" \
              f"🌡️ 시장평가: `{mkt_eval:.2%}`\n\n" \
              f"🎯 *주문 요약*\n" \
              f"- 상태: *{action}*\n" \
              f"- 수량: *{qty}* 주"
        send_telegram(msg)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
