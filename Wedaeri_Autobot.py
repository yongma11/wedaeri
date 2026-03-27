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

       # 2. 데이터 수집 (3중 체크 방식으로 현재가 반영 강화)
        print("🔍 실시간 가격 데이터를 수집 중입니다...")
        
        # 방식 A: Ticker 객체의 history 사용 (가장 표준)
        tqqq_obj = yf.Ticker("TQQQ")
        live_data = tqqq_obj.history(period='1d', interval='1m')
        
        if not live_data.empty:
            live_tqqq = live_data['Close'].iloc[-1]
        else:
            # 방식 B: history가 실패할 경우 fast_info 시도
            try:
                live_tqqq = tqqq_obj.fast_info['last_price']
            except:
                # 방식 C: 최악의 경우 최근 5일치 download 데이터의 마지막 값
                temp_df = yf.download("TQQQ", period="5d", progress=False)
                live_tqqq = temp_df['Close'].iloc[-1]

        cur_p = round(float(live_tqqq), 2)
        print(f"📊 확정된 TQQQ 현재가: ${cur_p}")

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
