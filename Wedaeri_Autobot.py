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
    print("🚀 [위대리 v2.8] 오토봇 가동 시작...")
    
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
        print(f"📊 현재가 포착: TQQQ ${cur_p}")

        # --- C. 지표 계산 ---
        df['Growth'] = calculate_longterm_growth(df['QQQ'], df['Date'])
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        
        # 주간(금요일) 데이터 필터링 및 시뮬레이션 준비
        weekly = df[df['Date'].dt.weekday == 4].copy()
        sim_data = weekly[weekly['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)

        if sim_data.empty:
            print("⚠️ 시작일 이후 데이터 부족")
            return

        # --- D. 잔고 및 수량 시뮬레이션 (위대리 표준 로직 반영) ---
        # 매매 설정 (v2.5 기준)
        settings = {
            'uhigh_cut': 0.10, 'high_cut': 0.05, 'low_cut': -0.06, 'ulow_cut': -0.10,
            'sell_ratios': {'UHIGH': 1.5, 'HIGH': 1.0, 'MID': 0.6, 'LOW': 0.6, 'ULOW': 0.3},
            'buy_ratios': {'UHIGH': 0.3, 'HIGH': 0.6, 'MID': 0.6, 'LOW': 1.2, 'ULOW': 2.0}
        }
        max_c = INITIAL_CAP  # 최대 가용 현금

        # 초기 진입 (START_DATE 기준)
        first_row = sim_data.iloc[0]
        shares = int((INITIAL_CAP * INITIAL_RATIO) / first_row['TQQQ'])
        cash = INITIAL_CAP - (shares * first_row['TQQQ'])
        avg_price = first_row['TQQQ']
        
        # 매주 금요일 매매 복기 루프
        for i in range(1, len(sim_data)):
            p = sim_data.loc[i, 'TQQQ']
            prev_p = sim_data.loc[i-1, 'TQQQ']
            m_eval = sim_data.loc[i, 'Eval']

            # 티어 판정
            tier = 'MID'
            if m_eval > settings['uhigh_cut']: tier = 'UHIGH'
            elif m_eval > settings['high_cut']: tier = 'HIGH'
            elif m_eval < settings['ulow_cut']: tier = 'ULOW'
            elif m_eval < settings['low_cut']: tier = 'LOW'

            price_diff = (shares * p) - (shares * prev_p)
            
            # 매도 시그널 (수익 발생 시 일부 매도)
            if price_diff > 0 and shares > 0:
                q_sell = int(min(round((price_diff * settings['sell_ratios'][tier]) / p), shares))
                if q_sell > 0:
                    shares -= q_sell
                    cash += (q_sell * p)
            
            # 매수 시그널 (하락 시 일부 추매)
            elif price_diff < 0:
                avail = max_c - (INITIAL_CAP - cash) # 가용 현금 범위 내
                if avail > 0:
                    q_buy = int(min(cash, abs(price_diff) * settings['buy_ratios'][tier], avail) / p)
                    if q_buy > 0:
                        # 평단가 업데이트
                        total_cost = (shares * avg_price) + (q_buy * p)
                        shares += q_buy
                        avg_price = total_cost / shares
                        cash -= (q_buy * p)

        # 현재 기준 최종 자산가치
        total_asset = cash + (shares * cur_p)
        profit_rate = (total_asset / INITIAL_CAP - 1) * 100

        # --- E. 구글 시트 업데이트 (L4:O4) ---
        # 오늘의 최종 액션 판독
        last_mkt_eval = df['Eval'].iloc[-1]
        action = "관망"
        if last_mkt_eval < -0.06: action = "매수"
        elif last_mkt_eval > 0.05: action = "매도"
        
        ws.update_acell('L4', action)
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', shares) # 현재 보유 수량 반영
        print(f"📤 시트 업데이트 완료! 수량: {shares}주")

        # --- F. 텔레그램 리포트 전송 ---
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        msg = f"🚀 *[위대리] 주간 매매 리포트* ({today_str})\n\n" \
              f"💰 *자산 현황*\n" \
              f"- 예상 총 자산: `${total_asset:,.2f}`\n" \
              f"- 누적 수익률: *{profit_rate:.2f}%*\n" \
              f"- 보유 현금: `${cash:,.2f}`\n\n" \
              f"📈 *보유 주식 정보*\n" \
              f"- 종목: *TQQQ*\n" \
              f"- 보유 수량: *{shares}* 주\n" \
              f"- 평단가: `${avg_price:.2f}`\n" \
              f"- 현재가: `${cur_p}`\n\n" \
              f"🌡️ *시장 평가*: `{last_mkt_eval:.2%}`\n" \
              f"🎯 *상태*: *{action}*\n\n" \
              f"📅 기준일: `{START_DATE}` (초기비중 59%)"
        
        send_telegram(msg)
        print("✅ 텔레그램 리포트 전송 완료!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
