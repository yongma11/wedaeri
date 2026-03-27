import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
import warnings
from datetime import datetime

# Polyfit 연산 시 발생하는 불필요한 경고 무시
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# ==========================================
# [설정] 용성님, 시트 정보와 원금을 확인하세요!
# ==========================================
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE = '2025-01-01'  # 실제 매매 시작일
INITIAL_CAP = 100000       # 투자 원금 ($)
# ==========================================

# 1. 위대리 장기 추세선 엔진 (25년 로그 회귀)
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

# 2. 시장 데이터 수집 및 실시간가(예상종가) 반영
def get_market_data():
    try:
        df = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)['Close'].dropna()
        
        # 장마감 15분 전 실시간 가격 가져오기
        tqqq_ticker = yf.Ticker("TQQQ")
        qqq_ticker = yf.Ticker("QQQ")
        live_tqqq = tqqq_ticker.fast_info['last_price']
        live_qqq = qqq_ticker.fast_info['last_price']
        
        # 마지막 행 가격 업데이트
        df.iloc[-1, df.columns.get_loc('TQQQ')] = live_tqqq
        df.iloc[-1, df.columns.get_loc('QQQ')] = live_qqq

        df_reset = df.reset_index()
        df_reset['Growth'] = calculate_longterm_growth(df_reset['QQQ'], df_reset['Date'])
        df_reset['Eval'] = (df_reset['QQQ'] / df_reset['Growth']) - 1
        
        # 주간(금요일) 데이터 추출
        weekly_df = df_reset[df_reset['Date'].dt.weekday == 4].copy()
        weekly_df['TQQQ_Prev'] = weekly_df['TQQQ'].shift(1)
        
        return weekly_df.dropna(subset=['Growth', 'TQQQ_Prev']).reset_index(drop=True)
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None

# 3. 텔레그램 알림 함수
def send_telegram(text):
    bot_token = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
    chat_id = "1442265681"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try: requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except: pass

# 4. 메인 실행 루틴
def main():
    print("🚀 위대리 오토봇 가동 (GitHub Actions)")
    
    # 1. 환경 변수 로드
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw:
        print("❌ 에러: GCP_CREDENTIALS 설정이 필요합니다.")
        return

    try:
        # 데이터 타입 체크 및 변환
        creds_json = json.loads(creds_raw.strip()) if isinstance(creds_raw, str) else creds_raw
        
        # [무조건 성공하는 치트키] 
        # 구글 인증 키의 핵심은 첫 줄과 끝 줄, 그리고 그 사이의 줄바꿈입니다.
        if 'private_key' in creds_json:
            pk = creds_json['private_key']
            # 기존에 섞여있을 수 있는 잘못된 줄바꿈 기호들을 모두 정리하고 
            # 구글 표준인 \n (개행문자)로 통일합니다.
            pk = pk.replace('\\n', '\n').replace('\n', '\\n').replace('\\n', '\n')
            creds_json['private_key'] = pk

        # 서비스 계정 인증
        gc = gspread.service_account_from_dict(creds_json)
        
        # 시트 열기
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
        
        print("✅ 구글 시트 연결 성공!")

        # 데이터 계산 시작
        df_market = get_market_data()
        if df_market is None: return

        # 위대리 v2.5 설정값
        settings = {
            'max_cash_pct': 100, 'initial_entry_pct': 50,
            'uhigh_cut': 0.10, 'high_cut': 0.05, 'low_cut': -0.06, 'ulow_cut': -0.10,
            'sell_ratios': {'UHIGH': 1.5, 'HIGH': 1.0, 'MID': 0.6, 'LOW': 0.6, 'ULOW': 0.3},
            'buy_ratios': {'UHIGH': 0.3, 'HIGH': 0.6, 'MID': 0.6, 'LOW': 1.2, 'ULOW': 2.0}
        }
        
        # 잔고 시뮬레이션 로직
        sim_data = df_market[df_market['Date'] >= pd.to_datetime(START_DATE)].iloc[:-1]
        cash, shares = INITIAL_CAP, 0
        is_first = True
        max_c = INITIAL_CAP * (settings['max_cash_pct'])

        for _, row in sim_data.iterrows():
            p, prev_p, m_eval = row['TQQQ'], row['TQQQ_Prev'], row['Eval']
            tier = 'MID'
            if m_eval > settings['uhigh_cut']: tier = 'UHIGH'
            elif m_eval > settings['high_cut']: tier = 'HIGH'
            elif m_eval < settings['ulow_cut']: tier = 'ULOW'
            elif m_eval < settings['low_cut']: tier = 'LOW'

            if is_first:
                q = round(min(INITIAL_CAP * (settings['initial_entry_pct']), max_c) / p)
                shares = q; cash -= (q * p); is_first = False
            else:
                diff = (shares * p) - (shares * prev_p)
                if diff > 0:
                    q = int(min(round((diff * settings['sell_ratios'][tier]) / p), shares))
                    shares -= q; cash += (q * p)
                elif diff < 0:
                    avail = max_c - (INITIAL_CAP - cash)
                    if avail > 0:
                        q = round(min(cash, abs(diff) * settings['buy_ratios'][tier], avail) / p)
                        if (q * p) > cash: q = int(cash // p)
                        shares += q; cash -= (q * p)

        # 오늘의 최종 주문 계산
        last = df_market.iloc[-1]
        cur_p, m_eval = round(last['TQQQ'], 2), last['Eval']
        tier = 'MID'
        if m_eval > settings['uhigh_cut']: tier = 'UHIGH'
        elif m_eval > settings['high_cut']: tier = 'HIGH'
        elif m_eval < settings['ulow_cut']: tier = 'ULOW'
        elif m_eval < settings['low_cut']: tier = 'LOW'

        diff = (shares * cur_p) - (shares * last['TQQQ_Prev'])
        action, qty = "관망", 0
        
        if diff > 0:
            action = "매도"
            qty = int(min(round((diff * settings['sell_ratios'][tier]) / cur_p), shares))
        elif diff < 0:
            action = "매수"
            avail = max_c - (INITIAL_CAP - cash)
            qty = round(min(cash, abs(diff) * settings['buy_ratios'][tier], avail) / cur_p)
            if (qty * cur_p) > cash: qty = int(cash // cur_p)

        # 구글 시트 업데이트 (L4:O4)
        ws.update('L4:O4', [[action, "LOC", cur_p, qty if qty > 0 else 0]])
        print(f"📤 시트 업데이트 완료: {action} {qty}주 @ {cur_p}")

        # 텔레그램 전송
        total_asset = cash + (shares * cur_p)
        msg = f"🚀 *[위대리] 주간 매매 시그널*\n\n" \
              f"📊 시장모드: `{tier}`\n" \
              f"💰 총 자산: `${total_asset:,.0f}`\n\n" \
              f"🎯 *주문 요약 (LOC)*\n" \
              f"- 종류: *{action}*\n" \
              f"- 가격: `${cur_p}`\n" \
              f"- 수량: *{qty}* 주"
        send_telegram(msg)

    except Exception as e:
        print(f"❌ 인증/연결 오류 발생: {e}")

if __name__ == "__main__":
    main()
