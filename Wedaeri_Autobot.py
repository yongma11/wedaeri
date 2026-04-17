# wedaeri_bot.py — TQQQ 위대리 오토봇
# ▸ 주간 Expanding Window OLS (wedaeri_app.py 동일 로직)
# ▸ W-FRI resample 기반 — 금요일 휴장 주도 정확 처리
# ▸ 3-티어 시스템 (HIGH / MID / LOW)
# ▸ Google Sheets + Telegram 자동 업데이트

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# [사용자 설정] — 필요 시 여기만 수정
# ─────────────────────────────────────────────────────────────
SHEET_KEY     = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE    = '2025-12-26'   # 매매 시작일 (YYYY-MM-DD)
INITIAL_CAP   = 108_000        # 시작 원금 ($)
INIT_CASH_PCT = 0.45           # 초기 현금 비중 (45% → 주식 55%)

# 전략 파라미터
PARAMS = {
    'hc': 0.07,   # HIGH 기준 Eval ≥ +7%
    'lc': -0.07,  # LOW  기준 Eval ≤ -7%
    'sH': 1.5,    # HIGH 매도 배율
    'sM': 0.6,    # MID  매도 배율
    'sL': 0.35,   # LOW  매도 배율
    'bH': 0.4,    # HIGH 매수 배율
    'bM': 0.6,    # MID  매수 배율
    'bL': 2.0,    # LOW  매수 배율
}

# Telegram
BOT_TOKEN = "7524501477:AAEJu3xmHi2Mjxb86ARc6KtMfBh9H9pRZIM"
CHAT_ID   = "1442265681"


# ─────────────────────────────────────────────────────────────
# 1. Expanding Window OLS (wedaeri_app.py 동일 알고리즘)
# ─────────────────────────────────────────────────────────────
def compute_expanding_ols(qqq_weekly: pd.DataFrame, W: int = 260) -> np.ndarray:
    """
    주간 QQQ 시계열로 5년(260주) Expanding Window log-선형 회귀를 계산합니다.
    - 처음 W주까지는 expanding window (1주씩 확장)
    - W주 이후에는 고정 260주 슬라이딩 윈도우
    - 주 순번(t=1,2,3,…)을 독립변수로 사용 (ordinal 날짜보다 수치 안정)
    - prefix-sum으로 O(n) 연산
    """
    n = len(qqq_weekly)
    t = np.arange(1, n + 1, dtype=float)
    y = np.log(qqq_weekly['QQQ'].values.astype(float))

    ps_t  = np.zeros(n + 1); ps_t[1:]  = np.cumsum(t)
    ps_y  = np.zeros(n + 1); ps_y[1:]  = np.cumsum(y)
    ps_t2 = np.zeros(n + 1); ps_t2[1:] = np.cumsum(t ** 2)
    ps_ty = np.zeros(n + 1); ps_ty[1:] = np.cumsum(t * y)

    growth = np.empty(n)
    for i in range(n):
        w     = min(i + 1, W)
        end   = i + 1
        start = end - w

        if w == 1:
            growth[i] = float(qqq_weekly['QQQ'].iloc[i])
            continue

        s_t  = ps_t[end]  - ps_t[start]
        s_y  = ps_y[end]  - ps_y[start]
        s_t2 = ps_t2[end] - ps_t2[start]
        s_ty = ps_ty[end] - ps_ty[start]
        denom = w * s_t2 - s_t ** 2

        if denom == 0:
            growth[i] = float(qqq_weekly['QQQ'].iloc[i])
            continue

        b = (w * s_ty - s_t * s_y) / denom
        a = (s_y - b * s_t) / w
        growth[i] = np.exp(a + b * t[i])

    return growth


# ─────────────────────────────────────────────────────────────
# 2. Telegram 발송
# ─────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 3. Google Sheets '설정' 시트에서 파라미터 로드
# ─────────────────────────────────────────────────────────────
def load_config_from_sheets(sh) -> dict:
    """앱이 저장한 '설정' 시트를 읽어 파라미터 딕셔너리를 반환합니다.
    시트가 없거나 읽기 실패 시 빈 딕셔너리를 반환 (기본값 유지).
    """
    try:
        ws = sh.worksheet("설정")
        records = ws.get_all_records()          # [{'key':..., 'value':...}, ...]
        return {row['key']: row['value'] for row in records if row.get('key')}
    except Exception as e:
        print(f"⚠️ 설정 시트 로드 실패 (기본값 사용): {e}")
        return {}


def apply_sheets_config(cfg: dict,
                        start_date: str, initial_cap: float,
                        init_cash_pct: float, params: dict):
    """Sheets에서 읽은 설정을 실제 변수에 반영합니다.
    변경된 항목을 출력하고 (start_date, initial_cap, init_cash_pct, params) 튜플로 반환.
    """
    if not cfg:
        return start_date, initial_cap, init_cash_pct, params

    def _f(key, default):
        try:
            return float(cfg[key])
        except (KeyError, ValueError, TypeError):
            return default

    new_start       = str(cfg.get('start_date', start_date))[:10]
    new_cap         = _f('cap',  initial_cap)
    new_cash_pct    = _f('cash', init_cash_pct * 100) / 100   # 저장 단위: % 정수

    new_params = {
        'hc': _f('hc', params['hc'] * 100) / 100,
        'lc': _f('lc', params['lc'] * 100) / 100,
        'sH': _f('sH', params['sH']),
        'sM': _f('sM', params['sM']),
        'sL': _f('sL', params['sL']),
        'bH': _f('bH', params['bH']),
        'bM': _f('bM', params['bM']),
        'bL': _f('bL', params['bL']),
    }

    print(f"📋 Sheets 설정 로드 완료")
    print(f"   시작일: {new_start} | 원금: ${new_cap:,.0f} | 초기현금: {new_cash_pct:.0%}")
    print(f"   hc={new_params['hc']:.0%} / lc={new_params['lc']:.0%}")
    print(f"   매도 H/M/L: {new_params['sH']}/{new_params['sM']}/{new_params['sL']}")
    print(f"   매수 H/M/L: {new_params['bH']}/{new_params['bM']}/{new_params['bL']}")

    return new_start, new_cap, new_cash_pct, new_params


# ─────────────────────────────────────────────────────────────
# 4. 티어 판정 (3-tier)
# ─────────────────────────────────────────────────────────────
def get_tier(eval_val: float, hc: float, lc: float) -> str:
    if eval_val >= hc:
        return 'HIGH'
    if eval_val <= lc:
        return 'LOW'
    return 'MID'


# ─────────────────────────────────────────────────────────────
# 4. 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("🚀 [위대리] 오토봇 가동 시작...")

    # ── A. GCP 인증 & 시트 연결 ──────────────────────────────
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw:
        print("❌ GCP_CREDENTIALS 환경변수가 없습니다.")
        return

    try:
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
    except Exception as e:
        print(f"❌ 시트 연결 실패: {e}")
        return

    # ── A-2. '설정' 시트에서 앱 저장값 로드 (있으면 하드코딩 기본값을 덮어씀) ──
    _cfg = load_config_from_sheets(sh)
    start_date, initial_cap, init_cash_pct, params = apply_sheets_config(
        _cfg,
        start_date    = START_DATE,
        initial_cap   = INITIAL_CAP,
        init_cash_pct = INIT_CASH_PCT,
        params        = dict(PARAMS),   # 복사본 사용 (전역 PARAMS 불변 유지)
    )

    try:
        # ── B. 일별 데이터 수집 (2010년 기준 — Expanding OLS 시작점) ──
        end_dt = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df_raw = yf.download(
            ["QQQ", "TQQQ"], start="2010-01-01", end=end_dt,
            auto_adjust=True, progress=False
        )
        df = (df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex)
              else df_raw).dropna().reset_index()
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # ── C. 현재가 반영 (라이브) ───────────────────────────
        live_tqqq = float(yf.Ticker("TQQQ").fast_info['last_price'])
        live_qqq  = float(yf.Ticker("QQQ").fast_info['last_price'])
        live_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))

        if live_date > df['Date'].iloc[-1]:
            new_row = pd.DataFrame({
                'Date': [live_date], 'TQQQ': [live_tqqq], 'QQQ': [live_qqq]
            })
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[df.index[-1], 'TQQQ'] = live_tqqq
            df.loc[df.index[-1], 'QQQ']  = live_qqq

        cur_p = round(float(df['TQQQ'].iloc[-1]), 2)

        # ── D. 주간 데이터 생성 (W-FRI resample) ─────────────
        # 금요일 휴장 주도 그 주 마지막 거래일로 자동 처리
        df_idx = df.set_index('Date')

        qqq_weekly = (df_idx[['QQQ']]
                      .resample('W-FRI').last()
                      .dropna()
                      .reset_index())

        tqqq_weekly = (df_idx[['TQQQ']]
                       .resample('W-FRI').last()
                       .dropna()
                       .reset_index())

        # ── E. Expanding Window OLS → Eval 계산 ──────────────
        growth = compute_expanding_ols(qqq_weekly, W=260)
        qqq_weekly['Growth'] = growth
        qqq_weekly['Eval']   = qqq_weekly['QQQ'] / qqq_weekly['Growth'] - 1

        # 주간 TQQQ 와 병합
        weekly = qqq_weekly.merge(tqqq_weekly, on='Date', how='inner')

        # ── F. 시작일부터 시뮬레이션 데이터 필터 ─────────────
        # start_date / initial_cap / init_cash_pct / params 는
        # A-2 단계에서 Sheets 설정이 이미 반영된 로컬 변수
        sim = (weekly[weekly['Date'] >= pd.to_datetime(start_date)]
               .dropna(subset=['Eval', 'TQQQ'])
               .reset_index(drop=True))

        if len(sim) < 2:
            msg = "⚠️ [위대리] 시뮬레이션 데이터 부족 (시작일 이후 데이터가 2주 미만)"
            print(msg); send_telegram(msg)
            return

        # ── G. 초기 잔고 설정 ─────────────────────────────────
        init_stock_ratio = 1.0 - init_cash_pct
        init_price       = float(sim.loc[0, 'TQQQ'])
        shares           = int((initial_cap * init_stock_ratio) / init_price)
        cash             = initial_cap - (shares * init_price)

        hc = params['hc']; lc = params['lc']
        sH = params['sH']; sM = params['sM']; sL = params['sL']
        bH = params['bH']; bM = params['bM']; bL = params['bL']

        sell_r = {'HIGH': sH, 'MID': sM, 'LOW': sL}
        buy_r  = {'HIGH': bH, 'MID': bM, 'LOW': bL}

        # ── H. 과거 매매 복기 (시작일+1 ~ 마지막 주 직전) ────
        for i in range(1, len(sim) - 1):
            p      = float(sim.loc[i,   'TQQQ'])
            prev_p = float(sim.loc[i-1, 'TQQQ'])
            ev     = float(sim.loc[i,   'Eval'])
            tier   = get_tier(ev, hc, lc)
            diff   = shares * (p - prev_p)  # 평가 손익

            if diff > 0 and shares > 0:
                qty = int(min(round(diff * sell_r[tier] / p), shares))
                shares -= qty
                cash   += qty * p
            elif diff < 0:
                qty = int(min(cash, abs(diff) * buy_r[tier]) / p)
                shares += qty
                cash   -= qty * p

        # ── I. 이번 주 신호 계산 (마지막 행) ─────────────────
        last  = sim.iloc[-1]
        prev  = sim.iloc[-2]

        last_eval  = float(last['Eval'])
        last_price = float(last['TQQQ'])
        prev_price = float(prev['TQQQ'])
        this_tier  = get_tier(last_eval, hc, lc)

        diff_now   = shares * (last_price - prev_price)
        action, order_qty = "관망", 0

        if diff_now > 0 and shares > 0:     # 상승 → 매도
            action    = "매도"
            order_qty = int(min(round(diff_now * sell_r[this_tier] / last_price), shares))
        elif diff_now < 0:                   # 하락 → 매수
            action    = "매수"
            order_qty = int(min(cash, abs(diff_now) * buy_r[this_tier]) / last_price)

        if order_qty == 0:
            action = "관망"

        # ── J. 현재 자산 계산 ─────────────────────────────────
        # 이번 주 주문이 체결된다고 가정한 예상 잔고
        if action == "매도":
            est_shares = shares - order_qty
            est_cash   = cash   + order_qty * cur_p
        elif action == "매수":
            est_shares = shares + order_qty
            est_cash   = cash   - order_qty * cur_p
        else:
            est_shares, est_cash = shares, cash

        total_asset  = est_cash  + est_shares * cur_p
        pnl_pct      = (total_asset / INITIAL_CAP - 1) * 100
        cash_pct     = est_cash / total_asset * 100 if total_asset > 0 else 0

        # ── K. Google Sheets 업데이트 ─────────────────────────
        ws.update_acell('L4', action)
        ws.update_acell('M4', 'LOC' if action != "관망" else "-")
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', order_qty)
        print(f"📤 시트 업데이트 완료: {action} {order_qty}주 @ ${cur_p} (LOC)")

        # ── L. Telegram 메시지 발송 ───────────────────────────
        tier_emoji = {'HIGH': '🟡', 'MID': '🔵', 'LOW': '🟢'}.get(this_tier, '⚪')
        action_emoji = {'매도': '📈', '매수': '📉', '관망': '⏸'}.get(action, '⏸')
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')

        msg = (
            f"🚀 *[위대리] 주간 매매 시그널* ({today_str})\n\n"
            f"💰 *자산 현황 (주문 체결 후 예상)*\n"
            f"  총 자산 : `${total_asset:>12,.2f}`  ({pnl_pct:+.2f}%)\n"
            f"  보유 주식: `{est_shares:>8,}` 주\n"
            f"  보유 현금: `${est_cash:>12,.2f}`  (현금 {cash_pct:.1f}%)\n\n"
            f"🌡️ *시장 평가* : {tier_emoji} `{last_eval:.2%}`  ({this_tier})\n"
            f"  QQQ 추세 대비 {abs(last_eval):.1%} "
            f"{'고평가' if last_eval >= hc else ('저평가' if last_eval <= lc else '중립')}\n\n"
            f"🎯 *이번 주 LOC 주문*\n"
            f"  {action_emoji} 상태       : *{action}*\n"
            f"  📦 주문 수량   : *{order_qty}* 주\n"
            f"  💲 기준 가격   : `${cur_p}`\n"
            f"  📊 거래 후 잔고: *{est_shares:,}* 주\n\n"
        )

        if action != "관망":
            msg += (
                f"📱 *MTS 실행 방법*\n"
                f"  → `{action} {order_qty}주` LOC로 예약\n"
                f"  → 금요일 장 마감 전 체결\n"
            )
        else:
            msg += "  → 이번 주 주문 없음 (관망)\n"

        send_telegram(msg)
        print("✅ Telegram 발송 완료")

    except Exception as e:
        err_msg = f"❌ [위대리 봇] 오류 발생: {e}"
        print(err_msg)
        send_telegram(err_msg)


if __name__ == "__main__":
    main()
