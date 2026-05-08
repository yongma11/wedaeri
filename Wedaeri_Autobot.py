# wedaeri_bot.py — TQQQ 위대리 오토봇 (v4.3)
# ▸ 주간 Expanding Window OLS (wedaeri_app.py 동일 로직)
# ▸ W-FRI resample 기반 — 금요일 휴장 주도 정확 처리
# ▸ 3-티어 시스템 (HIGH / MID / LOW)
# ▸ Google Sheets + Telegram 자동 업데이트
#
# v4.3 변경사항 (vs v4.2):
#   • F-가드 (변동성 정규화) 제거
#       — 사용자 실데이터 운용 구간에서 성과 저하 확인됨
#       — 워크포워드 평균에서는 Calmar 개선이지만 단기 구간에선 변동성 추정 노이즈가 큼
#       — 향후 충분한 운용 데이터가 쌓이면 재검토 가능
#   • v4.1 의 버그 수정 + v4.2 의 pre-market 타이밍 가드는 모두 유지
#
# v4.1/v4.2 유지 사항:
#   • 실행 타이밍 가드 — 금요일 ET 04:00–09:30 (pre-market) + 16:05–토 06:00 (post-close)
#   • PnL 계산이 Sheets 의 initial_cap 사용 (상수 제거)
#   • 봇 가상잔고 vs 실잔고 reconciliation
#   • 주석 +6% 정정

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import requests
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# [사용자 설정]
# ─────────────────────────────────────────────────────────────
SHEET_KEY     = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE    = '2025-12-26'
INITIAL_CAP   = 108_000
INIT_CASH_PCT = 0.45

PARAMS = {
    'hc': 0.06,   # HIGH 기준 Eval ≥ +6%
    'lc': -0.06,  # LOW  기준 Eval ≤ -6%
    'sH': 2.0, 'sM': 0.3, 'sL': 0.2,
    'bH': 1.0, 'bM': 0.6, 'bL': 2.0,
}

# ── 실행 타이밍 가드 ─────────────────────────────────────
# 정상 실행 윈도우 (모두 미국 동부 기준):
#   ① 금요일 04:00 ~ 09:30  (pre-market: 어제 종가 기준, 오늘 LOC 주문 준비)
#   ② 금요일 15:00 ~ 16:05  (near-close: 거의 마감가로 즉시 LOC 주문, KST 토 04:00~05:05)
#   ③ 금요일 16:05 ~ 23:59  (post-close: 마감가로 다음주 신호 확정)
#   ④ 토요일 00:00 ~ 06:00  (post-close 한국시간 새벽 커버)
STRICT_TIMING = True

# Telegram
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID   = os.environ.get('CHAT_ID', '')

print(f"🔍 ENV CHECK: BOT_TOKEN={'있음(' + str(len(BOT_TOKEN)) + '자)' if BOT_TOKEN else '없음'}, "
      f"CHAT_ID={'있음' if CHAT_ID else '없음'}")


# ─────────────────────────────────────────────────────────────
# 0. 실행 타이밍 가드 (pre-market & post-close 모두 허용)
# ─────────────────────────────────────────────────────────────
def check_execution_timing() -> tuple[bool, str, str]:
    """
    Returns: (정상여부, 메시지, mode)
      mode ∈ {'pre_open', 'near_close', 'post_close', 'invalid'}
    """
    et = datetime.now(ZoneInfo("America/New_York"))
    weekday = et.weekday()
    hhmm = et.hour * 60 + et.minute

    # ① 금요일 pre-market
    if weekday == 4 and 4 * 60 <= hhmm < 9 * 60 + 30:
        return True, (
            f"✅ Pre-market 모드: ET {et:%Y-%m-%d %H:%M} (금요일 장 시작 전)\n"
            f"   목요일 종가 기준 신호 → 오늘 LOC 주문 준비"
        ), 'pre_open'

    # ② 금요일 near-close (마감 직전 1시간 5분)
    if weekday == 4 and 15 * 60 <= hhmm < 16 * 60 + 5:
        mins_to_close = max(0, 16 * 60 - hhmm)
        return True, (
            f"✅ Near-close 모드: ET {et:%Y-%m-%d %H:%M} (마감 {mins_to_close}분 전)\n"
            f"   장중 실시간 가격 ≈ 마감가, 즉시 LOC 주문 권장"
        ), 'near_close'

    # ③ 금요일 post-close
    if weekday == 4 and hhmm >= 16 * 60 + 5:
        return True, (
            f"✅ Post-close 모드: ET {et:%Y-%m-%d %H:%M} (금요일 마감 후)\n"
            f"   금요일 종가 기준 신호 확정"
        ), 'post_close'

    # ④ 토요일 새벽 (post-close 연장)
    if weekday == 5 and hhmm < 6 * 60:
        return True, (
            f"✅ Post-close 모드: ET {et:%Y-%m-%d %H:%M} (토요일 새벽)"
        ), 'post_close'

    return False, (
        f"⚠️ 비정상 타이밍: ET {et:%Y-%m-%d %H:%M} (요일={weekday}). "
        f"정상 실행 시간:\n"
        f"  • 금요일 ET 04:00~09:30 (pre-market, KST 17:00~22:30)\n"
        f"  • 금요일 ET 15:00~16:05 (near-close, KST 토 04:00~05:05)\n"
        f"  • 금요일 ET 16:05~23:59 (post-close, KST 토 05:05~12:59)\n"
        f"  • 토요일 ET 00:00~06:00 (post-close 연장)"
    ), 'invalid'


# ─────────────────────────────────────────────────────────────
# 1. Expanding Window OLS
# ─────────────────────────────────────────────────────────────
def compute_expanding_ols(qqq_weekly: pd.DataFrame, W: int = 260) -> np.ndarray:
    """주간 QQQ 시계열로 5년(260주) Expanding Window log-선형 회귀를 계산합니다.
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
            growth[i] = float(qqq_weekly['QQQ'].iloc[i]); continue
        s_t  = ps_t[end]  - ps_t[start]
        s_y  = ps_y[end]  - ps_y[start]
        s_t2 = ps_t2[end] - ps_t2[start]
        s_ty = ps_ty[end] - ps_ty[start]
        denom = w * s_t2 - s_t ** 2
        if denom == 0:
            growth[i] = float(qqq_weekly['QQQ'].iloc[i]); continue
        b = (w * s_ty - s_t * s_y) / denom
        a = (s_y - b * s_t) / w
        growth[i] = np.exp(a + b * t[i])
    return growth


# ─────────────────────────────────────────────────────────────
# 2. Telegram
# ─────────────────────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    """텔레그램 메시지 발송. 성공/실패를 명시적으로 로그.
    Returns: 발송 성공 여부 (True/False)
    """
    if not BOT_TOKEN:
        print("🔴 Telegram 발송 실패: BOT_TOKEN 환경변수 없음"); return False
    if not CHAT_ID:
        print("🔴 Telegram 발송 실패: CHAT_ID 환경변수 없음"); return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"
        }, timeout=10)
        # 텔레그램 API 응답 상태 검증
        if r.status_code != 200:
            print(f"🔴 Telegram API 에러 {r.status_code}: {r.text[:300]}")
            # Markdown 파싱 에러일 가능성 → plain text 로 재시도
            if r.status_code == 400 and 'parse' in r.text.lower():
                print("   ↻ Markdown 제거하고 plain text 로 재시도...")
                r2 = requests.post(url, json={
                    "chat_id": CHAT_ID, "text": text
                }, timeout=10)
                if r2.status_code == 200:
                    print("   ✅ plain text 발송 성공"); return True
                else:
                    print(f"   🔴 재시도도 실패: {r2.status_code} {r2.text[:200]}")
            return False
        try:
            j = r.json()
            if not j.get('ok', False):
                print(f"🔴 Telegram API ok=false: {j.get('description','')}")
                return False
        except Exception:
            pass
        print(f"✅ Telegram 발송 OK ({len(text)} chars)")
        return True
    except requests.Timeout:
        print("🔴 Telegram 발송 실패: 타임아웃 (10초)"); return False
    except Exception as e:
        print(f"🔴 Telegram 발송 실패 (예외): {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# 3. Sheets 설정 로드
# ─────────────────────────────────────────────────────────────
def load_config_from_sheets(sh) -> dict:
    try:
        ws = sh.worksheet("설정")
        records = ws.get_all_records()
        return {row['key']: row['value'] for row in records if row.get('key')}
    except Exception as e:
        print(f"⚠️ 설정 시트 로드 실패 (기본값 사용): {e}")
        return {}


def apply_sheets_config(cfg, start_date, initial_cap, init_cash_pct, params):
    """Sheets에서 읽은 설정을 실제 변수에 반영합니다.
    use_volF 키는 v4.3 에서 무시 — F-가드 코드 자체가 제거됨.
    """
    if not cfg:
        return start_date, initial_cap, init_cash_pct, params

    def _f(key, default):
        try: return float(cfg[key])
        except (KeyError, ValueError, TypeError): return default

    new_start    = str(cfg.get('start_date', start_date))[:10]
    new_cap      = _f('cap',  initial_cap)
    new_cash_pct = _f('cash', init_cash_pct * 100) / 100
    new_params = {
        'hc': _f('hc', params['hc'] * 100) / 100,
        'lc': _f('lc', params['lc'] * 100) / 100,
        'sH': _f('sH', params['sH']), 'sM': _f('sM', params['sM']),
        'sL': _f('sL', params['sL']),
        'bH': _f('bH', params['bH']), 'bM': _f('bM', params['bM']),
        'bL': _f('bL', params['bL']),
    }
    print(f"📋 Sheets 설정 로드 완료")
    print(f"   시작일: {new_start} | 원금: ${new_cap:,.0f} | 초기현금: {new_cash_pct:.0%}")
    print(f"   hc={new_params['hc']:.0%} / lc={new_params['lc']:.0%}")
    print(f"   매도 H/M/L: {new_params['sH']}/{new_params['sM']}/{new_params['sL']}")
    print(f"   매수 H/M/L: {new_params['bH']}/{new_params['bM']}/{new_params['bL']}")

    # use_volF 키가 시트에 있으면 안내만 출력 (실제로는 무시됨)
    if 'use_volF' in cfg:
        print(f"   ℹ️ use_volF={cfg['use_volF']} 시트에 있으나 v4.3 봇은 F-가드 미적용")

    return new_start, new_cap, new_cash_pct, new_params


# ─────────────────────────────────────────────────────────────
# 3-b. Reconciliation
# ─────────────────────────────────────────────────────────────
def load_actual_balance(sh) -> dict:
    """
    '실잔고' 시트에서 가장 최근 행을 읽어 실제 증권사 잔고를 반환합니다.
    시트 형식 (header):  date | shares | cash | total
    """
    try:
        ws = sh.worksheet("실잔고")
        records = ws.get_all_records()
        if not records: return {}
        latest = records[-1]
        return {
            'date':   str(latest.get('date', '')),
            'shares': int(float(latest.get('shares', 0))),
            'cash':   float(latest.get('cash', 0)),
            'total':  float(latest.get('total', 0)),
        }
    except Exception as e:
        print(f"ℹ️ '실잔고' 시트 없음 — reconciliation 스킵: {e}")
        return {}


def reconcile(virtual_shares, virtual_cash, actual, init_cap):
    if not actual: return ""
    drift_shares = virtual_shares - actual['shares']
    drift_cash   = virtual_cash   - actual['cash']
    drift_total  = drift_cash + drift_shares
    pct_total    = drift_total / actual['total'] * 100 if actual['total'] > 0 else 0
    flag = "✅" if abs(pct_total) < 1 else ("⚠️" if abs(pct_total) < 5 else "🔴")
    return (
        f"\n{flag} *Reconciliation* ({actual['date']} 기준)\n"
        f"  주식 차이 : `{drift_shares:+,d}` 주\n"
        f"  현금 차이 : `${drift_cash:+,.2f}`\n"
        f"  총자산 편차: `{pct_total:+.2f}%`\n"
        f"  → 1% 이내 정상, 5% 초과 시 동기화 점검\n"
    )


# ─────────────────────────────────────────────────────────────
# 4. 티어 판정
# ─────────────────────────────────────────────────────────────
def get_tier(eval_val, hc, lc):
    if eval_val >= hc: return 'HIGH'
    if eval_val <= lc: return 'LOW'
    return 'MID'


# ─────────────────────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("🚀 [위대리] 오토봇 v4.3 가동 시작...")

    # 0. 타이밍 가드
    ok_timing, timing_msg, mode = check_execution_timing()
    print(timing_msg)
    if not ok_timing:
        send_telegram(timing_msg)
        if STRICT_TIMING:
            print("❌ STRICT_TIMING=True → 봇 중단 (False 로 바꾸면 강제 진행)")
            return

    # A. GCP 인증
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw:
        err = "❌ [위대리 봇] GCP_CREDENTIALS 환경변수 없음. 봇 종료."
        print(err); send_telegram(err); return
    try:
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
    except Exception as e:
        err_str = str(e)
        # 자주 발생하는 케이스별 친절한 진단 메시지
        if 'account not found' in err_str.lower():
            diagnosis = ("서비스 계정이 GCP 에 존재하지 않습니다. "
                         "앱과 봇이 *서로 다른* 서비스 계정을 쓰고 있을 가능성이 높습니다. "
                         "GCP Console 에서 살아있는 계정의 JSON 키를 봇 환경변수 GCP_CREDENTIALS 에 다시 넣어주세요.")
        elif 'invalid jwt signature' in err_str.lower() or 'invalid_grant' in err_str.lower():
            diagnosis = ("서비스 계정 키가 무효합니다. private_key 형식 손상 또는 키 회전 후 옛 키 사용 중. "
                         "새 JSON 키를 발급받아 GCP_CREDENTIALS 에 다시 넣어주세요.")
        elif 'permission' in err_str.lower() or '403' in err_str:
            diagnosis = ("권한 부족. 서비스 계정 이메일을 시트 공유 대상에 *편집자* 로 추가했는지 확인하세요.")
        elif 'not found' in err_str.lower() and '404' in err_str:
            diagnosis = (f"시트 키({SHEET_KEY[:20]}...) 또는 워크시트 이름이 잘못됐습니다.")
        else:
            diagnosis = "위 에러 메시지를 확인하세요."
        err = f"❌ [위대리 봇] 시트 연결 실패\n\n에러: `{err_str[:300]}`\n\n진단: {diagnosis}"
        print(err); send_telegram(err); return

    # A-2. 설정 로드
    _cfg = load_config_from_sheets(sh)
    start_date, initial_cap, init_cash_pct, params = apply_sheets_config(
        _cfg, START_DATE, INITIAL_CAP, INIT_CASH_PCT, dict(PARAMS),
    )
    actual_balance = load_actual_balance(sh)

    try:
        # B. 일별 데이터
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

        # C. 라이브 가격 — mode 별 분기
        live_tqqq = float(yf.Ticker("TQQQ").fast_info['last_price'])
        live_qqq  = float(yf.Ticker("QQQ").fast_info['last_price'])

        if mode == 'pre_open':
            # pre-market: live 값은 어제(목요일) 종가 — df 가 이미 그걸 가지고 있음
            # cur_p 는 df 의 마지막 종가(목요일).
            cur_p = round(float(df['TQQQ'].iloc[-1]), 2)
        else:  # near_close 또는 post_close — 둘 다 오늘 가격을 사용
            # near_close: live_tqqq 는 장중 실시간 가격 (마감가에 매우 근접)
            # post_close: live_tqqq 는 정식 마감가
            live_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            if live_date > df['Date'].iloc[-1]:
                df = pd.concat([df, pd.DataFrame({
                    'Date': [live_date], 'TQQQ': [live_tqqq], 'QQQ': [live_qqq]
                })], ignore_index=True)
            else:
                df.loc[df.index[-1], 'TQQQ'] = live_tqqq
                df.loc[df.index[-1], 'QQQ']  = live_qqq
            cur_p = round(float(df['TQQQ'].iloc[-1]), 2)

        # D. 주간 데이터
        df_idx = df.set_index('Date')
        qqq_weekly  = df_idx[['QQQ']].resample('W-FRI').last().dropna().reset_index()
        tqqq_weekly = df_idx[['TQQQ']].resample('W-FRI').last().dropna().reset_index()

        # E. OLS → Eval
        growth = compute_expanding_ols(qqq_weekly, W=260)
        qqq_weekly['Growth'] = growth
        qqq_weekly['Eval']   = qqq_weekly['QQQ'] / qqq_weekly['Growth'] - 1
        weekly = qqq_weekly.merge(tqqq_weekly, on='Date', how='inner')

        # F. 시작일 필터
        sim = (weekly[weekly['Date'] >= pd.to_datetime(start_date)]
               .dropna(subset=['Eval', 'TQQQ'])
               .reset_index(drop=True))
        if len(sim) < 2:
            msg = "⚠️ [위대리] 시뮬레이션 데이터 부족 (시작일 이후 데이터 < 2주)"
            print(msg); send_telegram(msg); return

        # G. 초기 잔고
        init_stock_ratio = 1.0 - init_cash_pct
        init_price       = float(sim.loc[0, 'TQQQ'])
        shares           = int((initial_cap * init_stock_ratio) / init_price)
        cash             = initial_cap - (shares * init_price)

        hc, lc = params['hc'], params['lc']
        sell_r = {'HIGH': params['sH'], 'MID': params['sM'], 'LOW': params['sL']}
        buy_r  = {'HIGH': params['bH'], 'MID': params['bM'], 'LOW': params['bL']}

        # H. 과거 매매 복기 (시작일+1 ~ 마지막 주 직전)
        for i in range(1, len(sim) - 1):
            p, prev_p, ev = float(sim.loc[i,'TQQQ']), float(sim.loc[i-1,'TQQQ']), float(sim.loc[i,'Eval'])
            tier = get_tier(ev, hc, lc)
            diff = shares * (p - prev_p)

            if diff > 0 and shares > 0:
                qty = int(min(round(diff * sell_r[tier] / p), shares))
                shares -= qty; cash += qty * p
            elif diff < 0:
                qty = int(min(cash, abs(diff) * buy_r[tier]) / p)
                shares += qty; cash -= qty * p

        # I. 이번 주 신호
        last  = sim.iloc[-1]; prev = sim.iloc[-2]
        last_eval  = float(last['Eval'])
        last_price = float(last['TQQQ'])
        prev_price = float(prev['TQQQ'])
        this_tier  = get_tier(last_eval, hc, lc)
        diff_now   = shares * (last_price - prev_price)

        action, order_qty = "관망", 0
        if diff_now > 0 and shares > 0:
            action    = "매도"
            order_qty = int(min(round(diff_now * sell_r[this_tier] / last_price), shares))
        elif diff_now < 0:
            action    = "매수"
            order_qty = int(min(cash, abs(diff_now) * buy_r[this_tier]) / last_price)
        if order_qty == 0:
            action = "관망"

        # J. 자산
        if action == "매도":
            est_shares, est_cash = shares - order_qty, cash + order_qty * cur_p
        elif action == "매수":
            est_shares, est_cash = shares + order_qty, cash - order_qty * cur_p
        else:
            est_shares, est_cash = shares, cash

        total_asset = est_cash + est_shares * cur_p
        # ✅ Sheets 의 initial_cap 사용 (v4.1 버그 수정)
        pnl_pct  = (total_asset / initial_cap - 1) * 100
        cash_pct = est_cash / total_asset * 100 if total_asset > 0 else 0
        recon_msg = reconcile(est_shares, est_cash, actual_balance, initial_cap)

        # K. Sheets 업데이트
        ws.update_acell('L4', action)
        ws.update_acell('M4', 'LOC' if action != "관망" else "-")
        ws.update_acell('N4', cur_p)
        ws.update_acell('O4', order_qty)
        print(f"📤 시트 업데이트: {action} {order_qty}주 @ ${cur_p}")

        # L. Telegram
        tier_emoji   = {'HIGH':'🟡','MID':'🔵','LOW':'🟢'}[this_tier]
        action_emoji = {'매도':'📈','매수':'📉','관망':'⏸'}[action]
        mode_label   = {'pre_open':'Pre-market (어제 종가 기준, 오늘 LOC 준비)',
                        'near_close':'Near-close (마감 직전, 즉시 LOC 권장)',
                        'post_close':'Post-close (마감가 확정)'}.get(mode, '')
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        msg = (
            f"🚀 *[위대리 v4.3] 주간 시그널* ({today_str})\n"
            f"📡 모드: {mode_label}\n\n"
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
            f"  📊 거래 후 잔고: *{est_shares:,}* 주\n"
            f"{recon_msg}\n"
        )
        if action != "관망":
            urgency = ("⚡ *지금 즉시* MTS 에서 LOC 주문 입력 (마감까지 시간 적음)\n"
                       if mode == 'near_close' else "")
            msg += (
                f"📱 *MTS 실행 방법*\n"
                f"  {urgency}"
                f"  → `{action} {order_qty}주` LOC로 예약\n"
                f"  → 금요일 장 마감 직전 체결\n"
            )
        else:
            msg += "  → 이번 주 주문 없음 (관망)\n"

        ok_send = send_telegram(msg)
        if ok_send:
            print("✅ 모든 작업 완료 (Sheets + Telegram)")
        else:
            print("⚠️ Sheets 는 업데이트됐지만 Telegram 발송 실패")
            print("   → 위 로그에서 원인 확인 + BOT_TOKEN/CHAT_ID 환경변수 점검 필요")

    except Exception as e:
        err_msg = f"❌ [위대리 봇 v4.3] 오류 발생: {e}"
        print(err_msg)
        send_telegram(err_msg)


if __name__ == "__main__":
    main()
