# wedaeri_h_bot.py — TQQQ 위대리-H 오토봇 (vH1.0)
# ▸ 주간 Expanding Window OLS (wedaeri_h_app.py 동일 로직)
# ▸ W-FRI resample 기반 — 금요일 휴장 주도 정확 처리
# ▸ 3-티어 시스템 (HIGH / MID / LOW)
# ▸ Google Sheets + Telegram 자동 업데이트
#
# ══════════════════════════════════════════════════════════════════════════
#  위대리-H (Hedge-hardened) — 원작자 위대리 v1.0 + 노출 하한 15%
#  ※ 이 봇이 *기준 엔진* 이며 앱(wedaeri_h_app.py)이 이 주간 패널 구성에 맞춰 동작.
# ──────────────────────────────────────────────────────────────────────────
#  vH1.0 변경사항 (vs v5.1):
#   ① 매도/매수를 '상승·하락분의 %' 방식으로 (원작자 원문, 251211 수정판)
#        매도율:  HIGH 100% / MID 50% / LOW 33.3%    (sH/sM/sL = 1.0/0.5/0.333)
#        매수율:  HIGH 50%  / MID 50% / LOW 100%     (bH/bM/bL = 0.5/0.5/1.0)
#      ※ 기존 v5.1은 diff×배율(sH=2.0 등)이라 원작자보다 과공격적 → 경화 심화.
#   ② HIGH 컷 +5% (hc=0.05), LOW 컷 -6% (lc=-0.06)
#   ③ ★ 노출 하한 15% (FLOOR_EXPO) — 매도 후 노출이 15% 밑이면 매도 취소.
#      경화(현금 갇힘)를 원천 차단. 시작 시점 무관하게 노출 15% 유지.
#   ④ v4.8 신호(12월 이연·멜트업 필터) 전면 제거.
#      검증 결과 위대리-H 위에서는 세후 Calmar를 오히려 낮춤(0.519→0.503).
#      -H 구조가 이미 과매도를 막고 있어 중복·역효과. 원작자 철학(단순)과도 합치.
#

# [운용] 단독이 아니라 동파공의 헤지 슬리브. 동파공70/위대리-H30 권장(세후 Calmar 1.12).
#   ※ 외부 유입 현금은 즉시 전량 매수되지 않는다. 매수 예산이 '하락분×비율'이라
#     유입금은 하락 주간마다 분할로만 시장에 들어간다(목돈 고점 진입 방지).

#  [유지] 실행 타이밍 가드 / reconciliation / 양도세 차감 / 캐시 폴백 구조 동일.

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import time
import requests
import warnings
from pathlib import Path
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

# ★ 위대리-H 파라미터 (원작자 v1.0 % 방식)
#   sH/sM/sL : 상승분에 곱하는 매도 비율 (최대 1.0 = 100%)
#   bH/bM/bL : 하락분에 곱하는 매수 비율 (최대 1.0 = 100%)
PARAMS = {
    'hc': 0.05,   # HIGH 기준 Eval >= +5%  (원작자)
    'lc': -0.06,  # LOW  기준 Eval <= -6%
    'sH': 1.000, 'sM': 0.500, 'sL': 0.333,   # 매도 100/50/33.3 %
    'bH': 0.500, 'bM': 0.500, 'bL': 1.000,   # 매수 50/50/100 %
}

# ★ 위대리-H 핵심: 노출 하한. 매도 후 노출이 이 값 밑이면 매도 취소.
FLOOR_EXPO = 0.15

# Telegram
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID   = os.environ.get('CHAT_ID', '')

print(f"🔍 ENV CHECK: BOT_TOKEN={'있음(' + str(len(BOT_TOKEN)) + '자)' if BOT_TOKEN else '없음'}, "
      f"CHAT_ID={'있음' if CHAT_ID else '없음'}")


# ─────────────────────────────────────────────────────────────
# 0. 실행 타이밍 가드 (pre-market & post-close 모두 허용)
# ─────────────────────────────────────────────────────────────
def check_execution_timing() -> tuple:
    """Returns: (실거래여부, 메시지, mode)
      mode in {'pre_open','near_close','post_close','test'}
      실거래여부=False 면 'test' — 지난 종가 기준 시뮬, Sheets 미업데이트."""
    et = datetime.now(ZoneInfo("America/New_York"))
    weekday = et.weekday(); hhmm = et.hour * 60 + et.minute
    if weekday == 4 and 4 * 60 <= hhmm < 9 * 60 + 30:
        return True, (f"✅ Pre-market 모드: ET {et:%Y-%m-%d %H:%M} (금요일 장 시작 전)\n"
                      f"   목요일 종가 기준 신호 → 오늘 LOC 주문 준비"), 'pre_open'
    if weekday == 4 and 15 * 60 <= hhmm < 16 * 60 + 5:
        mins_to_close = max(0, 16 * 60 - hhmm)
        return True, (f"✅ Near-close 모드: ET {et:%Y-%m-%d %H:%M} (마감 {mins_to_close}분 전)\n"
                      f"   장중 실시간 가격 ≈ 마감가, 즉시 LOC 주문 권장"), 'near_close'
    if weekday == 4 and hhmm >= 16 * 60 + 5:
        return True, (f"✅ Post-close 모드: ET {et:%Y-%m-%d %H:%M} (금요일 마감 후)\n"
                      f"   금요일 종가 기준 신호 확정"), 'post_close'
    if weekday == 5 and hhmm < 6 * 60:
        return True, (f"✅ Post-close 모드: ET {et:%Y-%m-%d %H:%M} (토요일 새벽)"), 'post_close'
    return False, (f"🧪 테스트 모드: ET {et:%Y-%m-%d %H:%M} (요일={weekday})\n"
                   f"   정규 실행 윈도우 밖 — 지난 종가 기준으로 신호만 계산\n"
                   f"   ⚠️ Google Sheets 주문표 업데이트는 건너뜁니다 (자동매매 보호)"), 'test'


# ─────────────────────────────────────────────────────────────
# 1. Expanding Window OLS (앱과 동일)
# ─────────────────────────────────────────────────────────────
def compute_expanding_ols(qqq_weekly: pd.DataFrame, W: int = 260) -> np.ndarray:
    n = len(qqq_weekly)
    t = np.arange(1, n + 1, dtype=float)
    y = np.log(qqq_weekly['QQQ'].values.astype(float))
    ps_t  = np.zeros(n + 1); ps_t[1:]  = np.cumsum(t)
    ps_y  = np.zeros(n + 1); ps_y[1:]  = np.cumsum(y)
    ps_t2 = np.zeros(n + 1); ps_t2[1:] = np.cumsum(t ** 2)
    ps_ty = np.zeros(n + 1); ps_ty[1:] = np.cumsum(t * y)
    growth = np.empty(n)
    for i in range(n):
        w = min(i + 1, W); end = i + 1; start = end - w
        if w == 1:
            growth[i] = float(qqq_weekly['QQQ'].iloc[i]); continue
        s_t  = ps_t[end]  - ps_t[start]; s_y  = ps_y[end]  - ps_y[start]
        s_t2 = ps_t2[end] - ps_t2[start]; s_ty = ps_ty[end] - ps_ty[start]
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
    if not BOT_TOKEN:
        print("🔴 Telegram 발송 실패: BOT_TOKEN 환경변수 없음"); return False
    if not CHAT_ID:
        print("🔴 Telegram 발송 실패: CHAT_ID 환경변수 없음"); return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        if r.status_code != 200:
            print(f"🔴 Telegram API 에러 {r.status_code}: {r.text[:300]}")
            if r.status_code == 400 and 'parse' in r.text.lower():
                print("   ↻ Markdown 제거하고 plain text 로 재시도...")
                r2 = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
                if r2.status_code == 200:
                    print("   ✅ plain text 발송 성공"); return True
                print(f"   🔴 재시도도 실패: {r2.status_code} {r2.text[:200]}")
            return False
        try:
            j = r.json()
            if not j.get('ok', False):
                print(f"🔴 Telegram API ok=false: {j.get('description','')}"); return False
        except Exception:
            pass
        print(f"✅ Telegram 발송 OK ({len(text)} chars)"); return True
    except requests.Timeout:
        print("🔴 Telegram 발송 실패: 타임아웃 (10초)"); return False
    except Exception as e:
        print(f"🔴 Telegram 발송 실패 (예외): {type(e).__name__}: {e}"); return False


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
    """Sheets 설정 반영. Returns: (start_date, cap, cash_pct, params, floor, tax_apply)."""
    def _f(key, default):
        try: return float(cfg[key])
        except (KeyError, ValueError, TypeError): return default
    def _b(key, default):
        raw = cfg.get(key) if cfg else None
        if raw is None: return default
        s = str(raw).strip().lower()
        if s in ('true', '1', 'yes', 'on'): return True
        if s in ('false', '0', 'no', 'off', ''): return False
        return default
    if not cfg:
        return start_date, initial_cap, init_cash_pct, params, FLOOR_EXPO, False
    new_start    = str(cfg.get('start_date', start_date))[:10]
    new_cap      = _f('cap',  initial_cap)
    new_cash_pct = _f('cash', init_cash_pct * 100) / 100
    new_params = {
        'hc': _f('hc', params['hc'] * 100) / 100,
        'lc': _f('lc', params['lc'] * 100) / 100,
        'sH': _f('sH', params['sH']), 'sM': _f('sM', params['sM']), 'sL': _f('sL', params['sL']),
        'bH': _f('bH', params['bH']), 'bM': _f('bM', params['bM']), 'bL': _f('bL', params['bL']),
    }
    # floor: 시트에 'floor'(%) 있으면 사용, 없으면 기본 15%
    new_floor = _f('floor', FLOOR_EXPO * 100) / 100
    tax_apply = _b('tax_apply_to_bot', False)
    print(f"📋 Sheets 설정 로드 완료")
    print(f"   시작일: {new_start} | 원금: ${new_cap:,.0f} | 초기현금: {new_cash_pct:.0%}")
    print(f"   hc={new_params['hc']:.0%} / lc={new_params['lc']:.0%} / 노출하한={new_floor:.0%}")
    print(f"   매도 H/M/L: {new_params['sH']}/{new_params['sM']}/{new_params['sL']} (상승분의 %)")
    print(f"   매수 H/M/L: {new_params['bH']}/{new_params['bM']}/{new_params['bL']} (하락분의 %)")
    print(f"   양도세 봇 차감: {'ON' if tax_apply else 'OFF (기본)'}")
    return new_start, new_cap, new_cash_pct, new_params, new_floor, tax_apply


# ─────────────────────────────────────────────────────────────
# 3-b. Reconciliation & 양도세 납부 로드
# ─────────────────────────────────────────────────────────────
def load_actual_balance(sh) -> dict:
    try:
        ws = sh.worksheet("실잔고")
        records = ws.get_all_records()
        if not records: return {}
        latest = records[-1]
        return {'date': str(latest.get('date', '')), 'shares': int(float(latest.get('shares', 0))),
                'cash': float(latest.get('cash', 0)), 'total': float(latest.get('total', 0))}
    except Exception as e:
        print(f"ℹ️ '실잔고' 시트 없음 — reconciliation 스킵: {e}")
        return {}


def load_tax_payments(sh) -> list:
    try:
        ws = sh.worksheet("양도세납부")
        records = ws.get_all_records()
        out = []
        for r in records:
            try:
                out.append({'date': str(r.get('date', '')), 'for_year': int(float(r.get('for_year', 0))),
                            'amount_usd': float(r.get('amount_usd', 0)), 'amount_krw': float(r.get('amount_krw', 0)),
                            'fx_rate': float(r.get('fx_rate', 1300)), 'note': str(r.get('note', ''))})
            except Exception:
                continue
        out.sort(key=lambda x: x['date'])
        return out
    except Exception as e:
        print(f"ℹ️ '양도세납부' 시트 없음 — 납부 차감 스킵: {e}")
        return []


def load_cash_adjustments(sh) -> list:
    """'현금조정' 탭: date | amount | note.
    리밸런싱 등 예수금만 조정(주식수 불변). amount 양수=입금, 음수=출금.
    앱(wedaeri_h_app.py)과 동일 규칙."""
    try:
        ws = sh.worksheet("현금조정")
        records = ws.get_all_records()
        out = []
        for r in records:
            try:
                amt_raw = str(r.get('amount', '')).replace('$', '').replace(',', '').strip()
                if amt_raw == '':
                    continue
                out.append({'date': str(r.get('date', '')), 'amount': float(amt_raw),
                            'note': str(r.get('note', ''))})
            except Exception:
                continue
        out.sort(key=lambda x: x['date'])
        return out
    except Exception as e:
        print(f"ℹ️ '현금조정' 시트 없음 — 조정 스킵: {e}")
        return []


def reconcile(virtual_shares, virtual_cash, actual, init_cap):
    if not actual: return ""
    drift_shares = virtual_shares - actual['shares']
    drift_cash   = virtual_cash   - actual['cash']
    drift_total  = drift_cash + drift_shares
    pct_total    = drift_total / actual['total'] * 100 if actual['total'] > 0 else 0
    flag = "✅" if abs(pct_total) < 1 else ("⚠️" if abs(pct_total) < 5 else "🔴")
    return (f"\n{flag} *Reconciliation* ({actual['date']} 기준)\n"
            f"  주식 차이 : `{drift_shares:+,d}` 주\n"
            f"  현금 차이 : `${drift_cash:+,.2f}`\n"
            f"  총자산 편차: `{pct_total:+.2f}%`\n"
            f"  → 1% 이내 정상, 5% 초과 시 동기화 점검\n")


# ─────────────────────────────────────────────────────────────
# 4. 티어 판정 & ★위대리-H 매도량 (노출 하한 적용)
# ─────────────────────────────────────────────────────────────
def get_tier(eval_val, hc, lc):
    if eval_val >= hc: return 'HIGH'
    if eval_val <= lc: return 'LOW'
    return 'MID'


def sell_qty_with_floor(diff, price, shares, cash, sell_rate, floor):
    """상승분(diff)의 sell_rate% 만큼 매도하되, 매도 후 노출이 floor 밑이면 매도량 축소.
    Returns: 매도 수량(int)."""
    qty = int(min(round(diff * sell_rate / price), shares))
    if floor > 0:
        total = cash + shares * price
        if total > 0:
            min_sh = int(floor * total / price)   # 유지해야 할 최소 주식수
            qty = max(0, min(qty, shares - min_sh))
    return qty


def buy_qty(diff, price, cash, buy_rate):
    """하락분(diff)의 buy_rate% 만큼 매수 (현금 한도 내). Returns: 매수 수량(int)."""
    if cash <= 0: return 0
    return int(min(cash, abs(diff) * buy_rate) / price)


# ─────────────────────────────────────────────────────────────
# 4-c. yfinance 안전 fetch (재시도 + 캐시 fallback)
# ─────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "wedaeri_h_yf_cache.parquet"

def _save_cache(df: pd.DataFrame):
    try:
        df.to_parquet(CACHE_FILE, index=False)
    except Exception as e:
        try:
            df.to_csv(CACHE_FILE.with_suffix('.csv'), index=False)
        except Exception:
            print(f"⚠️ 캐시 저장 실패: {e}")

def _load_cache() -> tuple:
    for path in (CACHE_FILE, CACHE_FILE.with_suffix('.csv')):
        if not path.exists():
            continue
        try:
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
            return df, age_h
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패 ({path.name}): {e}")
    return None, None

def fetch_yf_with_retry(tickers: list, start: str, end: str, max_retries: int = 4) -> tuple:
    last_err = None; last_rows = -1
    for attempt in range(1, max_retries + 1):
        try:
            df_raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df = df_raw['Close'].copy()
            else:
                df = df_raw.copy()
            df = df.dropna().reset_index()
            if 'Date' not in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            last_rows = len(df)
            missing = [t for t in tickers if t not in df.columns]
            if not df.empty and not missing:
                return df, f"✅ yfinance fetch 성공 (시도 {attempt}회, {len(df)}행)"
            last_err = f"빈 응답 또는 누락 컬럼 (rows={len(df)}, columns={list(df.columns)}, missing={missing})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        wait_s = min(2 ** attempt, 16)
        print(f"🔄 yfinance 시도 {attempt}/{max_retries} 실패 — {last_err}. {wait_s}초 후 재시도...")
        time.sleep(wait_s)
    return None, f"⚠️ {max_retries}회 모두 실패. 마지막 에러: {last_err} (last_rows={last_rows})"


def fetch_live_price_safe(ticker: str):
    try:
        return float(yf.Ticker(ticker).fast_info['last_price'])
    except Exception as e:
        print(f"⚠️ {ticker} live price fetch 실패: {e}")
        try:
            d = yf.download(ticker, period='5d', interval='1d', auto_adjust=True, progress=False, threads=False)
            if not d.empty:
                col = 'Close' if 'Close' in d.columns else d.columns[0]
                return float(d[col].dropna().iloc[-1])
        except Exception as e2:
            print(f"⚠️ {ticker} 5d 폴백도 실패: {e2}")
        return None


# ─────────────────────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("🚀 [위대리-H] 오토봇 vH1.0 가동 시작...")
    print(f"   매도 100/50/33.3% · 매수 50/50/100% · HIGH컷+5% · 노출하한 {FLOOR_EXPO:.0%}")
    print("   v4.8 신호(12월이연·멜트업) 제거됨 — 검증상 위대리-H 세후 Calmar 저하로 확인.")

    is_live, timing_msg, mode = check_execution_timing()
    print(timing_msg)
    is_test_mode = (mode == 'test')

    # A. GCP 인증
    creds_raw = os.environ.get('GCP_CREDENTIALS')
    if not creds_raw:
        err = "❌ [위대리-H 봇] GCP_CREDENTIALS 환경변수 없음. 봇 종료."
        print(err); send_telegram(err); return
    try:
        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        required_keys = ['type', 'client_email', 'private_key', 'token_uri']
        missing = [k for k in required_keys if not creds_dict.get(k)]
        if missing:
            raise ValueError(f"GCP_CREDENTIALS JSON 에 필수 필드 누락: {missing}. 보유 필드: {list(creds_dict.keys())}")
        print(f"📧 봇이 사용 중인 서비스 계정: {creds_dict.get('client_email', 'unknown')[:60]}")
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")
    except Exception as e:
        import traceback
        err_str = str(e) or "(빈 메시지)"; err_repr = repr(e); err_type = type(e).__name__
        tb = traceback.format_exc(); haystack = (err_str + ' ' + err_repr + ' ' + tb).lower()
        if 'account not found' in haystack:
            diagnosis = ("서비스 계정이 GCP 에 존재하지 않습니다. 앱과 봇이 서로 다른 계정을 쓸 가능성. "
                         "Streamlit Secrets 의 client_email 과 봇 client_email 일치 확인 후 새 JSON 키 주입.")
        elif 'invalid jwt signature' in haystack:
            diagnosis = "서비스 계정 키 형식 손상. private_key 개행 처리 확인. JSON 통째로 다시 붙여넣기."
        elif 'invalid_grant' in haystack:
            diagnosis = "자격증명 무효. 키 회전/계정 삭제. 새 JSON 키 필요."
        elif 'permission' in haystack or '403' in haystack:
            diagnosis = "권한 부족. 서비스 계정을 시트 공유에 *편집자* 로 추가했는지 확인."
        elif 'not found' in haystack and ('404' in haystack or 'sheet' in haystack):
            diagnosis = f"시트({SHEET_KEY[:20]}...) 또는 '위대리' 워크시트 접근 불가."
        elif 'jsondecode' in haystack:
            diagnosis = "GCP_CREDENTIALS 가 올바른 JSON 아님. 전체 다시 붙여넣기."
        else:
            diagnosis = "아래 에러 타입/traceback 으로 원인 추정."
        err = (f"❌ [위대리-H 봇] 시트 연결 실패\n\n*예외 타입*: `{err_type}`\n"
               f"*에러*: `{err_str[:200]}`\n*repr*: `{err_repr[:200]}`\n\n*진단*: {diagnosis}")
        print(err); print("\n--- Full traceback ---"); print(tb); send_telegram(err); return

    # A-2. 설정 로드
    _cfg = load_config_from_sheets(sh)
    start_date, initial_cap, init_cash_pct, params, floor, tax_apply_to_bot = apply_sheets_config(
        _cfg, START_DATE, INITIAL_CAP, INIT_CASH_PCT, dict(PARAMS))
    actual_balance = load_actual_balance(sh)
    if tax_apply_to_bot:
        tax_payments = load_tax_payments(sh)
        if tax_payments:
            print(f"🧾 양도세 납부 기록 {len(tax_payments)} 건 로드 — 가상 잔고 차감 적용 (토글 ON)")
            for p in tax_payments:
                print(f"   {p['date']} | {p['for_year']}년 분 | ₩{p['amount_krw']:,.0f} (≈${p['amount_usd']:,.2f}) | {p['note']}")
        else:
            print("ℹ️ 양도세 토글 ON 이지만 시트에 납부 기록 없음")
    else:
        tax_payments = []
        print("ℹ️ 양도세 봇 차감 토글 OFF — 시뮬은 세전 잔고로 진행")

    # 현금 조정(리밸런싱 입출금) 로드 — 항상 적용 (토글과 무관, 주식수 불변)
    cash_adjustments = load_cash_adjustments(sh)
    if cash_adjustments:
        _net_adj = sum(a['amount'] for a in cash_adjustments)
        print(f"💵 현금 조정 {len(cash_adjustments)}건 로드 (순 {_net_adj:+,.2f}) — 예수금만 반영, 주식수 불변")
        for a in cash_adjustments:
            print(f"   {a['date']} | {a['amount']:+,.2f} | {a['note']}")

    try:
        # B. 일별 데이터
        end_dt = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df, fetch_msg = fetch_yf_with_retry(["QQQ", "TQQQ"], "2010-01-01", end_dt)
        print(fetch_msg)
        if df is None or df.empty or 'TQQQ' not in df.columns or 'QQQ' not in df.columns:
            cached_df, cached_age_h = _load_cache()
            if cached_df is not None and not cached_df.empty:
                if cached_age_h <= 168:
                    print(f"💾 캐시 fallback 사용 — {cached_age_h:.1f}시간 전 데이터 ({len(cached_df)}행)")
                    df = cached_df
                    send_telegram(f"⚠️ *[위대리-H 봇] yfinance 데이터 부재 → 캐시 fallback*\n"
                                  f"마지막 정상 데이터: {cached_age_h:.1f}시간 전\n행 수: {len(df)}\n"
                                  f"이번 시그널은 *근사치* — 캐시가 7일 이상 오래되면 봇이 중단됩니다.")
                else:
                    err = f"❌ [위대리-H 봇] yfinance 실패 + 캐시도 {cached_age_h/24:.1f}일 오래됨. 수동 점검 필요.\n진단: {fetch_msg}"
                    print(err); send_telegram(err); return
            else:
                err = (f"❌ [위대리-H 봇] yfinance 데이터 없음 (캐시도 없음).\n진단: {fetch_msg}\n"
                       f"가능 원인: ① rate limit(15~30분 대기) ② Yahoo API 다운 ③ pip install -U yfinance")
                print(err); send_telegram(err); return
        else:
            _save_cache(df)
        if len(df) < 260:
            err = f"❌ [위대리-H 봇] 데이터 부족: {len(df)}행 (5년치 미만). OLS 불가."
            print(err); send_telegram(err); return

        # C. 라이브 가격
        live_tqqq = fetch_live_price_safe("TQQQ"); live_qqq = fetch_live_price_safe("QQQ")
        if live_tqqq is None or live_qqq is None:
            live_tqqq = live_tqqq or float(df['TQQQ'].iloc[-1])
            live_qqq  = live_qqq  or float(df['QQQ'].iloc[-1])
            print(f"⚠️ 라이브 가격 fetch 실패 — df 마지막 종가 사용 (TQQQ=${live_tqqq:.2f}, QQQ=${live_qqq:.2f})")
        if mode in ('pre_open', 'test'):
            cur_p = round(float(df['TQQQ'].iloc[-1]), 2)
        else:
            live_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            if live_date > df['Date'].iloc[-1]:
                df = pd.concat([df, pd.DataFrame({'Date': [live_date], 'TQQQ': [live_tqqq], 'QQQ': [live_qqq]})], ignore_index=True)
            else:
                df.loc[df.index[-1], 'TQQQ'] = live_tqqq; df.loc[df.index[-1], 'QQQ'] = live_qqq
            cur_p = round(float(df['TQQQ'].iloc[-1]), 2)

        # D. 주간 데이터
        df_idx = df.set_index('Date')
        qqq_weekly  = df_idx[['QQQ']].resample('W-FRI').last().dropna().reset_index()
        tqqq_weekly = df_idx[['TQQQ']].resample('W-FRI').last().dropna().reset_index()

        # E. OLS → Eval  (v4.8 QQQ_MA 제거 — 더 이상 사용 안 함)
        growth = compute_expanding_ols(qqq_weekly, W=260)
        qqq_weekly['Growth'] = growth
        qqq_weekly['Eval']   = qqq_weekly['QQQ'] / qqq_weekly['Growth'] - 1
        weekly = qqq_weekly.merge(tqqq_weekly, on='Date', how='inner')

        # F. 시작일 필터
        sim = (weekly[weekly['Date'] >= pd.to_datetime(start_date)]
               .dropna(subset=['Eval', 'TQQQ']).reset_index(drop=True))
        if len(sim) < 2:
            msg = "⚠️ [위대리-H] 시뮬레이션 데이터 부족 (시작일 이후 < 2주)"
            print(msg); send_telegram(msg); return

        # G. 초기 잔고
        init_price = float(sim.loc[0, 'TQQQ'])
        shares     = int((initial_cap * (1.0 - init_cash_pct)) / init_price)
        cash       = initial_cap - (shares * init_price)
        hc, lc = params['hc'], params['lc']
        sell_r = {'HIGH': params['sH'], 'MID': params['sM'], 'LOW': params['sL']}
        buy_r  = {'HIGH': params['bH'], 'MID': params['bM'], 'LOW': params['bL']}

        tax_to_apply = list(tax_payments); cum_paid_tax = 0.0; n_floor_block = 0
        adj_to_apply = list(cash_adjustments); net_injected = 0.0

        # H0. 시작일(첫 주) 이전/당일 조정 반영 (초기 매수 직후 예수금에)
        if adj_to_apply:
            first_bar = pd.Timestamp(sim.loc[0, 'Date']); still = []
            for a in adj_to_apply:
                if pd.Timestamp(a['date']) <= first_bar:
                    cash += a['amount']; net_injected += a['amount']
                else:
                    still.append(a)
            adj_to_apply = still

        # H. 과거 매매 복기 (시작일+1 ~ 마지막 주 직전) — ★위대리-H 규칙
        for i in range(1, len(sim) - 1):
            p, prev_p, ev = float(sim.loc[i, 'TQQQ']), float(sim.loc[i-1, 'TQQQ']), float(sim.loc[i, 'Eval'])
            bar_date = pd.Timestamp(sim.loc[i, 'Date']); prev_bar_date = pd.Timestamp(sim.loc[i-1, 'Date'])
                    # 현금 조정 반영 (매매 전 예수금에 먼저 편입).
        #   단, 매수 예산은 min(현금, 하락분×매수비율)이라 하락분이 상한을 정한다.
        #   → 유입금은 '실탄 후보'일 뿐, 그 주에 전량 매수되지 않고 하락 폭에 좌우된다.
        #     하락이 크지 않으면 현금으로 남아 이후 하락 주간마다 분할 투입된다.

            still_adj = []
            for a in adj_to_apply:
                if prev_bar_date < pd.Timestamp(a['date']) <= bar_date:
                    cash += a['amount']; net_injected += a['amount']
                else:
                    still_adj.append(a)
            adj_to_apply = still_adj
            tier = get_tier(ev, hc, lc); diff = shares * (p - prev_p)
            if diff > 0 and shares > 0:
                qty = sell_qty_with_floor(diff, p, shares, cash, sell_r[tier], floor)
                if qty < int(min(round(diff * sell_r[tier] / p), shares)):
                    n_floor_block += 1
                shares -= qty; cash += qty * p
            elif diff < 0:
                qty = buy_qty(diff, p, cash, buy_r[tier])
                shares += qty; cash -= qty * p
            # 양도세 납부 차감
            still_pending = []
            for pay in tax_to_apply:
                pay_date = pd.Timestamp(pay['date'])
                if prev_bar_date < pay_date <= bar_date:
                    cash -= pay['amount_usd']; cum_paid_tax += pay['amount_usd']
                else:
                    still_pending.append(pay)
            tax_to_apply = still_pending
        # 복기 마지막 주 이후 ~ 현재 사이의 잔여 조정/세금
        last_replay_date = pd.Timestamp(sim.loc[len(sim) - 2, 'Date']); now_ts = pd.Timestamp.now().normalize()
        if adj_to_apply:
            for a in adj_to_apply:
                if last_replay_date < pd.Timestamp(a['date']) <= now_ts:
                    cash += a['amount']; net_injected += a['amount']
        if tax_to_apply:
            for pay in tax_to_apply:
                pay_date = pd.Timestamp(pay['date'])
                if last_replay_date < pay_date <= now_ts:
                    cash -= pay['amount_usd']; cum_paid_tax += pay['amount_usd']
        if abs(net_injected) > 0:
            print(f"💵 누적 현금 조정 반영: {net_injected:+,.2f} (주식수 불변)")
        print(f"💰 누적 양도세 차감: ${cum_paid_tax:,.2f} | 노출하한 발동(복기): {n_floor_block}회")

        # I. 이번 주 신호 — ★위대리-H 규칙 (노출 하한 적용)
        last = sim.iloc[-1]; prev = sim.iloc[-2]
        last_eval = float(last['Eval']); last_price = float(last['TQQQ']); prev_price = float(prev['TQQQ'])
        this_tier = get_tier(last_eval, hc, lc); diff_now = shares * (last_price - prev_price)
        cur_expo = (shares * cur_p) / (cash + shares * cur_p) if (cash + shares * cur_p) > 0 else 0
        floor_note = ""
        action, order_qty = "관망", 0
        if diff_now > 0 and shares > 0:
            raw_qty = int(min(round(diff_now * sell_r[this_tier] / last_price), shares))
            order_qty = sell_qty_with_floor(diff_now, last_price, shares, cash, sell_r[this_tier], floor)
            action = "매도"
            if order_qty < raw_qty:
                floor_note = (f"🛟 노출 하한 {floor:.0%} 발동 — 매도 {raw_qty}→{order_qty}주로 축소 "
                              f"(현재 노출 {cur_expo:.1%}, 하한 아래로는 매도 안 함)")
        elif diff_now < 0:
            action = "매수"; order_qty = buy_qty(diff_now, last_price, cash, buy_r[this_tier])
        if order_qty == 0:
            action = "관망"

        # J. 자산 (주문 체결 후 예상)
        if action == "매도":
            est_shares, est_cash = shares - order_qty, cash + order_qty * cur_p
        elif action == "매수":
            est_shares, est_cash = shares + order_qty, cash - order_qty * cur_p
        else:
            est_shares, est_cash = shares, cash
        total_asset = est_cash + est_shares * cur_p
        _basis = initial_cap + net_injected   # 입출금은 손익이 아니므로 기준에 반영
        pnl_pct  = (total_asset / _basis - 1) * 100 if _basis > 0 else 0
        cash_pct = est_cash / total_asset * 100 if total_asset > 0 else 0
        expo_pct = est_shares * cur_p / total_asset * 100 if total_asset > 0 else 0
        recon_msg = reconcile(est_shares, est_cash, actual_balance, initial_cap)

        # K. Sheets 업데이트 — 테스트 모드면 건너뜀
        if not is_test_mode:
            ws.update_acell('L4', action)
            ws.update_acell('M4', 'LOC' if action != "관망" else "-")
            ws.update_acell('N4', cur_p)
            ws.update_acell('O4', order_qty)
            print(f"📤 시트 업데이트: {action} {order_qty}주 @ ${cur_p}")
        else:
            print(f"🧪 테스트 모드 — 시트 업데이트 *건너뜀*. 시뮬 결과: {action} {order_qty}주 @ ${cur_p}")

        # L. Telegram
        tier_emoji   = {'HIGH': '🟡', 'MID': '🔵', 'LOW': '🟢'}[this_tier]
        action_emoji = {'매도': '📈', '매수': '📉', '관망': '⏸'}[action]
        mode_label   = {'pre_open': 'Pre-market (어제 종가 기준, 오늘 LOC 준비)',
                        'near_close': 'Near-close (마감 직전, 즉시 LOC 권장)',
                        'post_close': 'Post-close (마감가 확정)',
                        'test': '🧪 테스트 모드 (지난 종가 기준 시뮬, 시트 미업데이트)'}.get(mode, '')
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        tax_line = ""
        if cum_paid_tax > 0:
            tax_line = f"  📌 누적 양도세 차감 : `−${cum_paid_tax:>10,.2f}` ({len(tax_payments)}건 반영)\n"
        floor_line = f"\n{floor_note}\n" if floor_note else ""
        title_prefix = "🧪 *[위대리-H — 테스트]*" if is_test_mode else "🚀 *[위대리-H vH1.0] 주간 시그널*"
        msg = (
            f"{title_prefix} ({today_str})\n"
            f"📡 모드: {mode_label}\n\n"
            f"💰 *자산 현황 (주문 체결 후 예상)*\n"
            f"  총 자산 : `${total_asset:>12,.2f}`  ({pnl_pct:+.2f}%)\n"
            f"  보유 주식: `{est_shares:>8,}` 주\n"
            f"  보유 현금: `${est_cash:>12,.2f}`  (현금 {cash_pct:.1f}%)\n"
            f"  주식 노출: `{expo_pct:.1f}%`  (하한 {floor:.0%})\n"
            f"{tax_line}\n"
            f"🌡️ *시장 평가* : {tier_emoji} `{last_eval:.2%}`  ({this_tier})\n"
            f"  QQQ 추세 대비 {abs(last_eval):.1%} "
            f"{'고평가' if last_eval >= hc else ('저평가' if last_eval <= lc else '중립')}"
            f"{floor_line}"
            f"\n🎯 *이번 주 LOC 주문*\n"
            f"  {action_emoji} 상태       : *{action}*\n"
            f"  📦 주문 수량   : *{order_qty}* 주\n"
            f"  💲 기준 가격   : `${cur_p}`\n"
            f"  📊 거래 후 잔고: *{est_shares:,}* 주\n"
            f"{recon_msg}\n"
        )
        if is_test_mode:
            msg += (f"⚠️ *시뮬 결과만 확인용* — 시트 미업데이트, 자동매매 미작동.\n"
                    f"  실제 주문은 정규 윈도우에서 자동 처리됩니다.\n")
        elif action != "관망":
            urgency = ("⚡ *지금 즉시* MTS 에서 LOC 주문 입력 (마감까지 시간 적음)\n" if mode == 'near_close' else "")
            msg += (f"📱 *MTS 실행 방법*\n  {urgency}"
                    f"  → `{action} {order_qty}주` LOC로 예약\n  → 금요일 장 마감 직전 체결\n")
        else:
            msg += "  → 이번 주 주문 없음 (관망)\n"

        ok_send = send_telegram(msg)
        if ok_send:
            print("✅ 모든 작업 완료 (Sheets + Telegram)")
        else:
            print("⚠️ Sheets 는 업데이트됐지만 Telegram 발송 실패")
            print("   → 위 로그에서 원인 확인 + BOT_TOKEN/CHAT_ID 점검 필요")

    except Exception as e:
        import traceback
        tb = traceback.format_exc(); err_str = str(e) or "(빈 메시지)"; err_type = type(e).__name__
        recent_frames = '\n'.join(tb.strip().split('\n')[-6:])
        err_msg = (f"❌ [위대리-H 봇] 처리 중 오류\n\n*예외 타입*: `{err_type}`\n"
                   f"*에러*: `{err_str[:200]}`\n\n*위치 (최근 frame)*:\n```\n{recent_frames[:600]}\n```")
        print(err_msg); print("\n--- Full traceback ---"); print(tb); send_telegram(err_msg)


if __name__ == "__main__":
    main()