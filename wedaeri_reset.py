# wedaeri_reset.py — 위대리 주간 주문표 리셋
#   실행: GitHub Actions cron (토 03:00 UTC = KST 토 12:00)
#
# 변경점
#   ① '관망(리셋)' → '관망'  : Astra 등 자동매매가 읽는 값과 정확히 일치시킴
#   ② 503/500 일시 장애 재시도(지수 백오프)
#   ③ batch_update 로 API 호출 3회 → 1회
#   ④ 시간 가드: 금요일 마감 이후(ET) ~ 일요일에만 리셋 허용
#   ⑤ 에러 알림은 parse_mode 없이 발송 (Markdown 깨짐 방지)

import gspread
import os
import json
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
SHEET_TAB = '위대리'

BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID   = os.environ.get('CHAT_ID', '')

# 가드를 무시하고 강제 실행하려면 FORCE_RESET=1
FORCE_RESET = os.environ.get('FORCE_RESET', '') == '1'


# ─────────────────────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────────────────────
def send_telegram(text: str, markdown: bool = True) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram BOT_TOKEN/CHAT_ID 미설정 — 발송 스킵")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    if markdown:
        payload["parse_mode"] = "Markdown"
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"🔴 Telegram {r.status_code}: {r.text[:200]}")
            # Markdown 파싱 실패면 plain 으로 재시도
            if markdown:
                requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"🔴 Telegram 발송 예외: {e}")


# ─────────────────────────────────────────────────────────────
# 일시 장애 재시도 (503/500/429/timeout)
# ─────────────────────────────────────────────────────────────
TRANSIENT = ('503', '500', '502', '504', '429',
             'unavailable', 'internal error', 'timeout', 'deadline')


def retry(fn, tries: int = 4, label: str = ''):
    last = None
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(t in msg for t in TRANSIENT) and i < tries:
                wait = 2 ** i          # 2, 4, 8초
                print(f"⏳ 일시 장애 [{label}] {type(e).__name__}: {str(e)[:120]}")
                print(f"   {wait}초 후 재시도 ({i}/{tries})")
                time.sleep(wait)
                continue
            raise
    raise last


# ─────────────────────────────────────────────────────────────
# 시간 가드 — 장중 리셋으로 주문이 지워지는 사고 방지
# ─────────────────────────────────────────────────────────────
def timing_ok() -> tuple:
    """금요일 마감(ET 16:00) 이후 ~ 일요일 사이만 허용."""
    et = datetime.now(ZoneInfo("America/New_York"))
    wd = et.weekday()                       # 월0 … 금4 토5 일6
    if wd == 4 and et.hour >= 17:           # 금요일 마감+여유 1시간
        return True, f"금요일 마감 후 (ET {et:%Y-%m-%d %H:%M})"
    if wd in (5, 6):                        # 토·일
        return True, f"주말 (ET {et:%Y-%m-%d %H:%M})"
    return False, (f"리셋 허용 시간대가 아님 (ET {et:%Y-%m-%d %H:%M}, 요일={wd}). "
                   f"장중 리셋은 이번 주 주문을 지울 수 있어 중단합니다.")


# ─────────────────────────────────────────────────────────────
# 리셋
# ─────────────────────────────────────────────────────────────
def reset_wedaeri_orders():
    print(f"🔄 [{datetime.now()}] 위대리 주문 수량 초기화 시작...")

    ok, why = timing_ok()
    if not ok and not FORCE_RESET:
        print(f"⛔ {why}")
        send_telegram(f"⛔ [위대리 리셋] 건너뜀\n{why}", markdown=False)
        return
    if FORCE_RESET and not ok:
        print(f"⚠️ FORCE_RESET=1 — 시간 가드 무시하고 진행 ({why})")
    else:
        print(f"✅ {why}")

    try:
        creds_raw = os.environ.get('GCP_CREDENTIALS')
        if not creds_raw:
            raise RuntimeError("GCP_CREDENTIALS 환경변수가 없습니다.")

        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')

        print(f"📧 서비스 계정: {creds_dict.get('client_email', 'unknown')[:60]}")

        gc = gspread.service_account_from_dict(creds_dict)
        sh = retry(lambda: gc.open_by_key(SHEET_KEY), label='open_sheet')
        ws = retry(lambda: sh.worksheet(SHEET_TAB), label='open_tab')

        # 리셋 전 값 백업 (로그용)
        try:
            before = retry(lambda: ws.batch_get(['L4', 'M4', 'N4', 'O4']),
                           label='read_before')
            print(f"📄 리셋 전 L4~O4: {before}")
        except Exception as e:
            print(f"ℹ️ 리셋 전 값 읽기 실패(무시): {e}")

        # ★ L4 는 반드시 '관망' — 괄호 등 부가 문자열 금지 (자동매매 매칭 보호)
        retry(lambda: ws.batch_update([
            {'range': 'L4', 'values': [['관망']]},
            {'range': 'M4', 'values': [['-']]},
            {'range': 'O4', 'values': [[0]]},
        ]), label='batch_update')

        # 리셋 이력은 별도 셀에 (자동매매가 읽지 않는 위치)
        try:
            stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y-%m-%d %H:%M KST')
            retry(lambda: ws.update_acell('Q4', f'리셋 {stamp}'), label='stamp')
        except Exception as e:
            print(f"ℹ️ 리셋 스탬프 기록 실패(무시): {e}")

        msg = ("🧹 *[위대리] 주간 주문 초기화 완료*\n"
               "- L4 `관망` / M4 `-` / O4 `0` 으로 리셋\n"
               "- 즐거운 주말 보내세요, 용성님!")
        send_telegram(msg)
        print("✅ 초기화 완료 및 텔레그램 발송 성공")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        err = (f"🚨 [위대리 리셋 에러]\n"
               f"타입: {type(e).__name__}\n"
               f"내용: {str(e)[:300]}\n\n"
               f"※ 503/500 이면 구글 일시 장애입니다. "
               f"잠시 후 워크플로우를 수동 재실행하세요.")
        send_telegram(err, markdown=False)


if __name__ == "__main__":
    reset_wedaeri_orders()
