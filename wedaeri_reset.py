import gspread
import os
import json
import requests
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 1. 설정 정보 (위대리 오토봇과 동일)
# ─────────────────────────────────────────────────────────────
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'

# 환경변수에서 토큰 읽기 (GitHub Secrets 에서 주입)
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID   = os.environ.get('CHAT_ID', '')


# ─────────────────────────────────────────────────────────────
# 2. Telegram 발송
# ─────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram BOT_TOKEN/CHAT_ID 미설정 — 발송 스킵")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 3. 리셋 작업
# ─────────────────────────────────────────────────────────────
def reset_wedaeri_orders():
    print(f"🔄 [{datetime.now()}] 위대리 주문 수량 초기화 시작...")

    try:
        # GCP 인증 및 시트 연결
        creds_raw = os.environ.get('GCP_CREDENTIALS')
        if not creds_raw:
            print("❌ GCP_CREDENTIALS 환경변수가 없습니다.")
            return

        creds_dict = json.loads(creds_raw)
        if 'private_key' in creds_dict:
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')

        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet("위대리")

        # 주문 정보 초기화 (L4: 상태, M4: 방식, O4: 수량)
        # N4(현재가)는 참고용이므로 두거나 0으로 만듭니다.
        ws.update_acell('L4', '관망(리셋)')
        ws.update_acell('M4', '-')
        ws.update_acell('O4', 0)

        msg = (
            "🧹 *[위대리] 주간 주문 초기화 완료*\n"
            "- 이번 주 주문 수량이 `0`으로 리셋되었습니다.\n"
            "- 즐거운 주말 보내세요, 용성님!"
        )
        send_telegram(msg)
        print("✅ 초기화 완료 및 텔레그램 발송 성공")

    except Exception as e:
        err_msg = f"🚨 *[위대리 리셋 에러]*\n작업 중 오류 발생: {e}"
        send_telegram(err_msg)
        print(err_msg)


# ─────────────────────────────────────────────────────────────
# 4. 진입점
#    스케줄링은 GitHub Actions cron 이 담당합니다.
#    이 스크립트는 한 번 실행되고 종료됩니다.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reset_wedaeri_orders()
