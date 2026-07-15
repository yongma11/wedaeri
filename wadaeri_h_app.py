# wedaeri_h_app.py — TQQQ 위대리-H vH1.0
# Tab1: 실전 트레이딩 | Tab2: 백테스트 분석 | Tab3: 전략 로직
#
# ══════════════════════════════════════════════════════════════════════════
#  위대리-H (Hedge-hardened) — 원작자 위대리 v1.0 + 노출 하한 15%
# ──────────────────────────────────────────────────────────────────────────
#  [왜 -H 인가]
#   원작자 위대리는 강세장이 길어지면 스스로 현금에 갇힌다(경화).
#   계속 팔다 보니 보유 주식이 마르고 매수할 힘까지 잃어 노출이 3~4%로 굳는다.
#   → 3배 레버리지 상품인데 실질 노출 0.1배. 사실상 잠들어버림.
#   위대리-H는 '노출 하한 15%'를 넣어, 노출이 그 밑으로 내려가면 매도를 멈춘다.
#   시작 시점과 무관하게 노출이 15%로 통일되고, 폭락 때 반격할 실탄이 늘 남는다.
#
#  [핵심 변경점 — 원작자 v1.0 파라미터 채택 + 하한]
#   ① 매도/매수를 '상승·하락분의 %' 방식으로 (원작자 원문 표, 251211 수정판)
#        매도율:  HIGH(Eval>=+5%) 100%  MID 50%  LOW(<=-6%) 33.3%
#        매수율:  HIGH 50%  MID 50%  LOW 100%
#      ※ 기존 앱은 diff x 배율(sr=2.0 등)이라 원작자보다 과공격적이었음.
#   ② HIGH 컷 +5% (원작자), LOW 컷 -6% (동일)
#   ③ 노출 하한 15%: 매도 후 노출이 15% 밑이면 그 매도를 취소(주식 유지)
#
#  [검증 요약 — 2010.04~2026.07, 세후(양도세22%·수수료0.1%)]
#     위대리-H 100%          CAGR 16.8%  MDD -32.3%  Calmar 0.52
#     동파공70/위대리-H30      CAGR 35.5%  MDD -31.6%  Calmar 1.12  <- 권장(헤지)
#     동파공60/위H20/SGOV20   CAGR 30.0%  MDD -26.3%  Calmar 1.14  <- 방어강화형
#   · 위대리-H는 단독 완결 전략이 아니라 동파공의 '살아있는 헤지 슬리브'로 쓴다.
#   · 신규 편입 시 목돈을 3~6개월 분할 적립하면 진입 시점 리스크가 분산된다.
#
#  [유지] 실전/백테스트/게이트/양도세/시트연동 구조는 원본 v5.2와 동일.
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 위대리-H 파라미터 (원작자 v1.0 % 방식 + 노출 하한)
#   SELL_RATE / BUY_RATE : 상승분·하락분에 곱하는 '비율'(0~1). 최대 1.0 = 100%.
#   FLOOR_EXPO : 노출 하한. 매도 후 노출이 이 값 밑이면 매도 취소.
# ─────────────────────────────────────────────────────────────
SELL_RATE = {'HIGH': 1.000, 'MID': 0.500, 'LOW': 0.333}
BUY_RATE  = {'HIGH': 0.500, 'MID': 0.500, 'LOW': 1.000}
FLOOR_EXPO = 0.15   # ★ 위대리-H 핵심

CONFIG_FILE = Path(__file__).parent / "wedaeri_h_config.json"
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
DEFAULT_CONFIG = {
    'start_date': '2025-12-26',
    'cap':        108000,
    'cash':       45,
    'bt_cap':     10000,
    'bt_cash':    45,
    'bt_start':   '2010-01-01',
    'hc':  5.0,   'lc': -6.0,          # ★ HIGH 컷 +5% (원작자)
    'floor': 15.0,                     # ★ 노출 하한 %
    'tax_apply_to_bot': False,
    'use_gate': False,
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding='utf-8') as f:
                saved = json.load(f)
            return {**DEFAULT_CONFIG, **saved}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(ss) -> dict:
    raw_start = ss.get('p_start', DEFAULT_CONFIG['start_date'])
    start_str = raw_start.strftime("%Y-%m-%d") if hasattr(raw_start, 'strftime') else str(raw_start)[:10]
    raw_bt_start = ss.get('p_bt_start', DEFAULT_CONFIG['bt_start'])
    bt_start_str = raw_bt_start.strftime("%Y-%m-%d") if hasattr(raw_bt_start, 'strftime') else str(raw_bt_start)[:10]
    cfg = {
        'start_date': start_str,
        'cap':     int(ss.get('p_cap',      DEFAULT_CONFIG['cap'])),
        'cash':    int(ss.get('p_cash',     DEFAULT_CONFIG['cash'])),
        'bt_cap':  int(ss.get('p_bt_cap',   DEFAULT_CONFIG['bt_cap'])),
        'bt_cash': int(ss.get('p_bt_cash',  DEFAULT_CONFIG['bt_cash'])),
        'bt_start': bt_start_str,
        'hc':  float(ss.get('p_hc',  DEFAULT_CONFIG['hc'])),
        'lc':  float(ss.get('p_lc',  DEFAULT_CONFIG['lc'])),
        'floor': float(ss.get('p_floor', DEFAULT_CONFIG['floor'])),
        'tax_apply_to_bot': bool(ss.get('p_tax_apply_to_bot', DEFAULT_CONFIG['tax_apply_to_bot'])),
        'use_gate': bool(ss.get('p_use_gate', DEFAULT_CONFIG['use_gate'])),
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


@st.cache_resource(show_spinner=False)
def _gspread_client_cached(creds_hash: str, creds_json: str):
    import gspread
    creds = json.loads(creds_json)
    if 'private_key' in creds:
        creds['private_key'] = creds['private_key'].replace('\\n', '\n')
    return gspread.service_account_from_dict(creds)


def _get_gspread_client():
    try:
        creds = None
        try:
            sec = st.secrets["gcp_service_account"]
            creds = dict(sec)
        except Exception:
            pass
        if creds is None:
            try:
                raw = st.secrets.get("GCP_CREDENTIALS")
                if raw:
                    creds = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                pass
        if creds is None:
            raw = os.environ.get("GCP_CREDENTIALS")
            if raw:
                creds = json.loads(raw)
        if creds is None:
            return None, "Streamlit Secrets에 gcp_service_account 또는 GCP_CREDENTIALS가 없습니다"
        cache_key = f"{creds.get('client_email','')}:{creds.get('private_key_id','')}"
        creds_json = json.dumps(creds, sort_keys=True)
        gc = _gspread_client_cached(cache_key, creds_json)
        return gc, None
    except Exception as e:
        return None, str(e)


def _sheets_write(ws, rows: list) -> None:
    try:
        ws.update('A1', rows)
    except TypeError:
        ws.update(rows, 'A1')


def save_config_to_sheets(cfg: dict) -> tuple:
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return False, err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet("설정")
        except Exception:
            ws = sh.add_worksheet(title="설정", rows=30, cols=2)
        rows = [['key', 'value']]
        for k, v in cfg.items():
            rows.append([k, str(v)])
        ws.clear()
        _sheets_write(ws, rows)
        return True, ""
    except Exception as e:
        return False, str(e)


TAX_SHEET_NAME = "양도세납부"
TAX_SHEET_HEADER = ['date', 'for_year', 'amount_usd', 'amount_krw', 'fx_rate', 'note']


def load_tax_payments() -> tuple:
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return [], err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(TAX_SHEET_NAME)
        except Exception:
            return [], ""
        records = ws.get_all_records()
        out = []
        for r in records:
            try:
                out.append({
                    'date':       str(r.get('date', '')),
                    'for_year':   int(float(r.get('for_year', 0))),
                    'amount_usd': float(r.get('amount_usd', 0)),
                    'amount_krw': float(r.get('amount_krw', 0)),
                    'fx_rate':    float(r.get('fx_rate', 1300)),
                    'note':       str(r.get('note', '')),
                })
            except Exception:
                continue
        return out, ""
    except Exception as e:
        return [], str(e)


def save_tax_payment(payment: dict) -> tuple:
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return False, err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(TAX_SHEET_NAME)
        except Exception:
            ws = sh.add_worksheet(title=TAX_SHEET_NAME, rows=200, cols=6)
            _sheets_write(ws, [TAX_SHEET_HEADER])
        new_row = [str(payment.get(k, '')) for k in TAX_SHEET_HEADER]
        ws.append_row(new_row, value_input_option='USER_ENTERED')
        return True, ""
    except Exception as e:
        return False, str(e)


def delete_tax_payment(date_str: str) -> tuple:
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return False, err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(TAX_SHEET_NAME)
        except Exception:
            return False, "시트 없음"
        records = ws.get_all_records()
        for i, r in enumerate(records):
            if str(r.get('date', '')) == date_str:
                ws.delete_rows(i + 2)
                return True, ""
        return False, f"date={date_str} 기록을 찾을 수 없음"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────
# 동파공 총자산 읽기 (Dongpa 탭 B8) — 7:3 포트폴리오 통합용
# ─────────────────────────────────────────────────────────────
DONGPA_SHEET_NAME = "Dongpa"
DONGPA_TOTAL_CELL = "B8"


@st.cache_data(ttl=120, show_spinner=False)
def load_dongpa_total() -> tuple:
    """Dongpa 탭 B8 의 총자산(달러)을 읽는다. Returns: (value|None, error_str)."""
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return None, err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(DONGPA_SHEET_NAME)
        except Exception:
            return None, f"'{DONGPA_SHEET_NAME}' 탭을 찾을 수 없음"
        raw = ws.acell(DONGPA_TOTAL_CELL).value
        if raw is None or str(raw).strip() == "":
            return None, f"{DONGPA_SHEET_NAME}!{DONGPA_TOTAL_CELL} 이 비어 있음"
        # "$434,606" / "434606" / "434,606.00" 등 정리
        cleaned = str(raw).replace("$", "").replace(",", "").replace("₩", "").strip()
        return float(cleaned), ""
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# 현금 조정(리밸런싱 입출금) 기록 — 주식수 불변, 예수금만 변동
#   시트 '현금조정' 탭: date | amount | note
#   amount 양수=입금(위대리로 유입), 음수=출금(위대리에서 유출)
# ─────────────────────────────────────────────────────────────
CASHADJ_SHEET_NAME = "현금조정"
CASHADJ_HEADER = ['date', 'amount', 'note']


@st.cache_data(ttl=120, show_spinner=False)
def load_cash_adjustments() -> tuple:
    """Returns: (list[{date,amount,note}], error_str)."""
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return [], err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(CASHADJ_SHEET_NAME)
        except Exception:
            return [], ""
        records = ws.get_all_records()
        out = []
        for r in records:
            try:
                amt_raw = str(r.get('amount', '')).replace('$', '').replace(',', '').strip()
                if amt_raw == '':
                    continue
                out.append({'date': str(r.get('date', '')),
                            'amount': float(amt_raw),
                            'note': str(r.get('note', ''))})
            except Exception:
                continue
        out.sort(key=lambda x: x['date'])
        return out, ""
    except Exception as e:
        return [], str(e)


def save_cash_adjustment(adj: dict) -> tuple:
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return False, err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(CASHADJ_SHEET_NAME)
        except Exception:
            ws = sh.add_worksheet(title=CASHADJ_SHEET_NAME, rows=200, cols=3)
            _sheets_write(ws, [CASHADJ_HEADER])
        ws.append_row([str(adj.get(k, '')) for k in CASHADJ_HEADER],
                      value_input_option='USER_ENTERED')
        return True, ""
    except Exception as e:
        return False, str(e)


def delete_cash_adjustment(date_str: str, amount: float) -> tuple:
    try:
        gc, err = _get_gspread_client()
        if gc is None:
            return False, err or "인증 실패"
        sh = gc.open_by_key(SHEET_KEY)
        try:
            ws = sh.worksheet(CASHADJ_SHEET_NAME)
        except Exception:
            return False, "시트 없음"
        records = ws.get_all_records()
        for i, r in enumerate(records):
            try:
                amt_raw = str(r.get('amount', '')).replace('$', '').replace(',', '').strip()
                if str(r.get('date', '')) == date_str and abs(float(amt_raw) - amount) < 0.01:
                    ws.delete_rows(i + 2)
                    return True, ""
            except Exception:
                continue
        return False, "기록을 찾을 수 없음"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────
# 페이지 설정 & 스타일 — Deep-Teal Navigation Console
#   위대리-H 정체성: 심해 청록(#1e5f74) · 조개잡이가 썰물을 기다리는 심해
#   (동파공=불꽃 오렌지와 대비되는 짝. 리포트와 색 통일)
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="위대리-H · Navigation Console", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@300;400;500;600&display=swap');
:root{
--abyss:#0a1418; --abyss-2:#0e1b21; --panel:#122630; --panel-2:#16303c;
--teal:#2b9db8; --teal-deep:#1e5f74; --teal-glow:#4dd0e1;
--amber:#d99a1c; --crimson:#c65454; --good:#3fb37f;
--ink:#dce8ec; --mist:#7b98a3; --mist-2:#5a7480; --line:#1d3843;
--mono:'IBM Plex Mono',monospace; --disp:'Space Grotesk',sans-serif;
--sans:'IBM Plex Sans KR',system-ui,sans-serif;
}
.stApp{ background:
radial-gradient(1200px 600px at 20% -10%, #12303b55 0%, transparent 60%),
radial-gradient(900px 500px at 90% 0%, #1e5f7433 0%, transparent 55%),
var(--abyss); }
.main .block-container{ padding-top:2.2rem; max-width:1200px; }
body, .stMarkdown, p, span, div{ font-family:var(--sans); color:var(--ink); }
h1,h2,h3,h4{ font-family:var(--disp); letter-spacing:-0.01em; }
.stApp a{ color:var(--teal-glow); }
[data-testid="stHeader"]{ background:transparent; }

[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0c1b22,#0a1418);
border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] .stMarkdown h2{ font-size:1.05rem; color:var(--teal-glow); }

.hero{ position:relative; border:1px solid var(--line); border-radius:18px;
padding:30px 34px; margin-bottom:8px; overflow:hidden;
background:linear-gradient(135deg, #103039 0%, #0c2029 55%, #0a1418 100%); }
.hero::before{ content:""; position:absolute; inset:0;
background:radial-gradient(600px 200px at 85% 20%, #2b9db822, transparent 70%);
pointer-events:none; }
.hero-eyebrow{ font-family:var(--mono); font-size:11px; letter-spacing:.32em;
text-transform:uppercase; color:var(--teal); font-weight:600; }
.hero-title{ font-family:var(--disp); font-weight:700; font-size:2.5rem; line-height:1.05;
margin:10px 0 6px; color:#eaf6f9; }
.hero-title .h{ color:var(--teal-glow); }
.hero-sub{ font-size:.95rem; color:var(--mist); max-width:62ch; line-height:1.6; }
.hero-chips{ display:flex; gap:8px; flex-wrap:wrap; margin-top:16px; }
.chip{ font-family:var(--mono); font-size:11px; padding:5px 11px; border-radius:20px;
border:1px solid var(--line); background:#0e2129; color:var(--mist); }
.chip b{ color:var(--teal-glow); font-weight:600; }

[data-testid="stMetric"]{ background:linear-gradient(160deg,#132a34,#0e2027);
border:1px solid var(--line); border-radius:14px; padding:16px 18px;
box-shadow:0 4px 18px #05101499; transition:border-color .2s, transform .2s; }
[data-testid="stMetric"]:hover{ border-color:var(--teal-deep); transform:translateY(-2px); }
[data-testid="stMetricLabel"]{ font-family:var(--mono); font-size:11px !important;
letter-spacing:.05em; color:var(--mist) !important; text-transform:uppercase; }
[data-testid="stMetricValue"]{ font-family:var(--disp); font-weight:600;
color:#eaf6f9 !important; font-variant-numeric:tabular-nums; }
[data-testid="stMetricDelta"]{ font-family:var(--mono); font-size:12px !important; }

.stTabs [data-baseweb="tab-list"]{ gap:4px; background:#0c1c23; padding:5px;
border-radius:12px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ height:42px; border-radius:8px; padding:0 22px;
font-family:var(--disp); font-weight:500; font-size:.92rem; color:var(--mist);
background:transparent; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#1e5f74,#134350) !important;
color:#eaf6f9 !important; box-shadow:0 2px 12px #1e5f7455; }

.order-card-buy{ background:linear-gradient(160deg,#0f3328,#0c2620);
border:1px solid #1c5a44; border-radius:16px; padding:26px; text-align:center; }
.order-card-buy h4{ color:var(--good); font-family:var(--mono); font-size:12px;
letter-spacing:.15em; text-transform:uppercase; margin:0 0 8px; }
.order-card-buy .big{ font-family:var(--disp); font-size:2rem; font-weight:700; color:#5fe0a8; }
.order-card-sell{ background:linear-gradient(160deg,#2a1618,#231215);
border:1px solid #5a2c30; border-radius:16px; padding:26px; text-align:center; }
.order-card-sell h4{ color:var(--crimson); font-family:var(--mono); font-size:12px;
letter-spacing:.15em; text-transform:uppercase; margin:0 0 8px; }
.order-card-sell .big{ font-family:var(--disp); font-size:2rem; font-weight:700; color:#e88a8a; }
.order-card-hold{ background:linear-gradient(160deg,#16232b,#111c22);
border:1px solid var(--line); border-radius:16px; padding:26px; text-align:center; }
.order-card-hold h4{ color:var(--mist); font-family:var(--mono); font-size:12px;
letter-spacing:.15em; text-transform:uppercase; margin:0 0 8px; }
.order-card-hold .big{ font-family:var(--disp); font-size:2rem; font-weight:700; color:var(--mist); }

.tier-high{ background:linear-gradient(90deg,#3a2f0c,#1a1608); border-left:3px solid var(--amber);
padding:12px 16px; border-radius:8px; margin:6px 0; }
.tier-mid{ background:linear-gradient(90deg,#0c2a33,#0a1c22); border-left:3px solid var(--teal);
padding:12px 16px; border-radius:8px; margin:6px 0; }
.tier-low{ background:linear-gradient(90deg,#0c3326,#0a1f18); border-left:3px solid var(--good);
padding:12px 16px; border-radius:8px; margin:6px 0; }
.floor-badge{ background:linear-gradient(90deg,#0e2b34,#0b1e25); border-left:3px solid var(--teal-glow);
padding:14px 18px; border-radius:8px; margin:8px 0; font-size:.9rem; color:var(--ink); }

.sec-head{ font-family:var(--disp); font-weight:600; font-size:1.15rem; color:#eaf6f9;
margin:22px 0 10px; padding-left:12px; border-left:3px solid var(--teal); }
.sec-eyebrow{ font-family:var(--mono); font-size:10.5px; letter-spacing:.18em;
text-transform:uppercase; color:var(--teal); margin-bottom:2px; }

.callout{ border-left:3px solid var(--teal-deep); background:#0e232b;
padding:16px 20px; border-radius:0 10px 10px 0; margin:16px 0; font-size:.92rem;
line-height:1.7; color:var(--ink); }
.callout.amber{ border-left-color:var(--amber); background:#241c0c; }
.callout.crimson{ border-left-color:var(--crimson); background:#241214; }
.callout .k{ font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
text-transform:uppercase; color:var(--teal-glow); font-weight:600; display:block; margin-bottom:6px; }
.callout.amber .k{ color:var(--amber); } .callout.crimson .k{ color:#e88a8a; }

.stDataFrame{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stMarkdown table{ border-collapse:collapse; width:100%; margin:12px 0; font-family:var(--mono); font-size:.86rem; }
.stMarkdown th{ background:#0e232b; color:var(--teal-glow); font-weight:600;
padding:9px 12px; border-bottom:1px solid var(--teal-deep); text-align:left;
font-family:var(--mono); font-size:.8rem; text-transform:uppercase; letter-spacing:.03em; }
.stMarkdown td{ padding:8px 12px; border-bottom:1px solid var(--line); color:var(--ink); }
.stMarkdown tr:hover td{ background:#0e2129; }

.stButton>button{ font-family:var(--disp); font-weight:500; border-radius:9px;
border:1px solid var(--teal-deep); background:#0f2831; color:var(--teal-glow); transition:all .18s; }
.stButton>button:hover{ background:#1e5f74; color:#eaf6f9; border-color:var(--teal); }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:12px;
background:#0d1e25; overflow:hidden; }
[data-baseweb="tag"]{ background:var(--teal-deep) !important; }
hr{ border-color:var(--line); }
.stAlert{ border-radius:12px; }
</style>
""", unsafe_allow_html=True)


def sec(title, eyebrow=""):
    """섹션 헤더 렌더 헬퍼."""
    eb = f'<div class="sec-eyebrow">{eyebrow}</div>' if eyebrow else ''
    st.markdown(f'{eb}<div class="sec-head">{title}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# session_state 초기화
# ─────────────────────────────────────────────────────────────
if '_cfg_loaded' not in st.session_state:
    _cfg = load_config()
    ss   = st.session_state
    ss['_cfg_loaded'] = True
    ss['p_start']    = datetime.strptime(str(_cfg['start_date'])[:10], "%Y-%m-%d").date()
    ss['p_cap']      = _cfg['cap']
    ss['p_cash']     = _cfg['cash']
    ss['p_bt_cap']   = _cfg['bt_cap']
    ss['p_bt_cash']  = _cfg['bt_cash']
    ss['p_bt_start'] = datetime.strptime(str(_cfg.get('bt_start', '2010-01-01'))[:10], "%Y-%m-%d").date()
    ss['p_hc']      = _cfg['hc']
    ss['p_lc']      = _cfg['lc']
    ss['p_floor']   = _cfg.get('floor', 15.0)
    ss['p_tax_apply_to_bot'] = bool(_cfg.get('tax_apply_to_bot', False))
    ss['p_use_gate'] = bool(_cfg.get('use_gate', False))

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,25,31,0.6)',
    font=dict(color='#9fbcc7', size=11, family='IBM Plex Mono'),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(orientation='h', y=1.14, font=dict(size=10)),
    title_font=dict(family='Space Grotesk', size=14, color='#dce8ec'),
    hoverlabel=dict(bgcolor='#122630', bordercolor='#1e5f74', font=dict(family='IBM Plex Mono')),
)
# 위대리-H 시그니처 팔레트
C_TEAL = '#2b9db8'; C_TEAL_GLOW = '#4dd0e1'; C_AMBER = '#d99a1c'
C_CRIMSON = '#c65454'; C_GOOD = '#3fb37f'; C_BH = '#e8863f'


def apply_grid(fig):
    fig.update_xaxes(gridcolor='#12262e', zeroline=False, linecolor='#1d3843')
    fig.update_yaxes(gridcolor='#12262e', zeroline=False, linecolor='#1d3843')
    return fig


# ─────────────────────────────────────────────────────────────
# 데이터 로딩 — get_data 항상 2005부터 (OLS 워밍업 완전창; 봇과 동일)
#   ※ 원본 v5.2의 워밍업 아티팩트(2010부터 로드 시 초기 Eval 왜곡)를 방지.
# ─────────────────────────────────────────────────────────────
DATA_START_EXT  = "1999-01-01"
DATA_START      = "2005-01-01"   # ★ 완전창 확보: 260주 워밍업 후 2010부터 유효
SYNTH_FINANCING = 0.03
SYNTH_EXPENSE   = 0.0086


@st.cache_data(ttl=3600, show_spinner=False)
def load_wedaeri_data(extended: bool = False):
    try:
        start = DATA_START_EXT if extended else DATA_START
        raw = yf.download(["QQQ", "TQQQ"], start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw['Close'].copy()
        else:
            df = raw.copy()
        df = df.reset_index()
        df.columns = [str(c).strip() for c in df.columns]
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['QQQ']).reset_index(drop=True)

        df['synthetic'] = False
        if 'TQQQ' in df.columns and df['TQQQ'].notna().any():
            first_real_idx = int(df['TQQQ'].first_valid_index())
            if extended and first_real_idx > 0:
                qret = df['QQQ'].pct_change().fillna(0.0)
                daily_drag = (3 - 1) * SYNTH_FINANCING / 252 + SYNTH_EXPENSE / 252
                synth = (1 + (3 * qret - daily_drag)).cumprod()
                scale = float(df.loc[first_real_idx, 'TQQQ']) / float(synth.iloc[first_real_idx])
                pre = df.index < first_real_idx
                df.loc[pre, 'TQQQ'] = (synth * scale)[pre].values
                df.loc[pre, 'synthetic'] = True

        # 주간 QQQ OLS → Growth/Eval (표시용)
        qqq_wkly = (df.set_index('Date')[['QQQ']].resample('W-FRI').last().dropna().reset_index())
        qqq_wkly['Growth'] = compute_expanding_ols(qqq_wkly, W=260)
        qqq_wkly['Eval']   = qqq_wkly['QQQ'] / qqq_wkly['Growth'] - 1
        df = df.merge(qqq_wkly[['Date', 'Growth', 'Eval']], on='Date', how='left')
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()


def compute_expanding_ols(qqq_weekly: pd.DataFrame, W: int = 260) -> np.ndarray:
    """주간 QQQ로 5년(260주) Expanding Window log-선형 회귀. 봇과 동일."""
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


def build_weekly_panel(data: pd.DataFrame, gate_sma_days: int = 200) -> pd.DataFrame:
    """봇과 동일한 주간 패널 + 200일 추세 게이트 플래그."""
    d = data.set_index('Date')
    qqq_weekly  = d[['QQQ']].resample('W-FRI').last().dropna().reset_index()
    tqqq_weekly = d[['TQQQ']].resample('W-FRI').last().dropna().reset_index()
    qqq_weekly['Growth'] = compute_expanding_ols(qqq_weekly, W=260)
    qqq_weekly['Eval']   = qqq_weekly['QQQ'] / qqq_weekly['Growth'] - 1
    panel = qqq_weekly.merge(tqqq_weekly, on='Date', how='inner')
    sma = d['QQQ'].rolling(gate_sma_days, min_periods=gate_sma_days).mean()
    gate_daily = (d['QQQ'] > sma).where(sma.notna())
    gate_wk = gate_daily.resample('W-FRI').last().reindex(panel['Date'].values)
    panel['gate_on'] = pd.Series(gate_wk.values).fillna(True).astype(bool).values
    return panel


def get_last_friday(today: pd.Timestamp) -> pd.Timestamp:
    """오늘 기준 '가장 최근 금요일'을 반환.
    - 금요일이면 오늘, 토/일이면 그 주 금요일, 월~목이면 지난주 금요일.
    봇의 W-FRI 주봉 기준과 정확히 일치시키기 위함."""
    wd = today.weekday()  # 월0 ~ 일6
    if wd == 4:      # 금
        return today
    elif wd == 5:    # 토
        return today - pd.Timedelta(days=1)
    elif wd == 6:    # 일
        return today - pd.Timedelta(days=2)
    else:            # 월(0)~목(3) → 지난주 금요일
        return today - pd.Timedelta(days=wd + 3)


def inject_live_price(df: pd.DataFrame, use_live: bool = False) -> pd.DataFrame:
    """앱이 참조할 마지막 데이터 시점을 결정한다.

    use_live=False (기본, 봇과 동일):
        아무것도 주입하지 않고 데이터의 '가장 최근 금요일 종가'까지만 사용.
        → 봇의 post-close 모드(금요일 종가 확정)와 동일한 신호/주문표를 보여준다.
        장중 노이즈 없이 주 1회 확정 신호만 본다.

    use_live=True (선택):
        yfinance 실시간가를 마지막 행에 덮어써 장중 미리보기를 제공한다.
        (기존 동작. 봇과는 시점이 달라 주문표가 다를 수 있음)
    """
    if not use_live:
        # 가장 최근 금요일까지로 데이터를 자른다 (그 이후 임시 행이 있으면 제거).
        try:
            today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            last_fri = get_last_friday(today)
            df = df[df['Date'] <= last_fri].reset_index(drop=True)
        except Exception:
            pass
        return df

    # ── use_live=True: 실시간가 주입 (기존 동작) ──
    try:
        live_tqqq = float(yf.Ticker("TQQQ").fast_info['last_price'])
        live_qqq  = float(yf.Ticker("QQQ").fast_info['last_price'])
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        wd = today.weekday()
        if wd == 5:
            target = today - pd.Timedelta(days=1)
        elif wd == 6:
            target = today - pd.Timedelta(days=2)
        else:
            target = today
        last_df_date = df['Date'].iloc[-1]
        if target > last_df_date:
            last_eval   = float(df['Eval'].dropna().iloc[-1])
            last_growth = float(df['Growth'].dropna().iloc[-1])
            new_row = pd.DataFrame({
                'Date': [target], 'TQQQ': [live_tqqq], 'QQQ': [live_qqq],
                'Eval': [last_eval], 'Growth': [last_growth]
            })
            df = pd.concat([df, new_row], ignore_index=True).reset_index(drop=True)
        else:
            df.loc[df.index[-1], 'TQQQ'] = live_tqqq
            df.loc[df.index[-1], 'QQQ']  = live_qqq
            if pd.isna(df.loc[df.index[-1], 'Eval']):
                df.loc[df.index[-1], 'Eval']   = float(df['Eval'].dropna().iloc[-2])
                df.loc[df.index[-1], 'Growth'] = float(df['Growth'].dropna().iloc[-2])
    except Exception:
        pass
    return df


# ═════════════════════════════════════════════════════════════════════════
#  실전 시뮬레이션 (Tab 1) — 위대리-H 로직
#  매도: 상승분 x SELL_RATE[tier], 단 노출이 FLOOR 밑이면 매도 취소
#  매수: 하락분 x BUY_RATE[tier], 현금 한도 내
# ═════════════════════════════════════════════════════════════════════════
def run_wedaeri_sim(data, start_dt, init_cap, cash_ratio,
                    hc=0.05, lc=-0.06, floor=FLOOR_EXPO,
                    use_gate=False, gate_sma_days=200,
                    cash_adjustments=None):
    """cash_adjustments: [{'date':'YYYY-MM-DD','amount':float}, ...]
       리밸런싱 등으로 예수금만 조정(주식 수 불변). amount 양수=입금, 음수=출금.
       해당 날짜가 속한 주(그 주 금요일 종가 시점)에 현금에 반영된다."""
    panel = build_weekly_panel(data, gate_sma_days=gate_sma_days)
    wkly = (panel[panel['Date'] >= pd.to_datetime(start_dt)]
            .dropna(subset=['Eval', 'TQQQ']).reset_index(drop=True))
    if wkly.empty or len(wkly) < 2:
        return pd.DataFrame()
    init_price = float(wkly.loc[0, 'TQQQ'])
    shares = int((init_cap * (1 - cash_ratio)) / init_price)
    cash   = init_cap - shares * init_price
    # 현금 조정 스케줄 (날짜순, 아직 미반영분)
    adj_pending = sorted(
        [{'date': pd.to_datetime(a['date']), 'amount': float(a['amount'])}
         for a in (cash_adjustments or []) if a.get('amount')],
        key=lambda x: x['date'])
    net_injected = 0.0   # 누적 순입금 (수익률 보정용)
    logs   = []
    for i in range(len(wkly)):
        p    = float(wkly.loc[i, 'TQQQ'])
        bar_date = pd.Timestamp(wkly.loc[i, 'Date'])
        prev_bar = pd.Timestamp(wkly.loc[i-1, 'Date']) if i > 0 else None
        # ── 현금 조정 반영: prev_bar < 조정일 <= bar_date 인 것 (첫 주는 <= bar_date) ──
        still = []
        for a in adj_pending:
            due = (a['date'] <= bar_date) if i == 0 else (prev_bar < a['date'] <= bar_date)
            if due:
                cash += a['amount']; net_injected += a['amount']
            else:
                still.append(a)
        adj_pending = still
        ev   = float(wkly.loc[i, 'Eval'])
        tier = 'HIGH' if ev >= hc else ('LOW' if ev <= lc else 'MID')
        gate_on = (not use_gate) or bool(wkly.loc[i, 'gate_on'])
        action, disp_qty = "관망", 0

        if use_gate and not gate_on:
            if shares > 0:
                disp_qty = -shares; cash += shares * p; shares = 0; action = "게이트청산"
            else:
                action = "게이트대기"
        elif use_gate and shares == 0 and cash > 0:
            qty = int((cash * (1 - cash_ratio)) / p)
            if qty > 0:
                disp_qty = qty; shares += qty; cash -= qty * p; action = "게이트재진입"
        else:
            if i > 0:
                prev_p = float(wkly.loc[i-1, 'TQQQ'])
                diff   = shares * (p - prev_p)
                total  = cash + shares * p
                if diff > 0:   # 상승 → 매도 (상승분의 rate%)
                    rate = SELL_RATE[tier]
                    qty = int(min(round(diff * rate / p), shares))
                    # ★ 노출 하한: 매도 후 노출이 floor 밑이면 매도량 제한
                    if floor > 0 and total > 0:
                        min_sh = int(floor * total / p)
                        qty = max(0, min(qty, shares - min_sh))
                    if qty > 0:
                        action, disp_qty = "매도", -qty
                        shares -= qty; cash += qty * p
                elif diff < 0:  # 하락 → 매수 (하락분의 rate%)
                    rate = BUY_RATE[tier]
                    qty = int(min(cash, abs(diff) * rate) / p)
                    if qty > 0:
                        action, disp_qty = "매수", qty
                        shares += qty; cash -= qty * p
        total = cash + shares * p
        _basis = init_cap + net_injected   # 입출금은 손익이 아니므로 기준에 반영
        logs.append({
            '날짜':     wkly.loc[i, 'Date'].strftime('%Y-%m-%d'),
            '시장평가': f"{ev:.2%}",
            '티어':     tier,
            '게이트':   ('—' if not use_gate else ('보유' if gate_on else '현금(이탈)')),
            '액션':     action,
            '주문수량': disp_qty,
            '보유수량': shares,
            '현금':     round(cash, 2),
            '노출':     f"{(shares*p/total*100 if total>0 else 0):.1f}%",
            '총자산':   round(total, 2),
            '수익률':   f"{((total/_basis - 1)*100 if _basis > 0 else 0):.2f}%",
        })
    return pd.DataFrame(logs)


# ═════════════════════════════════════════════════════════════════════════
#  전체 기간 백테스트 (Tab 2) — 위대리-H 로직 + 수수료 + 양도세 + 게이트
# ═════════════════════════════════════════════════════════════════════════
TAX_SCHEDULES = {
    'A': {'name': '1월/5월 50/50',  'pattern': [(1, 0.50), (5, 0.50)]},
    'B': {'name': '5월 일괄',       'pattern': [(5, 1.00)]},
    'C': {'name': '1월 일괄',       'pattern': [(1, 1.00)]},
}


def run_full_backtest(data, init_cap=10_000, cash_ratio=0.45,
                      hc=0.05, lc=-0.06, floor=FLOOR_EXPO,
                      start_date=None,
                      apply_commission=False, comm_buy=0.001, comm_sell=0.001,
                      apply_tax=False, tax_deduction_usd=1923.0, tax_rate=0.22,
                      tax_rebalance=True, tax_strategy='B',
                      use_gate=False, gate_sma_days=200):
    wkly = build_weekly_panel(data, gate_sma_days=gate_sma_days)
    if start_date is not None:
        wkly = wkly[wkly['Date'] >= pd.to_datetime(start_date)].reset_index(drop=True)
    if len(wkly) < 2:
        return None
    P    = wkly['TQQQ'].values.astype(float)
    EV   = wkly['Eval'].values.astype(float)
    GATE = wkly['gate_on'].values.astype(bool)
    GROWTH_arr = wkly['Growth'].values.astype(float) if 'Growth' in wkly.columns else np.full(len(wkly), np.nan)
    QQQ_arr = wkly['QQQ'].values.astype(float)
    dates = wkly['Date'].values
    N = len(wkly)
    span_days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
    YEARS = max(span_days / 365.25, N / 52)

    cash = float(init_cap * cash_ratio)
    shares = int(init_cap * (1 - cash_ratio) / P[0])
    init_buy_cost = shares * P[0] * (1 + (comm_buy if apply_commission else 0))
    cash = init_cap - init_buy_cost
    total_cost_basis = init_buy_cost

    eq = np.empty(N); tiers = []; eq[0] = cash + shares * P[0]
    realized_gain_year = {}; cum_commission = 0.0; cum_tax = 0.0
    tax_events = []; year_of_last_bar = pd.Timestamp(dates[0]).year
    pending_payments = []
    schedule_pattern = TAX_SCHEDULES.get(tax_strategy, TAX_SCHEDULES['B'])['pattern']
    trade_log = []; n_floor_block = 0

    def _sell(qty, price, year, mark=True):
        nonlocal shares, cash, total_cost_basis, cum_commission
        if qty <= 0 or shares <= 0:
            return 0.0
        qty = min(qty, shares)
        gross = qty * price; fee = gross * comm_sell if apply_commission else 0.0
        net_proc = gross - fee; prop = qty / shares
        cost_of_sold = total_cost_basis * prop; gain = net_proc - cost_of_sold
        total_cost_basis -= cost_of_sold; shares -= qty; cash += net_proc; cum_commission += fee
        if mark:
            realized_gain_year[year] = realized_gain_year.get(year, 0.0) + gain
        return gain

    def _buy(qty, price):
        nonlocal shares, cash, total_cost_basis, cum_commission
        if qty <= 0:
            return
        gross = qty * price; fee = gross * comm_buy if apply_commission else 0.0
        cost = gross + fee
        if cost > cash:
            qty = int(cash / (price * (1 + (comm_buy if apply_commission else 0))))
            if qty <= 0:
                return
            gross = qty * price; fee = gross * comm_buy if apply_commission else 0.0; cost = gross + fee
        total_cost_basis += cost; shares += qty; cash -= cost; cum_commission += fee

    for i in range(N):
        p = P[i]; ev = EV[i]; date_t = pd.Timestamp(dates[i]); cur_year = date_t.year
        tier = 'HIGH' if ev >= hc else ('LOW' if ev <= lc else 'MID')
        tiers.append(tier)
        if cur_year > year_of_last_bar and apply_tax:
            prev_year = year_of_last_bar; prev_gain = realized_gain_year.get(prev_year, 0.0)
            taxable = max(0.0, prev_gain - tax_deduction_usd); total_tax_due = taxable * tax_rate
            if total_tax_due > 0:
                for (pm, frac) in schedule_pattern:
                    pending_payments.append({'pay_year': cur_year, 'pay_month': pm,
                                             'amount': total_tax_due * frac, 'orig_amount': total_tax_due * frac,
                                             'prev_year': prev_year, 'prev_year_gain': prev_gain, 'taxable': taxable})
        if cur_year > year_of_last_bar:
            year_of_last_bar = cur_year

        action_label = "관망"; qty_signed = 0; realized_pnl_this = 0.0
        eff_rate = 0.0; comm_this = 0.0; diff_val = 0.0; note = ''; floor_hit = 0
        prev_p = float(P[i-1]) if i > 0 else float('nan')
        gate_on_now = (not use_gate) or bool(GATE[i])

        if use_gate and not gate_on_now:
            if shares > 0:
                sold = shares; cb = cum_commission
                realized_pnl_this = _sell(sold, p, cur_year, True); comm_this = cum_commission - cb
                action_label = "매도"; qty_signed = -sold; note = "게이트청산"
            else:
                note = "게이트대기"
        elif use_gate and gate_on_now and shares == 0 and cash > 0:
            budget = cash * (1 - cash_ratio)
            qty = int(budget / (p * (1 + comm_buy))) if apply_commission else int(budget / p)
            if qty > 0:
                sb = shares; cb = cum_commission; _buy(qty, p); comm_this = cum_commission - cb
                qty_signed = shares - sb
                if qty_signed > 0:
                    action_label = "매수"; note = "게이트재진입"
        elif i > 0:
            diff = shares * (p - P[i-1]); diff_val = float(diff)
            total = cash + shares * p
            if diff > 0:   # 매도
                rate = SELL_RATE[tier]; eff_rate = rate
                qty = int(min(round(diff * rate / p), shares))
                if floor > 0 and total > 0:   # ★ 노출 하한
                    min_sh = int(floor * total / p)
                    qty2 = max(0, min(qty, shares - min_sh))
                    if qty2 < qty:
                        floor_hit = 1; n_floor_block += 1
                    qty = qty2
                if qty > 0:
                    cb = cum_commission
                    realized_pnl_this = _sell(qty, p, cur_year, True); comm_this = cum_commission - cb
                    action_label = "매도"; qty_signed = -qty
            elif diff < 0:  # 매수
                rate = BUY_RATE[tier]; eff_rate = rate
                budget = min(cash, abs(diff) * rate)
                qty = int(budget / (p * (1 + comm_buy))) if apply_commission else int(budget / p)
                if qty > 0:
                    sb = shares; cb = cum_commission; _buy(qty, p); comm_this = cum_commission - cb
                    qa = shares - sb
                    if qa > 0:
                        action_label = "매수"; qty_signed = qa

        avg_cost = (total_cost_basis / shares) if shares > 0 else 0.0
        total_asset = cash + shares * p
        growth_val = float(GROWTH_arr[i]) if not np.isnan(GROWTH_arr[i]) else float('nan')
        act_code = {'매수': 'BUY', '매도': 'SELL', '관망': 'HOLD'}[action_label]
        trade_log.append({
            'date': date_t.strftime('%Y-%m-%d'), 'week': int(i), 'action': act_code,
            'tier': tier, 'eval': float(ev), 'qqq_close': float(QQQ_arr[i]),
            'qqq_growth': growth_val, 'gate_on': int(bool(GATE[i])),
            'tqqq_close': float(p), 'tqqq_prev': prev_p,
            'tqqq_wk_ret': (float(p) / prev_p - 1.0) if (i > 0 and prev_p > 0) else float('nan'),
            'signal_diff': diff_val, 'rate': float(eff_rate), 'floor_hit': int(floor_hit),
            'trade_price': float(p) if action_label != "관망" else float('nan'),
            'buy_qty': int(qty_signed) if action_label == "매수" else 0,
            'sell_qty': int(-qty_signed) if action_label == "매도" else 0,
            'realized_pnl': float(realized_pnl_this), 'commission': float(comm_this), 'tax_paid': 0.0,
            'avg_cost': float(avg_cost), 'shares': int(shares), 'cash': float(cash),
            'stock_value': float(shares * p), 'total_asset': float(total_asset),
            'expo': float(shares * p / total_asset) if total_asset > 0 else 0.0,
            'cum_return': float(total_asset / init_cap - 1.0), 'note': note,
        })

        if apply_tax and pending_payments:
            cur_month = date_t.month; still = []
            for pay in pending_payments:
                due = (cur_year > pay['pay_year']) or (cur_year == pay['pay_year'] and cur_month >= pay['pay_month'])
                if not due:
                    still.append(pay); continue
                amount = pay['amount']; sold_for_tax = 0
                if cash < amount:
                    deficit = amount - cash; eff_p = p * (1 - comm_sell) if apply_commission else p
                    sq = int(np.ceil(deficit / eff_p)) if eff_p > 0 else 0; sq = min(sq, shares)
                    sold_for_tax = sq
                    if sq > 0:
                        _sell(sq, p, cur_year, True)
                paid = min(cash, amount); cash = max(0.0, cash - amount); cum_tax += paid
                trade_log.append({
                    'date': date_t.strftime('%Y-%m-%d'), 'week': int(i), 'action': 'TAX',
                    'tier': tier, 'eval': float(ev), 'qqq_close': float(QQQ_arr[i]),
                    'qqq_growth': growth_val, 'gate_on': int(bool(GATE[i])),
                    'tqqq_close': float(p), 'tqqq_prev': float('nan'), 'tqqq_wk_ret': float('nan'),
                    'signal_diff': 0.0, 'rate': 0.0, 'floor_hit': 0,
                    'trade_price': float(p) if sold_for_tax > 0 else float('nan'),
                    'buy_qty': 0, 'sell_qty': int(sold_for_tax) if sold_for_tax > 0 else 0,
                    'realized_pnl': 0.0, 'commission': 0.0, 'tax_paid': float(paid),
                    'avg_cost': (total_cost_basis / shares) if shares > 0 else 0.0,
                    'shares': int(shares), 'cash': float(cash), 'stock_value': float(shares * p),
                    'total_asset': float(cash + shares * p),
                    'expo': float(shares * p / (cash + shares * p)) if (cash + shares * p) > 0 else 0.0,
                    'cum_return': float((cash + shares * p) / init_cap - 1.0),
                    'note': f"{pay['prev_year']}년 분 ({tax_strategy})",
                })
                tax_events.append({'date': date_t.strftime('%Y-%m-%d'), 'prev_year': pay['prev_year'],
                                   'gain': pay['prev_year_gain'], 'taxable': pay['taxable'], 'tax': paid,
                                   'orig_tax': pay['orig_amount'], 'sold_for_tax': sold_for_tax,
                                   'split': f"{pay['orig_amount']:.0f} ({tax_strategy})"})
                if tax_rebalance:
                    remaining = cash + shares * p; target_cash = remaining * cash_ratio
                    if cash < target_cash:
                        d2 = target_cash - cash; eff_p = p * (1 - comm_sell) if apply_commission else p
                        sq = int(np.ceil(d2 / eff_p)) if eff_p > 0 else 0; sq = min(sq, shares)
                        if sq > 0:
                            _sell(sq, p, cur_year, True)
            pending_payments = still
        eq[i] = cash + shares * p

    _peak = None
    for _row in trade_log:
        ta = _row['total_asset']; _peak = ta if _peak is None else max(_peak, ta)
        _row['peak'] = float(_peak); _row['drawdown'] = float(ta / _peak - 1.0) if _peak and _peak > 0 else 0.0

    cagr = (eq[-1] / init_cap) ** (1 / YEARS) - 1
    peak = np.maximum.accumulate(eq); dd = eq / peak - 1; mdd = dd.min()
    rets = np.diff(eq) / eq[:-1]
    if len(rets) == 0:
        rets = np.array([0.0])
    neg = rets[rets < 0]; dstd = neg.std() * np.sqrt(52) if len(neg) > 1 else 1e-9
    ann_r = (1 + rets.mean()) ** 52 - 1; sor = ann_r / dstd if dstd > 0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    bh_peak = np.maximum.accumulate(P)
    expo_series = np.array([r['expo'] for r in trade_log])

    yearly_df = pd.DataFrame({'date': pd.to_datetime(dates), 'eq': eq})
    yearly_df['year'] = yearly_df['date'].dt.year
    yearly_rows = []; prev_end = init_cap
    for yr, grp in yearly_df.groupby('year'):
        start_eq = prev_end; end_eq = float(grp['eq'].iloc[-1])
        ret_pct = (end_eq / start_eq - 1) * 100 if start_eq > 0 else 0
        yr_peak = np.maximum.accumulate(grp['eq'].values); yr_mdd = float((grp['eq'].values / yr_peak - 1).min()) * 100
        yr_realized = realized_gain_year.get(int(yr), 0.0)
        yr_tax_paid = sum(t['tax'] for t in tax_events if t['prev_year'] == int(yr))
        yearly_rows.append({'연도': int(yr), '수익률': f"{ret_pct:+.1f}%", '연간 MDD': f"{yr_mdd:.1f}%",
                            '실현이익': f"${yr_realized:,.0f}", '양도세(다음해 정산)': f"${yr_tax_paid:,.0f}",
                            '기말 자산': f"${end_eq:,.0f}"})
        prev_end = end_eq
    return {
        'cagr': cagr, 'mdd': mdd, 'cal': cal, 'sor': sor, 'final': eq[-1], 'init': init_cap, 'years': YEARS,
        'eq': eq, 'dd': dd, 'dates': dates, 'tiers': tiers, 'prices': P, 'eval': EV,
        'expo_series': expo_series, 'avg_expo': float(expo_series.mean()), 'cur_expo': float(expo_series[-1]),
        'n_floor_block': n_floor_block,
        'bh_eq': P / P[0] * init_cap, 'bh_cagr': (P[-1] / P[0]) ** (1 / YEARS) - 1,
        'bh_mdd': (P / bh_peak - 1).min(), 'rets': rets, 'yearly': yearly_rows,
        'cum_commission': cum_commission, 'cum_tax': cum_tax, 'tax_events': tax_events,
        'realized_gain_year': realized_gain_year, 'tax_strategy': tax_strategy,
        'unpaid_tax': sum(p['amount'] for p in pending_payments), 'trade_log': trade_log,
    }


# ─────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("위대리-H 기본 설정")
    with st.container(border=True):
        st_start      = st.date_input("투자 시작일", key='p_start')
        st_cap        = st.number_input("시작 원금 ($)", step=1000.0, key='p_cap')
        st_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, key='p_cash') / 100
        st_floor      = st.slider("노출 하한 (%)", 0, 40, key='p_floor',
                                  help="위대리-H 핵심. 노출이 이 밑으로 내려가면 매도 중단. "
                                       "기본 15% (경화 방지 검증값). 0으로 두면 원작자 v1.0과 동일.") / 100
        col_ref, col_save = st.columns(2)
        with col_ref:
            if st.button("데이터 갱신", use_container_width=True):
                st.cache_data.clear(); st.rerun()
        with col_save:
            if st.button("설정 저장", use_container_width=True):
                cfg = save_config(st.session_state)
                ok, err = save_config_to_sheets(cfg)
                if ok:
                    st.toast("저장 + Sheets 동기화 완료!", icon="💾")
                else:
                    st.toast("로컬 저장 완료", icon="💾")
    st.divider()
    _gc, _gc_err = _get_gspread_client()
    if _gc:
        st.success("Sheets 동기화 활성화", icon="✅")
    else:
        st.error("Sheets 미연결 — 봇 자동 반영 불가", icon="🔴")
    st.divider()
    st_use_live = st.checkbox(
        "실시간 가격 미리보기", value=bool(st.session_state.get('p_use_live', False)),
        key='p_use_live',
        help="끄면(기본) 지난 금요일 종가 기준 — 봇의 확정 신호/주문표와 정확히 일치합니다. "
             "켜면 yfinance 실시간가로 장중 미리보기 (봇과 시점이 달라 주문표가 다를 수 있음).")
    if not st.session_state.get('p_use_live', False):
        st.caption("📅 지난 금요일 종가 기준 · 봇과 동일")
    else:
        st.caption("🔴 실시간가 · 장중 미리보기 (봇과 다를 수 있음)")
    st.divider()
    st_use_gate = st.checkbox(
        "200일 추세 게이트 (재난 보험)", value=bool(st.session_state.get('p_use_gate', False)),
        key='p_use_gate',
        help="QQQ가 200일 SMA 아래로 이탈하면 전량 현금화 후 재진입 대기. "
             "닷컴/2008/2022 레짐 붕괴 꼬리 절단용 선택적 보험(강세장 CAGR 반납).")
    st.divider()
    _hc_d = st.session_state.get('p_hc', 5.0); _lc_d = st.session_state.get('p_lc', -6.0)
    _fl_d = st.session_state.get('p_floor', 15.0)
    st.markdown(f"""
**위대리-H 매매율 (원작자 v1.0 %)**
| 티어 | Eval | 매도% | 매수% |
|------|------|-------|-------|
| HIGH | ≥ +{_hc_d:.0f}% | 100 | 50 |
| MID  | 중간 | 50 | 50 |
| LOW  | ≤ {_lc_d:.0f}% | 33.3 | 100 |

**노출 하한: {_fl_d:.0f}%** (그 밑으로는 매도 안 함)
""")

# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
with st.spinner("데이터 로딩 중 (최초 1회만 시간이 걸립니다)..."):
    df = load_wedaeri_data()
df = inject_live_price(df, use_live=st.session_state.get('p_use_live', False))
if df.empty:
    st.error("데이터 로드 실패. 잠시 후 새로고침 해주세요."); st.stop()

_ss = st.session_state
_cash_adjs, _cash_adj_err = load_cash_adjustments()
log_df = run_wedaeri_sim(
    df, st_start, st_cap, st_cash_ratio,
    hc=_ss.get('p_hc', 5.0) / 100, lc=_ss.get('p_lc', -6.0) / 100,
    floor=_ss.get('p_floor', 15.0) / 100,
    use_gate=_ss.get('p_use_gate', False),
    cash_adjustments=_cash_adjs,
)
tqqq_series = df['TQQQ'].dropna()
latest_tqqq = float(tqqq_series.iloc[-1]) if not tqqq_series.empty else 0.0
eval_ser  = df['Eval'].dropna()
latest_eval  = float(eval_ser.iloc[-1]) if not eval_ser.empty else 0.0
_hc_rt = _ss.get('p_hc', 5.0) / 100; _lc_rt = _ss.get('p_lc', -6.0) / 100
latest_tier  = 'HIGH' if latest_eval >= _hc_rt else ('LOW' if latest_eval <= _lc_rt else 'MID')

# 위대리-H 현재 총자산 (포트폴리오 탭에서 사용)
wedaeri_total = float(log_df.iloc[-1]['총자산']) if not log_df.empty else 0.0

_tier_label = {'HIGH': '고평가', 'MID': '중립', 'LOW': '저평가'}[latest_tier]
_tier_color = {'HIGH': C_AMBER, 'MID': C_TEAL_GLOW, 'LOW': C_GOOD}[latest_tier]
_basis_date = df['Date'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else '—'
_is_live = _ss.get('p_use_live', False)
_basis_txt = (f"🔴 실시간 · 미리보기" if _is_live
              else f"📅 {_basis_date} 종가 · 봇과 동일")
st.markdown(f"""
<div class="hero">
<div class="hero-eyebrow">Wedaeri-H · Navigation Console · vH1.0</div>
<div class="hero-title">썰물을 기다리는 <span class="h">심해의 매집기</span></div>
<div class="hero-sub">
QQQ 5년 추세선 대비 저평가 구간에서 TQQQ를 사 모으고, 고평가에서 덜어낸다.
노출 하한 15%로 절대 잠들지 않으며, 폭락이 오면 쌓아둔 현금으로 반격한다 —
동파공의 <b style="color:#4dd0e1">헤지 슬리브</b>로 설계된 역추세 엔진.
</div>
<div class="hero-chips">
<span class="chip">{_basis_txt}</span>
<span class="chip">현재 평가 <b style="color:{_tier_color}">{latest_eval:+.2%}</b> · {_tier_label}</span>
<span class="chip">TQQQ <b>${latest_tqqq:.2f}</b></span>
<span class="chip">매도 100/50/33 · 매수 50/50/100</span>
<span class="chip">노출 하한 <b>15%</b></span>
<span class="chip">권장 배분 <b>동파공 7 : 위대리-H 3</b></span>
</div>
</div>
""", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["◆ 실전 트레이딩", "◆ 백테스트 분석", "◆ 전략 로직", "◆ 포트폴리오 7:3"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — 실전 트레이딩
# ═══════════════════════════════════════════════════════════════
with tab1:
    if log_df.empty:
        st.warning("매매 데이터가 없습니다. 시작일 또는 설정을 확인해 주세요."); st.stop()
    last = log_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("시장 모드", last['티어'], last['시장평가'])
    c2.metric("TQQQ 현재가", f"${latest_tqqq:.2f}")
    c3.metric("현재 총 자산", f"${last['총자산']:,.2f}")
    if _ss.get('p_use_gate', False):
        c4.metric("추세 게이트", last['게이트'])
    else:
        c4.metric("현재 노출", last['노출'])

    if _ss.get('p_use_gate', False) and last['게이트'] == '현금(이탈)':
        st.error("**추세 게이트 OFF — QQQ가 200일선 아래.** 전량 현금 대기 중입니다.", icon="🛡️")

    sec(f"금주 장 마감(LOC) 주문표 · {datetime.now().strftime('%Y-%m-%d')}", "Weekly Order")
    b_col, s_col = st.columns(2)
    is_buy = last['액션'] in ("매수", "게이트재진입")
    is_sell = last['액션'] in ("매도", "게이트청산")
    with b_col:
        if is_buy:
            st.markdown(f'<div class="order-card-buy"><h4>▼ LOC 매수</h4>'
                        f'<div class="big">{last["주문수량"]} 주</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="order-card-hold"><h4>▼ LOC 매수</h4>'
                        f'<div class="big">—</div></div>', unsafe_allow_html=True)
    with s_col:
        if is_sell:
            st.markdown(f'<div class="order-card-sell"><h4>▲ LOC 매도</h4>'
                        f'<div class="big">{abs(last["주문수량"])} 주</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="order-card-hold"><h4>▲ LOC 매도</h4>'
                        f'<div class="big">—</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="floor-badge">🛟 <b>노출 하한 {_ss.get("p_floor",15):.0f}%</b> 작동 중 — '
                f'현재 노출 <b>{last["노출"]}</b>. 노출이 하한 밑이면 매도 신호가 나와도 주식을 유지합니다 '
                f'(경화 방지 · 폭락 시 반격 실탄 확보).</div>', unsafe_allow_html=True)
    st.divider()

    # ── 시그니처: 조석 게이지 (Eval 온도계) + Eval 히스토리 ──
    sec("조석 게이지 · 추세선 대비 현재 수위", "Tide Gauge")
    _hc_v = _ss.get('p_hc', 5.0); _lc_v = _ss.get('p_lc', -6.0)
    gcol1, gcol2 = st.columns([1, 1.4])
    with gcol1:
        # 게이지: LOW(저평가/매집) ← 0 → HIGH(고평가/차익)
        gval = latest_eval * 100
        gfig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gval,
            number={'suffix': '%', 'font': {'size': 34, 'family': 'Space Grotesk', 'color': _tier_color}},
            delta={'reference': 0, 'increasing': {'color': C_AMBER}, 'decreasing': {'color': C_GOOD}},
            title={'text': f"<span style='font-family:IBM Plex Mono;font-size:12px;color:#7b98a3'>"
                           f"현재 {_tier_label} · {latest_tier}</span>"},
            gauge={
                'axis': {'range': [-20, 20], 'tickcolor': '#3a5560',
                         'tickfont': {'color': '#7b98a3', 'size': 9, 'family': 'IBM Plex Mono'}},
                'bar': {'color': _tier_color, 'thickness': 0.28},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [-20, _lc_v], 'color': 'rgba(63,179,127,0.28)'},   # LOW 매집
                    {'range': [_lc_v, _hc_v], 'color': 'rgba(43,157,184,0.16)'}, # MID
                    {'range': [_hc_v, 20], 'color': 'rgba(217,154,28,0.28)'},    # HIGH 차익
                ],
                'threshold': {'line': {'color': _tier_color, 'width': 3}, 'thickness': 0.8, 'value': gval},
            }))
        gfig.update_layout(height=230, margin=dict(l=20, r=20, t=44, b=0),
                           paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#dce8ec'))
        st.plotly_chart(gfig, use_container_width=True)
        st.markdown(f"<div style='text-align:center;font-family:IBM Plex Mono;font-size:11px;color:#7b98a3'>"
                    f"🟢 저평가(≤{_lc_v:.0f}%) 매집 &nbsp;·&nbsp; 🔵 중립 &nbsp;·&nbsp; 🟡 고평가(≥{_hc_v:.0f}%) 차익</div>",
                    unsafe_allow_html=True)
    with gcol2:
        # Eval 히스토리 (최근 3년) — 밴드 배경 + 현재점
        _hist = df.dropna(subset=['Eval']).copy()
        _hist = _hist[_hist['Date'] >= (_hist['Date'].max() - pd.Timedelta(days=1100))]
        efig = go.Figure()
        efig.add_hrect(y0=_hc_v, y1=25, fillcolor='rgba(217,154,28,0.10)', line_width=0)
        efig.add_hrect(y0=-25, y1=_lc_v, fillcolor='rgba(63,179,127,0.10)', line_width=0)
        efig.add_trace(go.Scatter(x=_hist['Date'], y=_hist['Eval'] * 100, name='Eval',
                                  line=dict(color=C_TEAL_GLOW, width=1.8)))
        efig.add_hline(y=_hc_v, line_color=C_AMBER, line_dash='dot', line_width=1)
        efig.add_hline(y=_lc_v, line_color=C_GOOD, line_dash='dot', line_width=1)
        efig.add_trace(go.Scatter(x=[_hist['Date'].iloc[-1]], y=[latest_eval * 100], mode='markers',
                                  marker=dict(color=_tier_color, size=11, line=dict(color='#eaf6f9', width=1.5)),
                                  name='현재', showlegend=False))
        efig.update_layout(title='Eval 추이 · 최근 3년 (추세선 대비 %)', height=230, showlegend=False,
                           yaxis=dict(title='Eval (%)', ticksuffix='%'), **CHART_LAYOUT)
        apply_grid(efig)
        st.plotly_chart(efig, use_container_width=True)
    st.divider()

    # ── 양도세 납부 관리 ──
    _now = datetime.now(); _cur_year = _now.year; _cur_month = _now.month
    _is_tax_window = (_cur_month in [1, 5]) and (_now.day <= 14)
    _tax_payments, _tax_load_err = load_tax_payments()
    _total_paid_usd = sum(p['amount_usd'] for p in _tax_payments)
    _paid_for_year_curr = sum(p['amount_usd'] for p in _tax_payments if p['for_year'] == _cur_year - 1)
    if _is_tax_window:
        st.warning(f"**양도세 납부 시기 알림** — {_now.strftime('%Y년 %m월')} · "
                   f"{_cur_year-1}년 분으로 기록된 납부액 **${_paid_for_year_curr:,.2f}**", icon="🧾")

    with st.expander("양도세 납부 관리", expanded=_is_tax_window):
        with st.container(border=True):
            cc_a, cc_b = st.columns([2, 1])
            with cc_a:
                _tax_apply_now = st.checkbox("봇의 가상 잔고에 양도세 차감 반영",
                    value=bool(st.session_state.get('p_tax_apply_to_bot', False)), key='p_tax_apply_to_bot')
            with cc_b:
                if st.button("봇에 반영 (저장)", use_container_width=True, key='btn_save_tax_apply'):
                    cfg = save_config(st.session_state); ok, err = save_config_to_sheets(cfg)
                    st.toast("저장 완료" if ok else "로컬 저장 완료", icon="💾")
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown("**새 납부 기록 추가**")
            with st.form("tax_payment_form", clear_on_submit=True):
                p_date = st.date_input("납부일", value=_now.date(), key='p_tax_pay_date')
                p_year = st.number_input("대상 연도", value=_cur_year - 1, min_value=2010,
                                         max_value=_cur_year, step=1, key='p_tax_for_year')
                p_input_mode = st.radio("입력 방식", ["KRW", "USD"], horizontal=True, key='p_tax_input_mode')
                col_amt, col_fx = st.columns(2)
                with col_amt:
                    if p_input_mode == "KRW":
                        p_amount_krw = st.number_input("납부액 (KRW)", value=0, min_value=0, step=10000, key='p_tax_krw')
                    else:
                        p_amount_usd_in = st.number_input("납부액 ($)", value=0.0, min_value=0.0, step=10.0, key='p_tax_usd')
                with col_fx:
                    p_fx = st.number_input("환율 (KRW/USD)", value=1300.0, min_value=500.0, step=10.0, key='p_tax_fx')
                p_note = st.text_input("메모 (선택)", value="", key='p_tax_note')
                submitted = st.form_submit_button("납부 기록 저장", use_container_width=True)
                if submitted:
                    if p_input_mode == "KRW":
                        p_usd = float(p_amount_krw) / p_fx if p_fx > 0 else 0; p_krw = float(p_amount_krw)
                    else:
                        p_usd = float(p_amount_usd_in); p_krw = p_usd * p_fx
                    if p_usd <= 0:
                        st.error("납부액이 0 입니다.")
                    else:
                        ok, err = save_tax_payment({'date': p_date.strftime("%Y-%m-%d"), 'for_year': int(p_year),
                                                    'amount_usd': round(p_usd, 2), 'amount_krw': round(p_krw, 0),
                                                    'fx_rate': p_fx, 'note': p_note})
                        if ok:
                            st.success("저장 완료"); st.rerun()
                        else:
                            st.error(f"저장 실패: {err}")
        with col_p2:
            st.markdown("**납부 기록**")
            if _tax_load_err:
                st.error(f"시트 로드 실패: {_tax_load_err}")
            elif not _tax_payments:
                st.info("아직 기록된 납부가 없습니다.")
            else:
                _df_pay = pd.DataFrame([{'납부일': p['date'], '대상 연도': p['for_year'],
                                         '금액 ($)': f"${p['amount_usd']:,.2f}", '금액 (KRW)': f"{p['amount_krw']:,.0f}",
                                         '환율': f"{p['fx_rate']:.0f}", '메모': p['note']} for p in _tax_payments])
                st.dataframe(_df_pay, use_container_width=True, hide_index=True)
                st.caption(f"누적 납부: **${_total_paid_usd:,.2f}** ({len(_tax_payments)}건)")
                _del_dates = ['(선택 안 함)'] + [p['date'] for p in _tax_payments]
                _del_target = st.selectbox("삭제할 기록", _del_dates, key='p_tax_delete_target')
                if _del_target != '(선택 안 함)' and st.button(f"{_del_target} 삭제", use_container_width=True):
                    ok, err = delete_tax_payment(_del_target)
                    if ok:
                        st.success("삭제 완료"); st.rerun()
                    else:
                        st.error(f"삭제 실패: {err}")

    # ── 계좌 현황 ──
    sec("내 계좌 현황", "Account")
    _tax_apply_now = bool(st.session_state.get('p_tax_apply_to_bot', False))
    _eff_paid = _total_paid_usd if _tax_apply_now else 0.0
    _adj_total = float(last['총자산']) - _eff_paid
    _adj_cash = float(last['현금']) - _eff_paid
    _adj_cash_pct = (_adj_cash / _adj_total * 100 if _adj_total > 0 else 0)
    _adj_pnl_pct = (_adj_total / st_cap - 1) * 100 if st_cap > 0 else 0
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("보유 수량", f"{last['보유수량']:,} 주")
    a2.metric("보유 현금", f"${_adj_cash:,.2f}", f"-양도세 ${_eff_paid:,.0f}" if _eff_paid > 0 else None)
    a3.metric("수익률" + (" (세후)" if _eff_paid > 0 else ""), f"{_adj_pnl_pct:+.2f}%",
              f"세전 {last['수익률']}" if _eff_paid > 0 else None)
    a4.metric("평가 금액", f"${last['보유수량'] * latest_tqqq:,.2f}")
    a5.metric("현금 비중", f"{_adj_cash_pct:.2f}%")

    with st.expander("상세 매매 로그", expanded=True):
        st.dataframe(log_df.iloc[::-1], use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — 백테스트 분석
# ═══════════════════════════════════════════════════════════════
with tab2:
    sec("위대리-H 전체 기간 백테스트", "Backtest Engine")
    st.caption("시작일은 2010-01-01부터 자유롭게. 데이터는 2005년부터 로드해 260주 추세선을 "
               "완전히 워밍업하므로, 2010년 초에 시작해도 Eval이 왜곡되지 않습니다.")
    with st.container(border=True):
        st.markdown("**백테스트 설정**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            bt_cap = st.number_input("초기 자본 ($)", value=10_000, step=1000, key='p_bt_cap')
        with col_b:
            _synth_on = bool(st.session_state.get('p_bt_use_synth', False))
            # 합성 ON → 1999년까지, 합성 OFF → 2010-01-01 부터 (TQQQ 실상장 2010-02는 자동 클립)
            _bt_min = datetime(1999, 3, 10).date() if _synth_on else datetime(2010, 1, 1).date()
            bt_start_date = st.date_input("검증 시작일", value=datetime.strptime("2010-01-01", "%Y-%m-%d").date(),
                                          min_value=_bt_min, max_value=datetime.now().date(), key='p_bt_start')
        with col_c:
            bt_end_date = st.date_input("검증 종료일", value=datetime.now().date(), key='p_bt_end')
        bt_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, value=45, step=1, key='p_bt_cash') / 100
        bt_floor = st.slider("노출 하한 (%)", 0, 40, value=int(st.session_state.get('p_floor', 15)),
                             step=1, key='p_bt_floor',
                             help="위대리-H 핵심. 0=원작자 v1.0, 15=권장(경화 방지)") / 100
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            bt_use_gate = st.checkbox("200일 추세 게이트 적용", value=False, key='p_bt_use_gate')
            bt_compare_gate = st.checkbox("게이트 ON/OFF 동시 비교", value=False, key='p_bt_compare_gate')
        with gcol2:
            bt_use_synth = st.checkbox("2010년 이전 합성 데이터 사용", value=False, key='p_bt_use_synth',
                help="TQQQ 상장 이전을 QQQ 3배 합성으로. 게이트 유무 상대비교 용도.")
        st.caption("전략 파라미터: 위대리-H 고정 (매도 100/50/33.3 · 매수 50/50/100 · HIGH컷+5% · LOW컷-6%)")

    with st.expander("거래비용 & 양도세", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**거래 수수료**")
            apply_comm = st.checkbox("수수료 적용", value=False, key='p_apply_comm')
            comm_buy_pct = st.number_input("매수 수수료 (%)", value=0.1, step=0.01, format="%.3f",
                                           key='p_comm_buy', disabled=not apply_comm)
            comm_sell_pct = st.number_input("매도 수수료 (%)", value=0.1, step=0.01, format="%.3f",
                                            key='p_comm_sell', disabled=not apply_comm)
        with cc2:
            st.markdown("**양도세 (한국 거주자)**")
            apply_tax = st.checkbox("양도세 적용", value=False, key='p_apply_tax')
            tax_dedu = st.number_input("연간 공제 (USD)", value=1923.0, step=100.0, key='p_tax_dedu', disabled=not apply_tax)
            tax_rate_pct = st.number_input("세율 (%)", value=22.0, step=0.5, key='p_tax_rate', disabled=not apply_tax)
            tax_strategy_label = st.selectbox("양도세 인출 전략",
                ["B - 5월 일괄", "A - 1월/5월 50/50", "C - 1월 일괄"], key='p_tax_strategy', disabled=not apply_tax)
            tax_strategy = tax_strategy_label[0]
            tax_rebal = st.checkbox("매도-only 리밸런싱", value=True, key='p_tax_rebal', disabled=not apply_tax)
        compare_costs = st.checkbox("Gross vs Net 비교", value=False, key='p_compare_costs')

    df_source = load_wedaeri_data(extended=True) if bt_use_synth else df
    df_bt = df_source[df_source['Date'] <= pd.to_datetime(bt_end_date)].reset_index(drop=True)
    if bt_use_synth and 'synthetic' in df_source.columns and df_source['synthetic'].any():
        _syn_end = df_source.loc[df_source['synthetic'], 'Date'].max()
        st.info(f"⚠️ 합성 TQQQ 구간 포함 (~{_syn_end.strftime('%Y-%m')}까지). 게이트 유무 상대비교로 해석하세요.", icon="🧪")

    with st.spinner("백테스트 계산 중..."):
        cost_kwargs = dict(apply_commission=apply_comm, comm_buy=comm_buy_pct / 100, comm_sell=comm_sell_pct / 100,
                           apply_tax=apply_tax, tax_deduction_usd=tax_dedu, tax_rate=tax_rate_pct / 100,
                           tax_rebalance=tax_rebal, tax_strategy=tax_strategy)
        gate_kwargs = dict(use_gate=bt_use_gate, gate_sma_days=200)
        bt_cur = run_full_backtest(df_bt, bt_cap, bt_cash_ratio, hc=0.05, lc=-0.06, floor=bt_floor,
                                   start_date=bt_start_date, **cost_kwargs, **gate_kwargs)
        bt_alt = None
        if bt_compare_gate:
            bt_alt = run_full_backtest(df_bt, bt_cap, bt_cash_ratio, hc=0.05, lc=-0.06, floor=bt_floor,
                                       start_date=bt_start_date, **cost_kwargs, use_gate=(not bt_use_gate), gate_sma_days=200)
        bt_gross = None
        if compare_costs and (apply_comm or apply_tax):
            bt_gross = run_full_backtest(df_bt, bt_cap, bt_cash_ratio, hc=0.05, lc=-0.06, floor=bt_floor,
                                         start_date=bt_start_date, apply_commission=False, apply_tax=False, **gate_kwargs)
    if bt_cur is None:
        st.warning("백테스트 데이터가 부족합니다."); st.stop()

    dates = pd.to_datetime(bt_cur['dates']); ev = bt_cur['eval']
    n_h = int((ev >= 0.05).sum()); n_l = int((ev <= -0.06).sum()); n_m = len(ev) - n_h - n_l
    with st.expander("Eval 진단 — 티어 배분", expanded=False):
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("중앙값", f"{float(np.median(ev)):.2%}")
        d2.metric("p10", f"{float(np.percentile(ev,10)):.2%}")
        d3.metric("p90", f"{float(np.percentile(ev,90)):.2%}")
        d4.metric("HIGH 비중", f"{n_h/len(ev):.1%}")
        d5.metric("LOW 비중", f"{n_l/len(ev):.1%}")

    sec("성과 지표", "Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("CAGR", f"{bt_cur['cagr']:.2%}")
    m2.metric("MDD", f"{bt_cur['mdd']:.2%}")
    m3.metric("Calmar", f"{bt_cur['cal']:.2f}")
    m4.metric("Sortino", f"{bt_cur['sor']:.2f}")
    m5.metric("평균 노출", f"{bt_cur['avg_expo']*100:.0f}%")
    m6.metric("현재 노출", f"{bt_cur['cur_expo']*100:.1f}%", "하한 작동" if bt_floor > 0 else None)
    st.caption(f"기간: {dates[0].strftime('%Y.%m.%d')} ~ {dates[-1].strftime('%Y.%m.%d')} "
               f"({bt_cur['years']:.1f}년) | 노출 하한 {bt_floor*100:.0f}% | "
               f"하한 발동 {bt_cur['n_floor_block']}회 | 게이트 {'ON' if bt_use_gate else 'OFF'} / "
               f"수수료 {'ON' if apply_comm else 'OFF'} / 양도세 {'ON' if apply_tax else 'OFF'}")

    if bt_floor > 0:
        st.info(f"🛟 노출 하한 {bt_floor*100:.0f}% 가 **{bt_cur['n_floor_block']}회** 매도를 차단해 경화를 막았습니다. "
                f"현재 노출 {bt_cur['cur_expo']*100:.1f}% (하한 없으면 3~4%로 굳음). "
                f"위대리-H는 단독보다 **동파공 70 / 위대리-H 30** 헤지 슬리브로 쓸 때 진가를 발휘합니다 "
                f"(세후 Calmar 1.12).", icon="🛟")

    if bt_compare_gate and bt_alt is not None:
        sec("게이트 ON/OFF 비교", "Gate Compare")
        _on = bt_cur if bt_use_gate else bt_alt; _off = bt_alt if bt_use_gate else bt_cur
        cmp_g = pd.DataFrame({'구분': ['게이트 ON', '게이트 OFF', '차이(ON-OFF)'],
            'CAGR': [f"{_on['cagr']:.2%}", f"{_off['cagr']:.2%}", f"{(_on['cagr']-_off['cagr'])*100:+.2f}%p"],
            'MDD': [f"{_on['mdd']:.2%}", f"{_off['mdd']:.2%}", f"{(_on['mdd']-_off['mdd'])*100:+.2f}%p"],
            'Calmar': [f"{_on['cal']:.2f}", f"{_off['cal']:.2f}", f"{_on['cal']-_off['cal']:+.2f}"],
            '최종 자산': [f"${_on['final']:,.0f}", f"${_off['final']:,.0f}", f"${_on['final']-_off['final']:+,.0f}"]})
        st.dataframe(cmp_g, use_container_width=True, hide_index=True)

    if apply_comm or apply_tax:
        sec("거래비용 & 양도세 요약", "Costs & Tax")
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("누적 수수료", f"${bt_cur['cum_commission']:,.0f}")
        cs2.metric("누적 양도세", f"${bt_cur['cum_tax']:,.0f}")
        total_cost = bt_cur['cum_commission'] + bt_cur['cum_tax']
        cs3.metric("총 비용 합계", f"${total_cost:,.0f}")
        cs4.metric("최종 자산 대비", f"{total_cost / max(bt_cur['final'], 1) * 100:.2f}%")
        if bt_gross is not None:
            gn = pd.DataFrame({'구분': ['Gross', 'Net (현재)', '차이'],
                'CAGR': [f"{bt_gross['cagr']:.2%}", f"{bt_cur['cagr']:.2%}", f"{(bt_cur['cagr']-bt_gross['cagr'])*100:+.2f}%p"],
                'MDD': [f"{bt_gross['mdd']:.2%}", f"{bt_cur['mdd']:.2%}", f"{(bt_cur['mdd']-bt_gross['mdd'])*100:+.2f}%p"],
                'Calmar': [f"{bt_gross['cal']:.2f}", f"{bt_cur['cal']:.2f}", f"{bt_cur['cal']-bt_gross['cal']:+.2f}"]})
            st.dataframe(gn, use_container_width=True, hide_index=True)

    sec("성과 비교 · 위대리-H vs Buy&Hold", "Benchmark")
    compare = pd.DataFrame({'전략': ['위대리-H', 'TQQQ B&H'],
        'CAGR': [f"{bt_cur['cagr']:.2%}", f"{bt_cur['bh_cagr']:.2%}"],
        'MDD': [f"{bt_cur['mdd']:.2%}", f"{bt_cur['bh_mdd']:.2%}"],
        'Calmar': [f"{bt_cur['cal']:.2f}", "—"], 'Sortino': [f"{bt_cur['sor']:.2f}", "—"],
        '최종 자산': [f"${bt_cur['final']:,.0f}", f"${bt_cur['bh_eq'][-1]:,.0f}"]})
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.divider()
    sec("누적 자산 곡선 · 로그 스케일", "Equity Curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=dates, y=bt_cur['eq'], name='위대리-H' + (' (게이트 ON)' if bt_use_gate else ''),
                                line=dict(color=C_TEAL_GLOW, width=2.6),
                                fill='tozeroy', fillcolor='rgba(77,208,225,0.06)'))
    fig_eq.add_trace(go.Scatter(x=dates, y=bt_cur['bh_eq'], name='TQQQ B&H',
                                line=dict(color=C_BH, width=1.4, dash='dash'), opacity=0.7))
    if bt_compare_gate and bt_alt is not None:
        fig_eq.add_trace(go.Scatter(x=dates, y=bt_alt['eq'],
                                    name='위대리-H (게이트 ' + ('OFF' if bt_use_gate else 'ON') + ')',
                                    line=dict(color=C_GOOD, width=1.5, dash='dot'), opacity=0.8))
    fig_eq.add_hline(y=bt_cap, line_color='#3a5560', line_dash='dot', line_width=1,
                     annotation_text=f"초기자본 ${bt_cap:,}", annotation_font=dict(color='#7b98a3', size=10))
    fig_eq.update_layout(title='전략별 자산 추이', yaxis=dict(title='자산 ($)', type='log'), height=400, **CHART_LAYOUT)
    apply_grid(fig_eq); st.plotly_chart(fig_eq, use_container_width=True)

    sec("주식 노출 추이 · 경화 방지 확인", "Exposure")
    fig_ex = go.Figure()
    fig_ex.add_trace(go.Scatter(x=dates, y=bt_cur['expo_series'] * 100, name='주식 노출',
                                line=dict(color=C_TEAL, width=1.7), fill='tozeroy', fillcolor='rgba(43,157,184,0.14)'))
    fig_ex.add_hline(y=bt_floor * 100, line_color=C_AMBER, line_dash='dash', line_width=1.6,
                     annotation_text=f"노출 하한 {bt_floor*100:.0f}%", annotation_font=dict(color=C_AMBER, size=10))
    fig_ex.update_layout(title='주식 노출 % (하한 아래로 내려가지 않음)', yaxis=dict(title='노출 (%)'), height=260, **CHART_LAYOUT)
    apply_grid(fig_ex); st.plotly_chart(fig_ex, use_container_width=True)

    sec("연도별 상세 성과", "Yearly")
    if bt_cur.get('yearly'):
        st.dataframe(pd.DataFrame(bt_cur['yearly']), use_container_width=True, hide_index=True)

    if bt_cur.get('trade_log'):
        sec("전체 매매 로그", "Trade Log")
        raw_df = pd.DataFrame(bt_cur['trade_log'])
        n_buy = int((raw_df['action'] == 'BUY').sum()); n_sell = int((raw_df['action'] == 'SELL').sum())
        n_hold = int((raw_df['action'] == 'HOLD').sum()); n_tax = int((raw_df['action'] == 'TAX').sum())
        st.caption(f"총 {len(raw_df):,}행 (매수 {n_buy} / 매도 {n_sell} / 관망 {n_hold}"
                   f"{' / 양도세 ' + str(n_tax) if n_tax else ''}) · 하한 발동 {bt_cur['n_floor_block']}회")
        show_raw = st.checkbox("원본 숫자값", value=False, key='p_log_raw')
        if show_raw:
            st.dataframe(raw_df.iloc[::-1], use_container_width=True, hide_index=True, height=500)
        else:
            sub = raw_df[raw_df['action'].isin(['BUY', 'SELL', 'TAX'])].iloc[::-1]
            st.dataframe(sub, use_container_width=True, hide_index=True, height=500)
        csv = raw_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("매매 로그 CSV 다운로드", data=csv,
                           file_name=f"wedaeri_h_log_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')


# ═══════════════════════════════════════════════════════════════
# TAB 3 — 전략 로직
# ═══════════════════════════════════════════════════════════════
with tab3:
    # ── 전략 로직 히어로 ──
    st.markdown("""
<div class="hero" style="margin-bottom:20px">
<div class="hero-eyebrow">Strategy Deep-Dive · 전략 완전 해설</div>
<div class="hero-title" style="font-size:2rem">역추세 매집기의 <span class="h">해부도</span></div>
<div class="hero-sub">
왜 이 전략이 존재하는가, 어떻게 사고파는가, 무엇이 원본과 다른가,
그리고 왜 혼자 쓰면 안 되고 동파공과 짝을 이뤄야 하는가 —
위대리-H의 모든 설계 결정을 근거와 함께 풀어냅니다.
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="callout">
<span class="k">한 줄 정의</span>
<b>위대리-H</b>는 "가격은 결국 장기 추세선으로 되돌아온다"는 <b>평균회귀</b>에 베팅하는
역추세 매집 전략입니다. 나스닥100(QQQ)의 5년 추세선보다 쌀 때 3배 레버리지 ETF(TQQQ)를
사 모으고, 비쌀 때 덜어냅니다. 원작자 위대리 v1.0을 기반으로 <b>노출 하한 15%</b>를 더해,
강세장에서 스스로 현금에 갇히는 문제(경화)를 고친 <b>헤지 슬리브 전용</b> 버전입니다.
</div>
""", unsafe_allow_html=True)

    # ═══ 1. 왜 존재하는가 ═══
    sec("1 · 왜 이 전략이 존재하는가", "Rationale")
    st.markdown("""
레버리지 ETF(TQQQ 등)는 장기 보유가 어렵습니다. 하락 후 반등해도 원금을 회복하지 못하는
**변동성 감쇠(volatility drag)** 때문입니다. 예를 들어 −50% 후 +100%가 와야 본전인데,
3배 상품은 이 비대칭이 훨씬 가혹합니다. 그래서 "그냥 사서 들고 있기(Buy&Hold)"는
TQQQ에서 −80%대 낙폭을 감수해야 합니다.

위대리는 이 문제를 **역추세 매매**로 우회합니다. 쌀 때 사서 비쌀 때 파는 규칙을 기계적으로
반복하면, 변동성이 오히려 매매 기회가 됩니다. 원작자의 설계 목표는 명확합니다 —
*"TQQQ를 이용해 수익률은 QLD(2배) 수준, 낙폭은 QQQ(1배) 수준으로."* 즉 **3배의 수익 잠재력을
누리되 위험은 1~2배 수준으로 억제**하는 것이 목표입니다.
    """)
    st.markdown("""
| 동일 기간(2010~) Buy&Hold 비교 | CAGR | MDD |
|---|---|---|
| QQQ (1배) | 18.1% | −32.6% |
| QLD (2배) | 28.5% | −63.7% |
| TQQQ (3배) | 42.3% | **−81.8%** |
| **위대리 (원작자 목표)** | **30.9%** | **−30.9%** |

*원작자 문서 기준. 위대리의 핵심 가치는 CAGR이 아니라 **낙폭을 QQQ 수준으로 낮춘 것**입니다.*
    """)

    # ═══ 2. 매매 규칙 ═══
    sec("2 · 어떻게 사고파는가", "Trading Rules")
    st.markdown("""
매주 금요일 종가(LOC) 기준으로 **딱 한 번** 매매합니다. 판단은 두 단계입니다.

**① 지금 시장이 싼가 비싼가? — Eval 온도계**
QQQ의 최근 5년(260주) 종가에 로그-선형 회귀를 그어 **장기 추세선**을 만듭니다.
현재 QQQ가 그 추세선보다 얼마나 위/아래에 있는지가 **Eval**입니다.

`Eval = 현재 QQQ ÷ 추세선 − 1`

- Eval **≥ +5%** → 추세선보다 5% 이상 비쌈 → **HIGH(고평가)**
- **−6% ~ +5%** → 추세선 근처 → **MID(중립)**
- Eval **≤ −6%** → 추세선보다 6% 이상 쌈 → **LOW(저평가)**

**② 얼마나 사고팔 것인가? — 등락분의 %**
이번 주 TQQQ 평가금이 지난주보다 오르면 그 **상승분의 일정 %를 매도**, 내리면
**하락분의 일정 %를 매수**합니다. 그 %가 티어에 따라 달라집니다.
    """)
    st.markdown("""
| 구간 | Eval 밴드 | 매도율 (상승분의 %) | 매수율 (하락분의 %) | 의미 |
|---|---|---|---|---|
| 🟡 HIGH | ≥ +5% | **100%** | 50% | 비싸니 많이 팔고 적게 산다 |
| 🔵 MID | +5% ~ −6% | 50% | 50% | 중립, 대칭 |
| 🟢 LOW | ≤ −6% | 33.3% | **100%** | 싸니 적게 팔고 많이 산다 |
    """)
    st.markdown("""
**예시 (LOW 구간, 매수율 100%)**: 지난주 TQQQ 평가금 $10,000 → 이번주 $9,000 (−$1,000).
하락분 $1,000의 100% = $1,000어치를 추가 매수. 쌀 때 공격적으로 담는 것입니다.

> 매수는 항상 **보유 현금 한도 내**에서만 실행됩니다. 현금이 마르면 신호가 있어도 못 삽니다.
> 그래서 이 전략은 **현금 비중 관리**가 곧 실탄 관리입니다.
    """)

    # ═══ 3. 노출 하한 (핵심) ═══
    sec("3 · 노출 하한 15% — 위대리-H의 심장", "The Floor")
    _fl = _ss.get('p_floor', 15.0)
    st.markdown(f"""
원작자 위대리에는 치명적인 약점이 있었습니다. **강세장이 길어지면 스스로 현금에 갇힙니다(경화).**

작동 원리를 보면: 강세장에서 계속 팔다 보니 → 보유 주식이 줄고 → 그런데 매도량은
'보유 주식 × 등락'에 비례하니 → 팔 것도 줄고 → 결국 노출이 **3~4%로 굳어버립니다.**
3배 레버리지 상품을 들고 있으면서 실질 노출은 0.1배가 되는 셈이라, 사실상 잠들어버립니다.
실제 계좌에서도 **자산의 96%가 현금**으로 굳는 것이 확인됐습니다.
    """)
    st.markdown(f"""
<div class="callout">
<span class="k">해법 — 바닥을 깐다</span>
매도 신호가 나와도, <b>매도 후 노출이 {_fl:.0f}% 밑으로 내려가면 그 매도를 취소</b>합니다.
즉 항상 최소 {_fl:.0f}%는 주식으로 들고 갑니다. 딱 이 한 줄로 경화가 사라집니다.
</div>
""", unsafe_allow_html=True)
    st.markdown("""
**개선 효과 (세전, 2010-04~2026-07 검증)**

| 지표 | 원작자 v1.0 | 위대리-H | 변화 |
|---|---|---|---|
| 현재 노출 | 3~13% (경화) | **15% (통일)** | 잠들지 않음 |
| 시작일별 CAGR 산포(σ) | 11.2% | **6.8%** | 출생복권 절반↓ |
| Calmar | 0.97 | **1.02** | 위험조정 개선 |
| MDD | −30% | **−26%** | 낙폭 축소 |

바닥을 깔면 폭락이 왔을 때 **반격할 주식·현금 배분이 유지**되어, 헤지 슬리브로서 제 역할을 합니다.
    """)
    st.info("💡 사이드바와 백테스트에서 노출 하한을 0~40%로 조절할 수 있습니다. "
            "0으로 두면 원작자 v1.0과 동일해집니다. 15%가 경화 방지 검증값입니다.", icon="🛟")

    # ═══ 4. 출생복권 ═══
    sec("4 · 출생복권 — 남은 약점과 대처", "Birth Lottery")
    st.markdown("""
노출 하한으로도 **완전히는 못 없애는** 약점이 하나 있습니다. 위대리는 **'언제 시작했느냐'에
따라 성과가 갈립니다.** 시작 시점에 시장이 비싸면(HIGH) 초반에 크게 못 사서 수익이 낮고,
쌀 때(LOW) 시작하면 즉시 매집해 좋은 성과가 납니다.
    """)
    st.markdown("""
| 시작 시점 (위대리-H) | 시작 Eval | 결과 CAGR |
|---|---|---|
| 운 좋은 시작 (저평가) | −10.6% | 38.9% |
| 중간 | +3.5% | 31.7% |
| 운 나쁜 시작 (고평가) | +28.8% | 22.1% |

*같은 규칙인데 시작 Eval 하나로 CAGR이 22~39%로 갈립니다 (상관 −0.59).*
    """)
    st.markdown("""
**대처법 — 분할 적립(DCA)**: 목돈을 한 번에 넣지 말고 **3~6개월에 걸쳐 나눠 사면**
진입 시점이 평균화되어 이 복권이 크게 줄어듭니다. 이건 파라미터가 아니라 **운용 방식**의
문제이고, 원작자도 문서에서 *"적립식은 TQQQ 매수"*라며 분할 매수를 권합니다.
    """)

    # ═══ 5. 단독이 아니라 슬리브 ═══
    sec("5 · 왜 혼자 쓰면 안 되는가", "Why a Sleeve")
    st.markdown("""
<div class="callout crimson">
<span class="k">원작자의 경고</span>
"QQQ와 같이 움직입니다. <b>QQQ 하락장에서는 수익을 낼 수 없습니다.</b>
… 동파·떨사 같은 단기매매, RP·SGOV 같은 원금보장 상품과 <b>병행</b>하세요."
</div>
""", unsafe_allow_html=True)
    st.markdown("""
위대리는 처음부터 **단독 완결 전략이 아닙니다.** QQQ 베타를 그대로 받는 롱-바이어스
전략이라, 장기 하락장에서는 방법이 없습니다. **동파공(모멘텀)과 짝을 이뤄야** 완성됩니다.

**핵심은 두 전략의 상관 +0.10 — 거의 무상관.** 동파공은 오르는 장에서 벌고, 위대리-H는
무너지는 장에서 반격합니다. 서로의 약한 국면이 반대라, 섞으면 어느 장세에도 한쪽이 받쳐줍니다.
    """)
    st.markdown("""
| 가상 세계 | 동파공 Calmar | 위대리-H Calmar | 7:3 Calmar |
|---|---|---|---|
| 추세 지속 세계 | 0.74 | 0.55 | **0.78** |
| 중간(현실적) | 1.11 | 0.64 | **1.18** |
| 평균회귀 세계 | 0.45 | 2.02 | 1.05 |

*동파공은 '회귀 세계'에서 약하고(0.45), 위대리-H는 '추세 세계'에서 약하다(0.55).
약한 세계가 서로 반대라, 7:3은 어느 세계든 0.78 밑으로 안 내려간다.*
    """)

    # ═══ 6. 권장 배분 ═══
    sec("6 · 권장 배분 & 운용", "Allocation")
    st.markdown("""
**세후 기준 (양도세 22% 5월 일괄 · 수수료 0.1% · 2010.04~2026.07)**

| 구성 | CAGR | MDD | Calmar | 용도 |
|---|---|---|---|---|
| 위대리-H 100% | 16.8% | −32.3% | 0.52 | 단독 (비권장) |
| **동파공 70 / 위대리-H 30** | **35.5%** | **−31.6%** | **1.12** | **수익형 (권장)** |
| 동파공 60 / 위대리-H 20 / SGOV 20 | 30.0% | −26.3% | **1.14** | 방어 강화형 |
| 동파공 70 / SGOV 30 | 31.0% | −28.3% | 1.10 | 참고 (SGOV 대체) |

**위대리-H가 SGOV(현금성 안전자산)보다 나은 이유**: SGOV는 폭락 때 가만히 있지만,
위대리-H는 쌓아둔 현금으로 **바닥에서 QQQ를 사서 반격**합니다. '가만히 있는 현금'이 아니라
'살아있는 헤지'입니다.
    """)
    st.markdown("""
**운용 3원칙**
1. **분기 리밸런싱** — 3개월마다 7:3 비율로 되돌립니다. 안 하면 수익 난 쪽으로 쏠려
   분산 효과가 사라집니다.
2. **분할 적립 진입** — 위대리-H를 새로 편입할 땐 목돈을 3~6개월에 나눠 삽니다 (출생복권 방지).
3. **노출 하한 유지** — 위대리-H가 15% 노출로 '적게' 들고 있어도 억지로 늘리지 마세요.
   그 절제가 폭락 때 실탄이 됩니다.
    """)

    # ═══ 7. 200일 게이트 ═══
    sec("7 · 200일 게이트 — 선택적 재난 보험", "Disaster Gate")
    st.markdown("""
원작자는 닷컴·2008급 붕괴를 *"발생 확률이 낮아 개발에서 제외"*하고, 대신 별도 출구전략
(예: 200일선 이탈 시 전량 매도)을 권합니다. 사이드바·백테스트의 **200일 추세 게이트**가
그 구현체입니다.

- **작동**: QQQ가 200일 이동평균 아래로 이탈하면 전량 현금화, 회복하면 재진입.
- **성격**: 위험조정 우월 전략이 아니라 **보험**입니다 — 평상시 강세장 수익을 일부 반납하는
  대신, 레짐 붕괴급 꼬리 손실을 잘라냅니다.
- **선택 기준**: "닷컴/2008급은 다시 안 온다"고 보면 **OFF**, "2022급 하락은 또 온다"고
  보면 **ON**. 이건 순수하게 리스크 선호 결정입니다.
    """)

    # ═══ 8. 업데이트 이력 ═══
    sec("8 · 버전 이력", "Changelog")
    with st.expander("vH1.0 — 위대리-H (현재)", expanded=True):
        st.markdown("""
- **원작자 v1.0 파라미터 채택**: 매도 100/50/33.3%, 매수 50/50/100%, HIGH컷 +5%.
  (기존 앱의 `diff×배율(2.0)` 방식은 원작자보다 과공격적이어서 경화를 심화시켰음)
- **노출 하한 15% 신설**: 매도 후 노출이 15% 밑이면 매도 취소. 경화를 원천 차단.
  → 현재 노출 3.8% → 15%, 출생복권(시작일별 CAGR 산포) 약 절반으로 축소.
- **데이터 로드 2005년부터**: 260주 추세선 완전 워밍업. 원본 v5.2가 2010부터 로드해
  초기 Eval이 왜곡되던(백테스트가 실전 봇보다 CAGR +25%p 부풀던) 문제 해결.
- **v4.8 신호 제거**: 12월 매도축소·멜트업 필터가 위대리-H 위에서는 세후 Calmar를
  오히려 낮춰(0.519→0.503) 제거. -H 구조가 이미 과매도를 막고 있어 중복이었음.
- **헤지 슬리브로 재정의**: 동파공 70 / 위대리-H 30 권장. 상관 +0.10, 세후 Calmar 1.12.
- **UI 전면 개편**: 심해 청록 테마 · 조석 게이지 · Eval 히스토리.
        """)
    with st.expander("원작자 위대리 v1.0 (기반)"):
        st.markdown("""
- QQQ 5년(260주) 로그 추세선 대비 괴리(Eval)로 3구간 분류, 매도/매수율 조절.
- 매도 100/50/33.3% · 매수 50/50/100% (251211 수정판).
- 원작자 기준 성과: CAGR 30.87% / MDD −30.86% (2010.2~2025.11).
- "QQQ 하락장에선 수익 불가 → 동파·떨사와 병행" 명시. 적립식 매수 권장.
        """)

    st.divider()
    st.caption("위대리-H vH1.0 · 레버리지 ETF는 변동성 감쇠·경로의존 위험이 있으며 과거 성과는 "
               "미래를 보장하지 않습니다. 본 문서는 검증 노트이며 투자권유가 아닙니다. "
               "원작자 위대리 매매법에 경의를 표합니다.")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — 포트폴리오 7:3 (동파공 + 위대리-H 통합 리밸런싱)
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
<div class="hero" style="margin-bottom:20px">
<div class="hero-eyebrow">Portfolio Console · 동파공 × 위대리-H</div>
<div class="hero-title" style="font-size:2rem">두 엔진의 <span class="h">균형추</span></div>
<div class="hero-sub">
동파공(모멘텀)과 위대리-H(역추세)를 7:3으로 유지한다. 한쪽이 더 벌면 비율이 틀어지므로,
분기~반기마다 균형을 되돌린다. 위대리-H 평가금은 자동, 동파공은 Dongpa!B8에서 읽어온다.
</div>
</div>
""", unsafe_allow_html=True)

    # ── 목표 비율 설정 ──
    sec("목표 배분", "Target")
    rc1, rc2 = st.columns([1, 1])
    with rc1:
        dpg_ratio = st.slider("동파공 비중 (%)", 50, 90, 70, 5, key='p_port_ratio')
    wed_ratio = 100 - dpg_ratio
    with rc2:
        st.markdown(f"<div style='padding-top:8px;font-family:IBM Plex Mono;font-size:1.1rem'>"
                    f"<span style='color:{C_BH}'>동파공 {dpg_ratio}%</span> &nbsp;:&nbsp; "
                    f"<span style='color:{C_TEAL_GLOW}'>위대리-H {wed_ratio}%</span></div>",
                    unsafe_allow_html=True)

    # ── 현재 평가금 읽기 ──
    sec("현재 평가금", "Current Value")
    _dpg_total, _dpg_err = load_dongpa_total()

    vc1, vc2 = st.columns(2)
    with vc1:
        st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:12px;color:{C_BH};"
                    f"letter-spacing:.1em;text-transform:uppercase'>◆ 동파공 (SOXL)</div>",
                    unsafe_allow_html=True)
        if _dpg_total is not None:
            dpg_now = st.number_input("동파공 평가금 ($)", value=float(round(_dpg_total, 2)),
                                      step=1000.0, key='p_dpg_now',
                                      help="Dongpa!B8 에서 자동으로 불러온 값. 필요시 수정 가능.")
            st.caption(f"✅ Dongpa!{DONGPA_TOTAL_CELL} 에서 자동 로드")
        else:
            dpg_now = st.number_input("동파공 평가금 ($) — 직접 입력", value=0.0, step=1000.0,
                                      key='p_dpg_now',
                                      help="시트에서 못 읽어 직접 입력합니다.")
            st.caption(f"⚠️ 자동 로드 실패: {_dpg_err} — 직접 입력하세요")
    with vc2:
        st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:12px;color:{C_TEAL_GLOW};"
                    f"letter-spacing:.1em;text-transform:uppercase'>◆ 위대리-H (TQQQ)</div>",
                    unsafe_allow_html=True)
        wed_now = st.number_input("위대리-H 평가금 ($)", value=float(round(wedaeri_total, 2)),
                                  step=1000.0, key='p_wed_now',
                                  help="이 앱이 계산한 현재 총자산. 실계좌와 다르면 수정.")
        st.caption("✅ 이 앱의 실전 탭에서 자동 계산")

    total_now = dpg_now + wed_now
    if total_now <= 0:
        st.info("두 평가금을 입력하면 리밸런싱을 계산합니다.", icon="💡")
        st.stop()

    # ── 현재 비율 & 목표 ──
    cur_dpg_pct = dpg_now / total_now * 100
    cur_wed_pct = wed_now / total_now * 100
    tgt_dpg = total_now * dpg_ratio / 100
    tgt_wed = total_now * wed_ratio / 100
    move = tgt_dpg - dpg_now   # +면 동파공으로 이동, -면 위대리로 이동

    sec("현재 비율", "Balance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("총 자산", f"${total_now:,.0f}")
    mc2.metric("동파공 현재", f"{cur_dpg_pct:.1f}%", f"목표 {dpg_ratio}%")
    mc3.metric("위대리-H 현재", f"{cur_wed_pct:.1f}%", f"목표 {wed_ratio}%")
    drift = abs(cur_dpg_pct - dpg_ratio)
    mc4.metric("이탈 폭", f"{drift:.1f}%p",
               "리밸런싱 권장" if drift >= 5 else "허용 범위")

    # ── 비율 바 ──
    bar_html = f"""
<div style="margin:14px 0 6px">
  <div style="display:flex;height:34px;border-radius:8px;overflow:hidden;border:1px solid #1d3843">
    <div style="width:{cur_dpg_pct}%;background:linear-gradient(90deg,#c2410c,#e8863f);
         display:flex;align-items:center;justify-content:center;font-family:IBM Plex Mono;
         font-size:12px;font-weight:600;color:#fff">동파공 {cur_dpg_pct:.0f}%</div>
    <div style="width:{cur_wed_pct}%;background:linear-gradient(90deg,#1e5f74,#2b9db8);
         display:flex;align-items:center;justify-content:center;font-family:IBM Plex Mono;
         font-size:12px;font-weight:600;color:#fff">위대리-H {cur_wed_pct:.0f}%</div>
  </div>
  <div style="position:relative;height:16px">
    <div style="position:absolute;left:{dpg_ratio}%;top:0;width:2px;height:12px;
         background:#eaf6f9;"></div>
    <div style="position:absolute;left:{dpg_ratio}%;top:0;transform:translateX(-50%);
         font-family:IBM Plex Mono;font-size:10px;color:#eaf6f9;margin-top:12px">▲ 목표 {dpg_ratio}:{wed_ratio}</div>
  </div>
</div>
"""
    st.markdown(bar_html, unsafe_allow_html=True)

    # ── 리밸런싱 지시 ──
    sec("리밸런싱 실행", "Action")
    if drift < 3:
        st.success(f"✅ 현재 이탈 {drift:.1f}%p — 리밸런싱 불필요. 이대로 두세요.\n\n"
                   f"(3%p 미만이면 매매 마찰이 이득보다 큽니다. "
                   f"보통 5%p 이상 벌어졌을 때만 맞추면 충분합니다.)", icon="⚖️")
    else:
        if move > 0:
            src, dst, src_c, dst_c = "위대리-H", "동파공", C_TEAL_GLOW, C_BH
        else:
            src, dst, src_c, dst_c = "동파공", "위대리-H", C_BH, C_TEAL_GLOW
        amt = abs(move)
        st.markdown(f"""
<div class="callout" style="border-left-color:{dst_c}">
<span class="k" style="color:{dst_c}">이동 지시</span>
<b style="font-size:1.3rem;color:{src_c}">{src}</b> 계좌에서
<b style="font-size:1.3rem;color:{dst_c}">{dst}</b> 계좌로<br>
<b style="font-size:1.8rem;font-family:Space Grotesk;color:#eaf6f9">${amt:,.0f}</b> 이동하세요.
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
**이동 후 목표 잔고**
| 계좌 | 현재 | → | 목표 |
|---|---|---|---|
| 동파공 | ${dpg_now:,.0f} | | **${tgt_dpg:,.0f}** |
| 위대리-H | ${wed_now:,.0f} | | **${tgt_wed:,.0f}** |
""")
        wed_delta = -amt if src == "위대리-H" else amt   # 위대리 입장 증감 (+입금/−출금)
        st.markdown(f"""
<div class="callout amber">
<span class="k">실행 체크리스트</span>
① <b>현금으로 이동</b>하세요 (주식 매도 시 양도세 발생 — 위대리-H는 현금 비중이 높아 대개 현금으로 가능).<br>
② 위대리-H는 <b>아래 '현금 조정 기록'</b>에 <b>{wed_delta:+,.0f}</b>을 입력하세요.
   → 주식 수는 그대로 두고 <b>예수금만</b> 조정됩니다 (원금·주식수 재계산 없음).<br>
③ 동파공은 해당 앱/시트에서 운용원금을 <b>${tgt_dpg:,.0f}</b>로 갱신.
</div>
""", unsafe_allow_html=True)

    # ── 현금 조정 기록 (리밸런싱 입출금) ──
    sec("현금 조정 기록", "Cash Adjustment")
    st.markdown("""
리밸런싱으로 위대리-H 계좌의 **예수금이 늘거나 줄면** 여기에 기록하세요.
**주식 수는 그대로 두고 현금만** 조정합니다 (원금을 바꿔 전체를 재계산하는 것과 다릅니다).
""")
    st.markdown("""
<div class="callout">
<span class="k">부호 규칙</span>
· <b>양수(+)</b> = 위대리-H로 <b>현금 유입</b> (동파공에서 받아옴 → 매수 실탄 증가)<br>
· <b>음수(−)</b> = 위대리-H에서 <b>현금 유출</b> (동파공으로 보냄 → 매수 실탄 감소)
</div>
""", unsafe_allow_html=True)

    adj_c1, adj_c2 = st.columns([1, 1])
    with adj_c1:
        st.markdown("**새 조정 추가**")
        with st.form("cash_adj_form", clear_on_submit=True):
            a_date = st.date_input("조정일 (자금 이동한 날)", value=datetime.now().date(), key='p_adj_date')
            a_amt = st.number_input("금액 ($, 유입=+ / 유출=−)", value=0.0, step=1000.0, key='p_adj_amt',
                                    help="예: 동파공에서 $64,582 받아오면 +64582, 동파공으로 보내면 −64582")
            a_note = st.text_input("메모 (선택)", value="리밸런싱", key='p_adj_note')
            if st.form_submit_button("기록 저장", use_container_width=True):
                if abs(a_amt) < 0.01:
                    st.error("금액이 0입니다.")
                else:
                    ok, err = save_cash_adjustment({'date': a_date.strftime("%Y-%m-%d"),
                                                    'amount': round(a_amt, 2), 'note': a_note})
                    if ok:
                        load_cash_adjustments.clear()
                        st.success(f"기록 완료: {a_amt:+,.0f}"); st.rerun()
                    else:
                        st.error(f"저장 실패: {err}")
    with adj_c2:
        st.markdown("**기록된 조정**")
        if _cash_adj_err:
            st.error(f"시트 로드 실패: {_cash_adj_err}")
        elif not _cash_adjs:
            st.info("아직 기록된 조정이 없습니다.")
        else:
            _net = sum(a['amount'] for a in _cash_adjs)
            _df_adj = pd.DataFrame([{'날짜': a['date'], '금액': f"{a['amount']:+,.0f}", '메모': a['note']}
                                    for a in _cash_adjs])
            st.dataframe(_df_adj, use_container_width=True, hide_index=True)
            st.caption(f"누적 순조정: **{_net:+,.0f}** ({len(_cash_adjs)}건) — 현재 예수금에 반영됨")
            _del_opts = ['(선택 안 함)'] + [f"{a['date']} · {a['amount']:+,.0f}" for a in _cash_adjs]
            _del = st.selectbox("삭제할 기록", _del_opts, key='p_adj_del')
            if _del != '(선택 안 함)' and st.button("선택 기록 삭제", use_container_width=True):
                _idx = _del_opts.index(_del) - 1
                ok, err = delete_cash_adjustment(_cash_adjs[_idx]['date'], _cash_adjs[_idx]['amount'])
                if ok:
                    load_cash_adjustments.clear()
                    st.success("삭제 완료"); st.rerun()
                else:
                    st.error(f"삭제 실패: {err}")

    # ── 리밸런싱 빈도 안내 ──
    sec("리밸런싱 원칙", "Guideline")
    st.markdown("""
- **빈도**: 분기~반기 1회면 충분합니다. 검증상 분기(Calmar 1.75)와 연 1회(1.77)는 사실상 동일합니다.
  너무 자주 하면 매매 마찰·세금만 늘어납니다.
- **트리거**: 날짜보다 **이탈 폭 기준**이 실용적입니다. 비율이 5%p 이상 벌어졌을 때만 맞추세요.
- **방향**: 보통 동파공(수익 엔진)이 더 벌어 비중이 커집니다 → 위대리-H로 옮기게 됩니다.
  반대로 폭락기엔 위대리-H가 방어해 커질 수 있습니다.
- **현금 우선**: 양도세를 피하려면 파는 쪽이 아니라 **현금이 있는 쪽에서** 이동하세요.
""")

    st.divider()
    st.caption("포트폴리오 콘솔 · 동파공 평가금은 Dongpa!B8 에서 읽어옵니다. "
               "리밸런싱은 참고용 계산이며, 실제 자금 이동·주문은 직접 실행하세요. "
               "이 화면은 주문을 자동 실행하지 않습니다.")
