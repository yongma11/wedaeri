# wedaeri_app.py — TQQQ 위대리 v5.2
# Tab1: 실전 트레이딩 | Tab2: 백테스트 분석 | Tab3: 전략 로직
#
# v5.2 변경사항 (200일 추세 게이트 + 프리-2010 합성 데이터):
#   • [신규] 200일 추세 게이트 on/off — 기초지수(QQQ)가 200일 SMA 아래로
#     이탈하면 전량 현금화, 회복하면 재진입. 닷컴/2008/2022 같은 레짐 붕괴
#     꼬리를 절단하는 "선택적 재난 보험". 사이드바(실전)·Tab2(백테스트) 토글.
#     - 검증: 위험조정수익 우월 전략이 아니라 보험 — 강세장 CAGR 일부 반납.
#       리스크 선호에 따라 ON/OFF. (밴드/확인지연 정교화는 OOS에서 기각됨)
#   • [신규] 2010년 이전 합성 데이터 — TQQQ 상장(2010-02) 이전을 QQQ 3배
#     일간리밸런스 합성으로 채워 닷컴(2000-02)·금융위기(2008)까지 백테스트.
#     ※ 변동성 잠식은 정확하나 금융비용은 근사치 → 절대수익보다 게이트 유무
#       상대비교 용도. Tab2 토글 ON 시 검증 시작일을 1999-03 까지 선택 가능.
#
# v5.1 변경사항 (봇 엔진 동기화 — 주문 수량 불일치 해결):
#   • SGOV 파킹 로직 전면 제거 (v5.0 롤백) — RP 자동투자로 대체, 복잡도 감소
#   • [핵심 수정] 주간 패널을 봇(wedaeri_bot.py)과 동일하게 구성:
#     ① 금요일 휴장 주 포함 — 기존에는 Eval 을 일별 df 에 머지 후 리샘플해서
#        금요일 휴장 주(성금요일/준틴스/7.4 등)의 Eval 이 NaN → dropna 로
#        그 주 전체가 조용히 삭제되어 봇과 매매 체인이 어긋났음.
#        이제 주간 QQQ 에서 OLS/Eval 을 직접 계산 (봇 compute_expanding_ols 동일)
#     ② QQQ 13주 MA 를 *전체 히스토리* 기준으로 계산 — 기존에는 시작일 이후
#        데이터로만 계산해 첫 13주의 멜트업 필터 판정이 봇과 달랐음
#     ③ 초기 현금 = 원금 - 주식매수액 (봇과 동일; 정수 주수 잔액을 현금에 보존)
#   • 검증: 2025-12-26 시작 실계좌 구간에서 봇과 주문 4건 불일치 → 0건 일치
#
# v4.9 변경사항 (분석용 로그 강화):
#   • 백테스트 매매 로그를 "완전 주간 패널"로 재설계 — 매주 1행(관망 포함)
#   • 모든 값 raw 숫자(무포맷)로 저장 -> CSV 다운로드만으로 전략 재현/검증 가능
#   • 신호 진단 컬럼 추가: qqq_growth / qqq_ma / qqq_ma_ratio / signal_diff /
#     sell_rate / buy_rate(실효 배율) / mom_filter / dec_scale / commission /
#     tax_paid / peak / drawdown
#   • 화면 표는 읽기용 포맷 뷰 + "원본 숫자값" 토글, CSV 는 항상 raw
#
# v4.8 변경사항 (OOS-검증 +0.75%p CAGR):
#   • Task A — 12월 매도 축소 (dec_sell_scale=0.75 기본): 양도세 이연
#   • Task B — HIGH-tier 모멘텀 필터: QQQ >= 13주MA x 1.08 시 매도 배율 0.7x
#   • 워크포워드 7-fold OOS 검증 통과 — 5-tier 확장은 과적합으로 채택 안 함
#   • 모든 v4.8 신호는 사이드바/Tab2 토글로 ON/OFF 가능 (기본 ON)
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
# 설정 파일 & Google Sheets 연동
# ─────────────────────────────────────────────────────────────
CONFIG_FILE = Path(__file__).parent / "wedaeri_config.json"
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
DEFAULT_CONFIG = {
    'start_date': '2025-12-26',
    'cap':        108000,
    'cash':       45,
    'bt_cap':     10000,
    'bt_cash':    45,
    'bt_start':   '2017-01-01',
    'hc':  6.0,   'lc': -6.0,
    'sH':  2.0,   'sM':  0.3,   'sL': 0.2,
    'bH':  1.0,   'bM':  0.6,   'bL': 2.0,
    'tax_apply_to_bot': False,   # 봇 가상 잔고에 양도세 차감 반영 여부 (기본 OFF)
    'use_gate': False,           # 200일 추세 게이트 (실전) 기본 OFF
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
        'sH':  float(ss.get('p_sH',  DEFAULT_CONFIG['sH'])),
        'sM':  float(ss.get('p_sM',  DEFAULT_CONFIG['sM'])),
        'sL':  float(ss.get('p_sL',  DEFAULT_CONFIG['sL'])),
        'bH':  float(ss.get('p_bH',  DEFAULT_CONFIG['bH'])),
        'bM':  float(ss.get('p_bM',  DEFAULT_CONFIG['bM'])),
        'bL':  float(ss.get('p_bL',  DEFAULT_CONFIG['bL'])),
        'tax_apply_to_bot': bool(ss.get('p_tax_apply_to_bot',
                                        DEFAULT_CONFIG['tax_apply_to_bot'])),
        'use_gate': bool(ss.get('p_use_gate', DEFAULT_CONFIG['use_gate'])),
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg
@st.cache_resource(show_spinner=False)
def _gspread_client_cached(creds_hash: str, creds_json: str):
    """성공한 gspread 클라이언트만 캐시. creds_json 이 바뀌면 새 클라이언트 생성.
    실패는 절대 캐시되지 않도록 예외를 그대로 throw 해서 호출 측에서 처리."""
    import gspread
    creds = json.loads(creds_json)
    if 'private_key' in creds:
        creds['private_key'] = creds['private_key'].replace('\\n', '\n')
    return gspread.service_account_from_dict(creds)
def _get_gspread_client():
    """실패 결과는 캐시되지 않습니다. Secrets 수정 후 재시도 시 즉시 반영됨."""
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
    """설정 딕셔너리를 Google Sheets '설정' 시트에 저장."""
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
# ─────────────────────────────────────────────────────────────
# 양도세 납부 기록 (Sheets '양도세납부' 시트)
# 컬럼: date | for_year | amount_usd | amount_krw | fx_rate | note
# ─────────────────────────────────────────────────────────────
TAX_SHEET_NAME = "양도세납부"
TAX_SHEET_HEADER = ['date', 'for_year', 'amount_usd', 'amount_krw', 'fx_rate', 'note']
def load_tax_payments() -> tuple:
    """납부 기록 리스트 반환. 시트 없으면 빈 리스트."""
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
    """납부 1건을 시트에 *추가* (기존 기록 보존)."""
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
    """date 가 일치하는 첫 번째 행을 삭제."""
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
# 1. 페이지 설정 & 스타일
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="위대리 Quantum T-Flow v5.2", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px;
            border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.order-card-buy  { background-color: #e6f4ea; border: 1px solid #ceead6; padding: 25px;
                   border-radius: 12px; text-align: center; }
.order-card-sell { background-color: #f1f3f4; border: 1px solid #dadce0; padding: 25px;
                   border-radius: 12px; text-align: center; color: #5f6368; }
.tier-high { background-color: #fff3cd; border-left: 4px solid #ffc107;
             padding: 12px 16px; border-radius: 6px; margin: 6px 0; }
.tier-mid  { background-color: #d1ecf1; border-left: 4px solid #17a2b8;
             padding: 12px 16px; border-radius: 6px; margin: 6px 0; }
.tier-low  { background-color: #d4edda; border-left: 4px solid #28a745;
             padding: 12px 16px; border-radius: 6px; margin: 6px 0; }
</style>
""", unsafe_allow_html=True)
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
    ss['p_sH']      = _cfg['sH']
    ss['p_sM']      = _cfg['sM']
    ss['p_sL']      = _cfg['sL']
    ss['p_bH']      = _cfg['bH']
    ss['p_bM']      = _cfg['bM']
    ss['p_bL']      = _cfg['bL']
    ss['p_tax_apply_to_bot'] = bool(_cfg.get('tax_apply_to_bot', False))
    ss['p_use_gate'] = bool(_cfg.get('use_gate', False))
# ─────────────────────────────────────────────────────────────
# 차트 공통 레이아웃
# ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,15,30,0.8)',
    font=dict(color='#cbd5e1', size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation='h', y=1.12, font=dict(size=10)),
)
def apply_grid(fig):
    fig.update_xaxes(gridcolor='#1e293b', zeroline=False)
    fig.update_yaxes(gridcolor='#1e293b', zeroline=False)
    return fig
# ─────────────────────────────────────────────────────────────
# 2. 사이드바
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("위대리 기본 설정")
    with st.container(border=True):
        st_start      = st.date_input("투자 시작일", key='p_start')
        st_cap        = st.number_input("시작 원금 ($)", step=1000.0, key='p_cap')
        st_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, key='p_cash') / 100
        col_ref, col_save = st.columns(2)
        with col_ref:
            if st.button("데이터 갱신", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_save:
            if st.button("설정 저장", use_container_width=True):
                cfg = save_config(st.session_state)
                ok, err = save_config_to_sheets(cfg)
                if ok:
                    st.toast("저장 + Sheets 동기화 완료! 봇에 자동 반영됩니다.", icon="💾")
                else:
                    st.toast("로컬 저장 완료", icon="💾")
                    err_text = str(err)
                    if 'invalid_grant' in err_text.lower() or 'jwt' in err_text.lower():
                        st.error(
                            f"Sheets 동기화 실패 — 인증은 됐지만 키가 무효:\n```\n{err_text}\n```\n"
                            f"**원인 후보**:\n"
                            f"1. private_key 줄바꿈 깨짐 -> Streamlit Secrets 에서 삼중따옴표 + 실제 개행 사용\n"
                            f"2. GCP 서비스 계정 키가 삭제/회전됨 -> 새 키 발급 필요\n"
                            f"3. 시스템 시계 5분+ 어긋남 -> NTP 동기화\n"
                            f"4. 봇이 동일 키로 정상 동작한다면 -> 앱 Secrets 형식만 문제",
                            icon="🔴"
                        )
                    elif 'permission' in err_text.lower() or '403' in err_text:
                        st.error(
                            f"Sheets 동기화 실패 — 권한 부족:\n```\n{err_text}\n```\n"
                            f"서비스 계정 이메일을 시트의 공유 대상에 **편집자** 권한으로 추가하세요.",
                            icon="🔴"
                        )
                    elif 'not found' in err_text.lower() or '404' in err_text:
                        st.error(
                            f"Sheets 동기화 실패 — 시트 키 오류:\n```\n{err_text}\n```\n"
                            f"SHEET_KEY 상수 값을 확인하세요.",
                            icon="🔴"
                        )
                    else:
                        st.warning(
                            f"Sheets 동기화 실패: {err_text}\n\n"
                            f"Streamlit Secrets 에 gcp_service_account 또는 GCP_CREDENTIALS 가 있는지 확인하세요.",
                            icon="🔴"
                        )
    st.divider()
    _gc, _gc_err = _get_gspread_client()
    if _gc:
        st.success("Sheets 동기화 활성화", icon="✅")
    else:
        st.error("Sheets 미연결 — 봇 자동 반영 불가\n\nStreamlit Secrets에 GCP_CREDENTIALS 추가 필요", icon="🔴")
    st.divider()
    _p = st.session_state
    _hc_d = _p.get('p_hc', DEFAULT_CONFIG['hc']);  _lc_d = _p.get('p_lc', DEFAULT_CONFIG['lc'])
    _sH_d = _p.get('p_sH', DEFAULT_CONFIG['sH']);  _sM_d = _p.get('p_sM', DEFAULT_CONFIG['sM'])
    _sL_d = _p.get('p_sL', DEFAULT_CONFIG['sL'])
    _bH_d = _p.get('p_bH', DEFAULT_CONFIG['bH']);  _bM_d = _p.get('p_bM', DEFAULT_CONFIG['bM'])
    _bL_d = _p.get('p_bL', DEFAULT_CONFIG['bL'])
    # ── v4.8: OOS-검증된 추가 신호 토글 ──
    with st.expander("v4.8 추가 신호 (Task A + Task B)", expanded=False):
        v48_enable = st.checkbox(
            "v4.8 적용", value=True, key='p_v48_enable',
            help="OOS-검증 +0.75%p CAGR. 끄면 기존 v4.7 동작"
        )
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.number_input("12월 매도 축소율", min_value=0.0, max_value=1.0,
                            value=0.75, step=0.05, key='p_dec_scale',
                            disabled=not v48_enable,
                            help="12월 매도 배율에 곱함. OOS 최적값 0.75")
            st.number_input("MA 윈도우 (주)", min_value=4, max_value=26,
                            value=13, step=1, key='p_ma_w',
                            disabled=not v48_enable,
                            help="OOS 최적값 13주")
        with col_v2:
            st.number_input("추세 임계값 xMA", min_value=1.0, max_value=1.20,
                            value=1.08, step=0.01, key='p_thr',
                            disabled=not v48_enable,
                            help="QQQ가 MA x 이 값 이상이면 멜트업으로 간주. OOS 최적값 1.08")
            st.number_input("HIGH 매도 축소율", min_value=0.0, max_value=1.0,
                            value=0.70, step=0.05, key='p_mult',
                            disabled=not v48_enable,
                            help="멜트업 시 HIGH 매도 배율에 곱함. OOS 최적값 0.70")

    # ── v5.2: 200일 추세 게이트 (실전 트레이딩에 적용) ──
    st.divider()
    st_use_gate = st.checkbox(
        "200일 추세 게이트 (재난 보험)", value=bool(_p.get('p_use_gate', False)),
        key='p_use_gate',
        help="기초지수(QQQ)가 200일 SMA 아래로 이탈하면 전량 현금화 후 재진입 대기.\n"
             "닷컴/2008/2022 같은 레짐 붕괴 꼬리를 절단하는 선택적 보험.\n"
             "검증: 위험조정수익 우월 전략이 아니라 보험 — 강세장 수익 일부 반납.\n"
             "리스크 선호에 따라 ON/OFF. (변경 후 '설정 저장'하면 봇에도 반영)"
    )

    st.markdown(f"""
**현재 적용 파라미터**
| 티어 | Eval | 매도x | 매수x |
|------|------|-------|-------|
| HIGH | >= +{_hc_d:.1f}% | {_sH_d} | {_bH_d} |
| MID  | 중간 | {_sM_d} | {_bM_d} |
| LOW  | <= {_lc_d:.1f}% | {_sL_d} | {_bL_d} |

*(백테스트 탭에서 변경/저장 가능)*
""")
# ─────────────────────────────────────────────────────────────
# 3. 데이터 로딩
#    - extended=False: 실측만 (2010~). 실전/라이브 기본.
#    - extended=True : 프리-2010 합성 TQQQ 스플라이스 (1999-03~). 백테스트 전용.
# ─────────────────────────────────────────────────────────────
DATA_START_EXT  = "1999-01-01"   # QQQ 상장(1999-03)부터 — 닷컴/2008 포함
SYNTH_FINANCING = 0.03           # 연 금융비용 근사 (합성 3x)
SYNTH_EXPENSE   = 0.0086         # TQQQ 운용보수

@st.cache_data(ttl=3600, show_spinner=False)
def load_wedaeri_data(extended: bool = False):
    try:
        start = DATA_START_EXT if extended else "2010-01-01"
        raw = yf.download(["QQQ", "TQQQ"], start=start,
                          auto_adjust=True, progress=False)
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

        # ── v5.2: 합성 TQQQ — 실제 상장(2010-02) 이전을 QQQ 3배 일간리밸런스로 ──
        df['synthetic'] = False
        if extended and 'TQQQ' in df.columns and df['TQQQ'].notna().any():
            first_real_idx = int(df['TQQQ'].first_valid_index())
            if first_real_idx > 0:
                qret = df['QQQ'].pct_change().fillna(0.0)
                daily_drag = (3 - 1) * SYNTH_FINANCING / 252 + SYNTH_EXPENSE / 252
                synth = (1 + (3 * qret - daily_drag)).cumprod()
                # 실제 TQQQ 첫 종가에 맞춰 합성 구간 스케일 (연속성 보존)
                scale = float(df.loc[first_real_idx, 'TQQQ']) / float(synth.iloc[first_real_idx])
                pre = df.index < first_real_idx
                df.loc[pre, 'TQQQ'] = (synth * scale)[pre].values
                df.loc[pre, 'synthetic'] = True

        # ── 주간 QQQ OLS → Growth/Eval (일별 df 에 머지, 라이브/Tab3 표시용) ──
        qqq_wkly = (df.set_index('Date')[['QQQ']]
                    .resample('W-FRI').last()
                    .dropna()
                    .reset_index())
        n = len(qqq_wkly)
        W = 260
        t = np.arange(1, n + 1, dtype=float)
        y = np.log(qqq_wkly['QQQ'].values.astype(float))
        ps_t  = np.zeros(n + 1); ps_t[1:]  = np.cumsum(t)
        ps_y  = np.zeros(n + 1); ps_y[1:]  = np.cumsum(y)
        ps_t2 = np.zeros(n + 1); ps_t2[1:] = np.cumsum(t ** 2)
        ps_ty = np.zeros(n + 1); ps_ty[1:] = np.cumsum(t * y)
        growth_wkly = np.empty(n)
        for i in range(n):
            w     = min(i + 1, W)
            end   = i + 1
            start_ = end - w
            if w == 1:
                growth_wkly[i] = float(qqq_wkly['QQQ'].iloc[i])
                continue
            s_t  = ps_t[end]  - ps_t[start_]
            s_y  = ps_y[end]  - ps_y[start_]
            s_t2 = ps_t2[end] - ps_t2[start_]
            s_ty = ps_ty[end] - ps_ty[start_]
            denom = w * s_t2 - s_t ** 2
            if denom == 0:
                growth_wkly[i] = float(qqq_wkly['QQQ'].iloc[i])
                continue
            b = (w * s_ty - s_t * s_y) / denom
            a = (s_y - b * s_t) / w
            growth_wkly[i] = np.exp(a + b * t[i])
        qqq_wkly['Growth'] = growth_wkly
        qqq_wkly['Eval']   = qqq_wkly['QQQ'] / qqq_wkly['Growth'] - 1
        df = df.merge(qqq_wkly[['Date', 'Growth', 'Eval']], on='Date', how='left')
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()
# ─────────────────────────────────────────────────────────────
# 3-a. v5.1: 주간 패널 빌더 (봇 wedaeri_bot.py 와 동일 로직)
#   - 일별 QQQ/TQQQ 를 W-FRI 로 직접 리샘플 -> 금요일 휴장 주도 포함
#   - Growth/Eval 을 주간 QQQ 에서 재계산 (봇 compute_expanding_ols 동일)
#   - QQQ_MA 를 전체 히스토리 기준으로 계산 후 시작일 필터
#   - v5.2: gate_on (200일 SMA 게이트 플래그) 추가
# ─────────────────────────────────────────────────────────────
def compute_expanding_ols(qqq_weekly: pd.DataFrame, W: int = 260) -> np.ndarray:
    """주간 QQQ 시계열로 5년(260주) Expanding Window log-선형 회귀.
    봇(wedaeri_bot.py)의 compute_expanding_ols 와 완전 동일."""
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


def build_weekly_panel(data: pd.DataFrame, ma_window_weeks: int = 13,
                       gate_sma_days: int = 200) -> pd.DataFrame:
    """봇과 동일한 주간 패널 + 200일 추세 게이트 플래그.
    - 금요일 휴장 주 포함 (W-FRI 라벨, 목요일 종가로 마감)
    - Eval 은 주간 QQQ 에서 직접 재계산 (일별 머지 경유 X — 휴장 주 누락 방지)
    - QQQ_MA 는 전체 히스토리 기준 (시작일 필터는 호출 측에서)
    - gate_on: 일별 QQQ의 gate_sma_days일 SMA 대비 금요일 종가 위/아래 (인과적)
    """
    d = data.set_index('Date')
    qqq_weekly  = d[['QQQ']].resample('W-FRI').last().dropna().reset_index()
    tqqq_weekly = d[['TQQQ']].resample('W-FRI').last().dropna().reset_index()
    qqq_weekly['Growth'] = compute_expanding_ols(qqq_weekly, W=260)
    qqq_weekly['Eval']   = qqq_weekly['QQQ'] / qqq_weekly['Growth'] - 1
    if ma_window_weeks > 0:
        qqq_weekly['QQQ_MA'] = qqq_weekly['QQQ'].rolling(
            ma_window_weeks, min_periods=1).mean()
    else:
        qqq_weekly['QQQ_MA'] = qqq_weekly['QQQ']
    panel = qqq_weekly.merge(tqqq_weekly, on='Date', how='inner')
    # ── v5.2: 200일 추세 게이트 — 일별 QQQ의 SMA를 금요일에 샘플 ──
    sma = d['QQQ'].rolling(gate_sma_days, min_periods=gate_sma_days).mean()
    # 주의: QQQ > NaN 은 NaN 이 아니라 False 를 반환하므로, 워밍업 구간을
    # .where(sma.notna()) 로 명시적 NaN 처리해야 fillna(True) 가 먹는다.
    gate_daily = (d['QQQ'] > sma).where(sma.notna())
    gate_wk = gate_daily.resample('W-FRI').last().reindex(panel['Date'].values)
    # SMA 워밍업(첫 gate_sma_days 거래일) 구간 NaN → risk-on 으로 간주
    panel['gate_on'] = pd.Series(gate_wk.values).fillna(True).astype(bool).values
    return panel
# ─────────────────────────────────────────────────────────────
# 3-b. 실시간 종가 주입
# ─────────────────────────────────────────────────────────────
def inject_live_price(df: pd.DataFrame) -> pd.DataFrame:
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
# ─────────────────────────────────────────────────────────────
# 4. 실전 시뮬레이션 (Tab 1) — v5.1: 봇 동일 패널/초기현금, v5.2: 게이트
# ─────────────────────────────────────────────────────────────
def run_wedaeri_sim(data, start_dt, init_cap, cash_ratio,
                    hc=0.06, lc=-0.06,
                    sH=2.0, sM=0.3, sL=0.2,
                    bH=1.0, bM=0.6, bL=2.0,
                    # ── v4.8: OOS-검증된 추가 신호 (Task A + Task B) ──
                    dec_sell_scale=0.75,
                    ma_window_weeks=13,
                    trend_threshold=1.08,
                    sell_rate_multiplier=0.70,
                    # ── v5.2: 200일 추세 게이트 ──
                    use_gate=False,
                    gate_sma_days=200):
    # v5.1: 봇과 동일한 주간 패널 (휴장 금요일 포함, MA 전체 히스토리)
    panel = build_weekly_panel(data, ma_window_weeks=ma_window_weeks,
                               gate_sma_days=gate_sma_days)
    wkly = (panel[panel['Date'] >= pd.to_datetime(start_dt)]
            .dropna(subset=['Eval', 'TQQQ'])
            .reset_index(drop=True))
    if wkly.empty or len(wkly) < 2:
        return pd.DataFrame()
    # v5.1: 초기 현금 = 원금 - 주식매수액 (봇과 동일 — 정수 주수 잔액 보존)
    init_price = float(wkly.loc[0, 'TQQQ'])
    shares = int((init_cap * (1 - cash_ratio)) / init_price)
    cash   = init_cap - shares * init_price
    logs   = []
    for i in range(len(wkly)):
        p    = float(wkly.loc[i, 'TQQQ'])
        ev   = float(wkly.loc[i, 'Eval'])
        tier = 'HIGH' if ev >= hc else ('LOW' if ev <= lc else 'MID')
        gate_on = (not use_gate) or bool(wkly.loc[i, 'gate_on'])
        action, disp_qty = "관망", 0

        if use_gate and not gate_on:
            # v5.2 — 추세 이탈: 전량 현금화, 신규 매수 금지
            if shares > 0:
                disp_qty = -shares
                cash += shares * p
                shares = 0
                action = "게이트청산"
            else:
                action = "게이트대기"
        elif use_gate and shares == 0 and cash > 0:
            # v5.2 — 게이트 재진입: 목표 주식비중만큼 재투입
            qty = int((cash * (1 - cash_ratio)) / p)
            if qty > 0:
                disp_qty = qty
                shares += qty; cash -= qty * p
                action = "게이트재진입"
        else:
            sr   = {'HIGH': sH, 'MID': sM, 'LOW': sL}[tier]
            br   = {'HIGH': bH, 'MID': bM, 'LOW': bL}[tier]
            # v4.8 — HIGH-tier 모멘텀 필터: 진짜 멜트업에서만 매도 축소
            if tier == 'HIGH' and ma_window_weeks > 0:
                qqq_ma  = float(wkly.loc[i, 'QQQ_MA'])
                qqq_now = float(wkly.loc[i, 'QQQ'])
                if not pd.isna(qqq_ma) and qqq_now >= qqq_ma * trend_threshold:
                    sr *= sell_rate_multiplier
            # v4.8 — 12월 매도 축소 (양도세 이연)
            if pd.Timestamp(wkly.loc[i, 'Date']).month == 12:
                sr *= dec_sell_scale
            if i > 0:
                prev_p = float(wkly.loc[i-1, 'TQQQ'])
                diff   = shares * (p - prev_p)
                if diff > 0:
                    qty = int(min(round(diff * sr / p), shares))
                    if qty > 0:
                        action, disp_qty = "매도", -qty
                        shares -= qty; cash += qty * p
                elif diff < 0:
                    qty = int(min(cash, abs(diff) * br) / p)
                    if qty > 0:
                        action, disp_qty = "매수", qty
                        shares += qty; cash -= qty * p
        total = cash + shares * p
        logs.append({
            '날짜':     wkly.loc[i, 'Date'].strftime('%Y-%m-%d'),
            '시장평가': f"{ev:.2%}",
            '티어':     tier,
            '게이트':   ('—' if not use_gate else ('보유' if gate_on else '현금(이탈)')),
            '액션':     action,
            '주문수량': disp_qty,
            '보유수량': shares,
            '현금':     round(cash, 2),
            '총자산':   round(total, 2),
            '수익률':   f"{(total/init_cap - 1)*100:.2f}%",
        })
    return pd.DataFrame(logs)
# ─────────────────────────────────────────────────────────────
# 5. 전체 기간 백테스트 (Tab 2)
# ─────────────────────────────────────────────────────────────
TAX_SCHEDULES = {
    'A': {'name': '1월/5월 50/50',  'pattern': [(1, 0.50), (5, 0.50)]},
    'B': {'name': '5월 일괄',       'pattern': [(5, 1.00)]},
    'C': {'name': '1월 일괄',       'pattern': [(1, 1.00)]},
}
def run_full_backtest(data, init_cap=20_000, cash_ratio=0.45,
                      hc=0.06, lc=-0.06,
                      sH=2.0,  sM=0.3,  sL=0.2,
                      bH=1.0,  bM=0.6,  bL=2.0,
                      start_date=None,
                      apply_commission=False,
                      comm_buy=0.00015,
                      comm_sell=0.00017206,
                      apply_tax=False,
                      tax_deduction_usd=1923.0,
                      tax_rate=0.22,
                      tax_rebalance=True,
                      tax_strategy='A',
                      # ── v4.8: OOS-검증된 추가 신호 (Task A + Task B) ──
                      dec_sell_scale=0.75,
                      ma_window_weeks=13,
                      trend_threshold=1.08,
                      sell_rate_multiplier=0.70,
                      # ── v5.2: 200일 추세 게이트 ──
                      use_gate=False,
                      gate_sma_days=200):
    """위대리 전략 백테스트 (수수료 + 양도세 + 200일 게이트 옵션 포함).

    v5.1: 주간 패널을 봇과 동일하게 구성 (build_weekly_panel) —
      금요일 휴장 주 포함 + Eval 주간 재계산 + QQQ_MA 전체 히스토리.
    v5.2: use_gate 시 기초지수가 200일 SMA 이탈하면 전량 청산, 회복 시 재진입.
    """
    # v5.1: 봇과 동일한 주간 패널 (휴장 금요일 포함, Eval 재계산, MA 전체 히스토리)
    wkly = build_weekly_panel(data, ma_window_weeks=ma_window_weeks,
                              gate_sma_days=gate_sma_days)
    if start_date is not None:
        wkly = wkly[wkly['Date'] >= pd.to_datetime(start_date)].reset_index(drop=True)
    if len(wkly) < 2:
        return None
    P       = wkly['TQQQ'].values.astype(float)
    EV      = wkly['Eval'].values.astype(float)
    QQQ_arr = wkly['QQQ'].values.astype(float)
    QMA_arr = wkly['QQQ_MA'].values.astype(float)
    GATE    = wkly['gate_on'].values.astype(bool)
    if 'Growth' in wkly.columns:
        GROWTH_arr = wkly['Growth'].values.astype(float)
    else:
        GROWTH_arr = np.full(len(wkly), np.nan)
    dates   = wkly['Date'].values
    N     = len(wkly)
    span_days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
    YEARS = max(span_days / 365.25, N / 52)

    # 초기 포지션 — 첫 매수에도 수수료 적용 (현실 반영)
    cash   = float(init_cap * cash_ratio)
    init_stock_value = init_cap * (1 - cash_ratio)
    shares = int(init_stock_value / P[0])
    init_buy_cost = shares * P[0] * (1 + comm_buy if apply_commission else 1)
    cash = init_cap - init_buy_cost
    total_cost_basis = shares * P[0] * (1 + comm_buy if apply_commission else 1)

    eq        = np.empty(N)
    tiers     = []
    eq[0]     = cash + shares * P[0]
    realized_gain_year = {}
    cum_commission     = 0.0
    cum_tax            = 0.0
    tax_events         = []
    year_of_last_bar   = pd.Timestamp(dates[0]).year
    pending_payments   = []
    schedule_pattern   = TAX_SCHEDULES.get(tax_strategy, TAX_SCHEDULES['A'])['pattern']
    trade_log          = []

    def _sell(qty, price, year, mark_realized=True):
        """공통 매도 로직. cost_basis, shares, cash 갱신 + (옵션) 실현이익 기록."""
        nonlocal shares, cash, total_cost_basis, cum_commission
        if qty <= 0 or shares <= 0:
            return 0.0
        qty = min(qty, shares)
        gross   = qty * price
        fee     = gross * comm_sell if apply_commission else 0.0
        net_proc = gross - fee
        proportion = qty / shares
        cost_of_sold = total_cost_basis * proportion
        gain = net_proc - cost_of_sold
        total_cost_basis -= cost_of_sold
        shares -= qty
        cash   += net_proc
        cum_commission += fee
        if mark_realized:
            realized_gain_year[year] = realized_gain_year.get(year, 0.0) + gain
        return gain

    def _buy(qty, price):
        """공통 매수 로직. cost_basis, shares, cash 갱신."""
        nonlocal shares, cash, total_cost_basis, cum_commission
        if qty <= 0:
            return
        gross = qty * price
        fee   = gross * comm_buy if apply_commission else 0.0
        cost  = gross + fee
        if cost > cash:
            qty = int(cash / (price * (1 + comm_buy if apply_commission else 1)))
            if qty <= 0:
                return
            gross = qty * price
            fee   = gross * comm_buy if apply_commission else 0.0
            cost  = gross + fee
        total_cost_basis += cost
        shares += qty
        cash   -= cost
        cum_commission += fee

    for i in range(N):
        p  = P[i]
        ev = EV[i]
        date_t = pd.Timestamp(dates[i])
        cur_year = date_t.year
        tier = 'HIGH' if ev >= hc else ('LOW' if ev <= lc else 'MID')
        tiers.append(tier)

        # ── 이번 주 신호/체결 계산 ──
        action_label      = "관망"
        qty_signed        = 0
        realized_pnl_this = 0.0
        eff_sell_rate     = 0.0
        eff_buy_rate      = 0.0
        mom_fired         = 0
        dec_applied       = 0
        comm_this         = 0.0
        diff_val          = 0.0
        note_gate         = ''
        prev_p            = float(P[i-1]) if i > 0 else float('nan')
        gate_on_now       = (not use_gate) or bool(GATE[i])

        if use_gate and not gate_on_now:
            # v5.2 — 200일 게이트 OFF: 추세 이탈 → 전량 청산, 신규 매수 금지
            if shares > 0:
                sold = shares
                comm_before = cum_commission
                realized_pnl_this = _sell(sold, p, cur_year, mark_realized=True)
                comm_this = cum_commission - comm_before
                action_label = "매도"
                qty_signed = -sold
                note_gate = "게이트청산"
            else:
                note_gate = "게이트대기"
        elif use_gate and gate_on_now and shares == 0 and cash > 0:
            # v5.2 — 게이트 재진입: 목표 주식비중만큼 재투입
            budget = cash * (1 - cash_ratio)
            qty = int(budget / (p * (1 + comm_buy))) if apply_commission else int(budget / p)
            if qty > 0:
                shares_before = shares
                comm_before = cum_commission
                _buy(qty, p)
                comm_this = cum_commission - comm_before
                qty_signed = shares - shares_before
                if qty_signed > 0:
                    action_label = "매수"
                    note_gate = "게이트재진입"
        elif i > 0:
            diff = shares * (p - P[i-1])
            diff_val = float(diff)
            if diff > 0:
                sr = {'HIGH': sH, 'MID': sM, 'LOW': sL}[tier]
                if tier == 'HIGH' and ma_window_weeks > 0:
                    qma = QMA_arr[i]
                    if not np.isnan(qma) and QQQ_arr[i] >= qma * trend_threshold:
                        sr *= sell_rate_multiplier
                        mom_fired = 1
                if pd.Timestamp(dates[i]).month == 12:
                    sr *= dec_sell_scale
                    dec_applied = 1
                eff_sell_rate = sr
                qty = int(min(round(diff * sr / p), shares))
                if qty > 0:
                    comm_before = cum_commission
                    realized_pnl_this = _sell(qty, p, cur_year, mark_realized=True)
                    comm_this = cum_commission - comm_before
                    action_label = "매도"
                    qty_signed = -qty
            elif diff < 0:
                br = {'HIGH': bH, 'MID': bM, 'LOW': bL}[tier]
                eff_buy_rate = br
                budget = min(cash, abs(diff) * br)
                qty = int(budget / (p * (1 + comm_buy if apply_commission else 1))) \
                      if apply_commission else int(budget / p)
                if qty > 0:
                    shares_before = shares
                    comm_before = cum_commission
                    _buy(qty, p)
                    comm_this = cum_commission - comm_before
                    qty_actual = shares - shares_before
                    if qty_actual > 0:
                        action_label = "매수"
                        qty_signed = qty_actual

        # ── 매주 1행 완전 패널 기록 (관망 포함, raw 숫자) ──
        avg_cost    = (total_cost_basis / shares) if shares > 0 else 0.0
        total_asset = cash + shares * p
        qma_val = float(QMA_arr[i]) if (ma_window_weeks > 0 and not np.isnan(QMA_arr[i])) else float('nan')
        growth_val = float(GROWTH_arr[i]) if not np.isnan(GROWTH_arr[i]) else float('nan')
        ma_ratio = (float(QQQ_arr[i]) / qma_val) if (qma_val == qma_val and qma_val > 0) else float('nan')
        act_code = {'매수': 'BUY', '매도': 'SELL', '관망': 'HOLD'}[action_label]
        trade_log.append({
            'date':         date_t.strftime('%Y-%m-%d'),
            'week':         int(i),
            'action':       act_code,
            'tier':         tier,
            'eval':         float(ev),
            'qqq_close':    float(QQQ_arr[i]),
            'qqq_growth':   growth_val,
            'qqq_ma':       qma_val,
            'qqq_ma_ratio': ma_ratio,
            'gate_on':      int(bool(GATE[i])),
            'tqqq_close':   float(p),
            'tqqq_prev':    prev_p,
            'tqqq_wk_ret':  (float(p) / prev_p - 1.0) if (i > 0 and prev_p > 0) else float('nan'),
            'signal_diff':  diff_val,
            'sell_rate':    float(eff_sell_rate),
            'buy_rate':     float(eff_buy_rate),
            'mom_filter':   int(mom_fired),
            'dec_scale':    int(dec_applied),
            'trade_price':  float(p) if action_label != "관망" else float('nan'),
            'buy_qty':      int(qty_signed) if action_label == "매수" else 0,
            'sell_qty':     int(-qty_signed) if action_label == "매도" else 0,
            'realized_pnl': float(realized_pnl_this),
            'commission':   float(comm_this),
            'tax_paid':     0.0,
            'avg_cost':     float(avg_cost),
            'shares':       int(shares),
            'cash':         float(cash),
            'stock_value':  float(shares * p),
            'total_asset':  float(total_asset),
            'cum_return':   float(total_asset / init_cap - 1.0),
            'note':         note_gate,
        })

        # ── 양도세: 새 해 첫 weekly close 시점에 직전 연도 세액 계산 ──
        if apply_tax and cur_year > year_of_last_bar:
            prev_year = year_of_last_bar
            prev_gain = realized_gain_year.get(prev_year, 0.0)
            taxable   = max(0.0, prev_gain - tax_deduction_usd)
            total_tax_due = taxable * tax_rate
            if total_tax_due > 0:
                for (pm, frac) in schedule_pattern:
                    pending_payments.append({
                        'pay_year':    cur_year,
                        'pay_month':   pm,
                        'amount':      total_tax_due * frac,
                        'orig_amount': total_tax_due * frac,
                        'prev_year':   prev_year,
                        'prev_year_gain': prev_gain,
                        'taxable':     taxable,
                    })
            year_of_last_bar = cur_year

        # ── 양도세: 대기 중인 분할 납부 처리 ──────────────
        if apply_tax and pending_payments:
            cur_month = date_t.month
            still_pending = []
            for pay in pending_payments:
                due = (cur_year > pay['pay_year']) or \
                      (cur_year == pay['pay_year'] and cur_month >= pay['pay_month'])
                if not due:
                    still_pending.append(pay); continue
                amount = pay['amount']
                sold_for_tax = 0
                if cash < amount:
                    deficit = amount - cash
                    eff_p = p * (1 - comm_sell) if apply_commission else p
                    sell_qty = int(np.ceil(deficit / eff_p)) if eff_p > 0 else 0
                    sell_qty = min(sell_qty, shares)
                    sold_for_tax = sell_qty
                    if sell_qty > 0:
                        _sell(sell_qty, p, cur_year, mark_realized=True)
                paid = min(cash, amount)
                cash = max(0.0, cash - amount)
                cum_tax += paid
                trade_log.append({
                    'date':         date_t.strftime('%Y-%m-%d'),
                    'week':         int(i),
                    'action':       'TAX',
                    'tier':         tier,
                    'eval':         float(ev),
                    'qqq_close':    float(QQQ_arr[i]),
                    'qqq_growth':   growth_val,
                    'qqq_ma':       qma_val,
                    'qqq_ma_ratio': float('nan'),
                    'gate_on':      int(bool(GATE[i])),
                    'tqqq_close':   float(p),
                    'tqqq_prev':    float('nan'),
                    'tqqq_wk_ret':  float('nan'),
                    'signal_diff':  0.0,
                    'sell_rate':    0.0,
                    'buy_rate':     0.0,
                    'mom_filter':   0,
                    'dec_scale':    0,
                    'trade_price':  float(p) if sold_for_tax > 0 else float('nan'),
                    'buy_qty':      0,
                    'sell_qty':     int(sold_for_tax) if sold_for_tax > 0 else 0,
                    'realized_pnl': 0.0,
                    'commission':   0.0,
                    'tax_paid':     float(paid),
                    'avg_cost':     (total_cost_basis / shares) if shares > 0 else 0.0,
                    'shares':       int(shares),
                    'cash':         float(cash),
                    'stock_value':  float(shares * p),
                    'total_asset':  float(cash + shares * p),
                    'cum_return':   float((cash + shares * p) / init_cap - 1.0),
                    'note':         f"{pay['prev_year']}년 분 ({tax_strategy})",
                })
                tax_events.append({
                    'date':       date_t.strftime('%Y-%m-%d'),
                    'prev_year':  pay['prev_year'],
                    'gain':       pay['prev_year_gain'],
                    'taxable':    pay['taxable'],
                    'tax':        paid,
                    'orig_tax':   pay['orig_amount'],
                    'sold_for_tax': sold_for_tax,
                    'split':      f"{pay['orig_amount']:.0f} ({tax_strategy})",
                })
                if tax_rebalance:
                    remaining = cash + shares * p
                    target_cash = remaining * cash_ratio
                    if cash < target_cash:
                        d2 = target_cash - cash
                        eff_p = p * (1 - comm_sell) if apply_commission else p
                        sq = int(np.ceil(d2 / eff_p)) if eff_p > 0 else 0
                        sq = min(sq, shares)
                        if sq > 0:
                            _sell(sq, p, cur_year, mark_realized=True)
            pending_payments = still_pending

        eq[i] = cash + shares * p

    # ── 완전 패널 로그에 running peak / drawdown 부여 (시간순) ──
    _peak = None
    for _row in trade_log:
        ta = _row['total_asset']
        _peak = ta if _peak is None else max(_peak, ta)
        _row['peak'] = float(_peak)
        _row['drawdown'] = float(ta / _peak - 1.0) if _peak and _peak > 0 else 0.0

    cagr  = (eq[-1] / init_cap) ** (1 / YEARS) - 1
    peak  = np.maximum.accumulate(eq)
    dd    = eq / peak - 1
    mdd   = dd.min()
    rets  = np.diff(eq) / eq[:-1]
    if len(rets) == 0:
        rets = np.array([0.0])
    neg   = rets[rets < 0]
    dstd  = neg.std() * np.sqrt(52) if len(neg) > 1 else 1e-9
    ann_r = (1 + rets.mean()) ** 52 - 1
    sor   = ann_r / dstd if dstd > 0 else 0
    cal   = cagr / abs(mdd) if mdd != 0 else 0
    bh_peak = np.maximum.accumulate(P)
    yearly_df = pd.DataFrame({'date': pd.to_datetime(dates), 'eq': eq})
    yearly_df['year'] = yearly_df['date'].dt.year
    yearly_rows = []
    prev_end = init_cap
    for yr, grp in yearly_df.groupby('year'):
        start_eq = prev_end
        end_eq   = float(grp['eq'].iloc[-1])
        ret_pct  = (end_eq / start_eq - 1) * 100 if start_eq > 0 else 0
        yr_peak = np.maximum.accumulate(grp['eq'].values)
        yr_mdd  = float((grp['eq'].values / yr_peak - 1).min()) * 100
        yr_realized = realized_gain_year.get(int(yr), 0.0)
        yr_tax_paid = sum(t['tax'] for t in tax_events if t['prev_year'] == int(yr))
        yearly_rows.append({
            '연도': int(yr),
            '수익률': f"{ret_pct:+.1f}%",
            '연간 MDD': f"{yr_mdd:.1f}%",
            '실현이익': f"${yr_realized:,.0f}",
            '양도세(다음해 정산)': f"${yr_tax_paid:,.0f}",
            '기말 자산': f"${end_eq:,.0f}",
        })
        prev_end = end_eq
    return {
        'cagr': cagr,  'mdd': mdd,   'cal': cal,  'sor': sor,
        'final': eq[-1], 'init': init_cap, 'years': YEARS,
        'eq': eq, 'dd': dd, 'dates': dates,
        'tiers': tiers, 'prices': P, 'eval': EV,
        'bh_eq':   P / P[0] * init_cap,
        'bh_cagr': (P[-1] / P[0]) ** (1 / YEARS) - 1,
        'bh_mdd':  (P / bh_peak - 1).min(),
        'rets': rets,
        'yearly': yearly_rows,
        'cum_commission': cum_commission,
        'cum_tax':        cum_tax,
        'tax_events':     tax_events,
        'realized_gain_year': realized_gain_year,
        'tax_strategy':   tax_strategy,
        'unpaid_tax':     sum(p['amount'] for p in pending_payments),
        'trade_log':      trade_log,
    }
# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
with st.spinner("데이터 로딩 중 (최초 1회만 시간이 걸립니다)..."):
    df = load_wedaeri_data()
df = inject_live_price(df)
if df.empty:
    st.error("데이터 로드 실패. 잠시 후 새로고침 해주세요.")
    st.stop()
_ss = st.session_state
# v4.8: 토글 상태에 따라 v4.8 신호를 활성/no-op 으로 패스
_v48_on = _ss.get('p_v48_enable', True)
log_df = run_wedaeri_sim(
    df, st_start, st_cap, st_cash_ratio,
    hc  = _ss.get('p_hc', DEFAULT_CONFIG['hc']) / 100,
    lc  = _ss.get('p_lc', DEFAULT_CONFIG['lc']) / 100,
    sH  = _ss.get('p_sH', DEFAULT_CONFIG['sH']),
    sM  = _ss.get('p_sM', DEFAULT_CONFIG['sM']),
    sL  = _ss.get('p_sL', DEFAULT_CONFIG['sL']),
    bH  = _ss.get('p_bH', DEFAULT_CONFIG['bH']),
    bM  = _ss.get('p_bM', DEFAULT_CONFIG['bM']),
    bL  = _ss.get('p_bL', DEFAULT_CONFIG['bL']),
    dec_sell_scale       = _ss.get('p_dec_scale', 0.75) if _v48_on else 1.0,
    ma_window_weeks      = _ss.get('p_ma_w', 13)       if _v48_on else 0,
    trend_threshold      = _ss.get('p_thr', 1.08)      if _v48_on else 99.0,
    sell_rate_multiplier = _ss.get('p_mult', 0.70)     if _v48_on else 1.0,
    use_gate             = _ss.get('p_use_gate', False),
)
tqqq_series = df['TQQQ'].dropna()
latest_tqqq = float(tqqq_series.iloc[-1]) if not tqqq_series.empty else 0.0
eval_series  = df['Eval'].dropna()
latest_eval  = float(eval_series.iloc[-1]) if not eval_series.empty else 0.0
_hc_rt = _ss.get('p_hc', DEFAULT_CONFIG['hc']) / 100
_lc_rt = _ss.get('p_lc', DEFAULT_CONFIG['lc']) / 100
latest_tier  = 'HIGH' if latest_eval >= _hc_rt else ('LOW' if latest_eval <= _lc_rt else 'MID')
st.title("TQQQ [위대리] v5.2 : 균형형 트레이딩 시스템")
tab1, tab2, tab3 = st.tabs(["실전 트레이딩", "백테스트 분석", "전략 로직"])
# ═══════════════════════════════════════════════════════════════
# TAB 1 — 실전 트레이딩
# ═══════════════════════════════════════════════════════════════
with tab1:
    if log_df.empty:
        st.warning("매매 데이터가 없습니다. 시작일 또는 설정을 확인해 주세요.")
        st.stop()
    last = log_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("시장 모드",     last['티어'],                  last['시장평가'])
    c2.metric("TQQQ 현재가",  f"${latest_tqqq:.2f}")
    c3.metric("현재 총 자산", f"${last['총자산']:,.2f}")
    if _ss.get('p_use_gate', False):
        c4.metric("추세 게이트", last['게이트'])
    else:
        c4.metric("매매 주기",    "금요일 (주간 LOC)")
    if _ss.get('p_use_gate', False) and last['게이트'] == '현금(이탈)':
        st.error("**추세 게이트 OFF — QQQ가 200일선 아래.** 위대리는 전량 현금 대기 중입니다. "
                 "매수 신호를 내지 않으며, QQQ가 200일선을 회복하면 재진입합니다.", icon="🛡️")
    st.subheader(f"금주 장 마감(LOC) 주문표 ({datetime.now().strftime('%Y-%m-%d')})")
    b_col, s_col = st.columns(2)
    with b_col:
        val = f"수량: {last['주문수량']} 주" if last['액션'] in ("매수", "게이트재진입") else "대기 (신호 없음)"
        st.markdown(
            f'<div class="order-card-buy"><h4>LOC 매수 주문</h4>'
            f'<h1 style="color:#188038;">{val}</h1></div>',
            unsafe_allow_html=True)
    with s_col:
        val = f"수량: {abs(last['주문수량'])} 주" if last['액션'] in ("매도", "게이트청산") else "대기 (신호 없음)"
        st.markdown(
            f'<div class="order-card-sell"><h4>LOC 매도 주문</h4>'
            f'<h1>{val}</h1></div>',
            unsafe_allow_html=True)
    st.divider()
    # ── 양도세 납부 관리 ─────────────────────────
    _now = datetime.now()
    _cur_year = _now.year
    _cur_month = _now.month
    _is_tax_window = (_cur_month in [1, 5]) and (_now.day <= 14)
    _tax_strategy_label = "1월/5월 50/50 분할 (전략 A)"

    _tax_payments, _tax_load_err = load_tax_payments()
    _total_paid_usd = sum(p['amount_usd'] for p in _tax_payments)
    _paid_for_year_curr = sum(p['amount_usd'] for p in _tax_payments
                               if p['for_year'] == _cur_year - 1)

    if _is_tax_window:
        st.warning(
            f"**양도세 납부 시기 알림** — {_now.strftime('%Y년 %m월')}\n\n"
            f"전략 **{_tax_strategy_label}** 기준, {_cur_year - 1}년 분 양도세 납부 권장 시기입니다. "
            f"현재까지 {_cur_year - 1}년 분으로 기록된 납부액: **${_paid_for_year_curr:,.2f}**\n\n"
            f"실제 납부 후 아래 폼에 기록하면 봇이 자동으로 가상 잔고에서 차감합니다.",
            icon="🧾",
        )

    with st.expander("양도세 납부 관리", expanded=_is_tax_window):
        with st.container(border=True):
            cc_a, cc_b = st.columns([2, 1])
            with cc_a:
                _tax_apply_now = st.checkbox(
                    "봇의 가상 잔고에 양도세 차감 반영",
                    value=bool(st.session_state.get('p_tax_apply_to_bot', False)),
                    key='p_tax_apply_to_bot',
                    help="OFF (기본): 봇은 세전 가상 잔고로 매매 시그널을 계산.\n"
                         "ON: 봇이 이 시트의 양도세 납부 기록을 시뮬에 반영해 cash 를 차감.\n"
                         "-> 운용 계획이 이미 세워져 있다면 OFF, 봇 잔고와 실잔고를 정확히 맞추고 싶으면 ON.",
                )
            with cc_b:
                if st.button("봇에 반영 (저장)", use_container_width=True,
                             key='btn_save_tax_apply'):
                    cfg = save_config(st.session_state)
                    ok, err = save_config_to_sheets(cfg)
                    if ok:
                        st.toast(
                            f"저장 완료. 봇 양도세 차감: "
                            f"{'ON' if _tax_apply_now else 'OFF'}",
                            icon="💾",
                        )
                    else:
                        st.toast("로컬 저장 완료 (Sheets 미동기화)", icon="💾")
            st.caption(
                f"현재 상태: **{'봇이 양도세 반영 중' if _tax_apply_now else '봇은 양도세 무시 (세전 시뮬)'}**"
                + (" — Streamlit 화면도 후처리로 차감 표시" if _tax_apply_now
                   else " — Streamlit 후처리 차감 비활성")
            )

        col_p1, col_p2 = st.columns([1, 1])

        with col_p1:
            st.markdown("**새 납부 기록 추가**")
            with st.form("tax_payment_form", clear_on_submit=True):
                p_date = st.date_input("납부일", value=_now.date(),
                                       key='p_tax_pay_date')
                p_year = st.number_input("대상 연도",
                                         value=_cur_year - 1,
                                         min_value=2010, max_value=_cur_year,
                                         step=1, key='p_tax_for_year',
                                         help="어느 연도의 양도세인지")
                p_input_mode = st.radio("입력 방식", ["KRW", "USD"],
                                         horizontal=True, key='p_tax_input_mode')
                col_amt, col_fx = st.columns(2)
                with col_amt:
                    if p_input_mode == "KRW":
                        p_amount_krw = st.number_input(
                            "납부액 (KRW)", value=0, min_value=0, step=10000,
                            key='p_tax_krw',
                        )
                    else:
                        p_amount_usd_in = st.number_input(
                            "납부액 ($)", value=0.0, min_value=0.0, step=10.0,
                            key='p_tax_usd',
                        )
                with col_fx:
                    p_fx = st.number_input(
                        "환율 (KRW/USD)", value=1300.0, min_value=500.0,
                        step=10.0, key='p_tax_fx',
                        help="USD <-> KRW 변환에 사용",
                    )
                p_note = st.text_input("메모 (선택)", value="",
                                        placeholder="예: 1차 납부 (50%)",
                                        key='p_tax_note')
                submitted = st.form_submit_button("납부 기록 저장",
                                                  use_container_width=True)
                if submitted:
                    if p_input_mode == "KRW":
                        p_amount_krw_val = float(p_amount_krw)
                        p_amount_usd_val = p_amount_krw_val / p_fx if p_fx > 0 else 0
                    else:
                        p_amount_usd_val = float(p_amount_usd_in)
                        p_amount_krw_val = p_amount_usd_val * p_fx
                    if p_amount_usd_val <= 0:
                        st.error("납부액이 0 입니다. 다시 확인하세요.")
                    else:
                        ok, err = save_tax_payment({
                            'date':       p_date.strftime("%Y-%m-%d"),
                            'for_year':   int(p_year),
                            'amount_usd': round(p_amount_usd_val, 2),
                            'amount_krw': round(p_amount_krw_val, 0),
                            'fx_rate':    p_fx,
                            'note':       p_note,
                        })
                        if ok:
                            st.success(
                                f"저장 완료: {p_date} | {p_year}년 분 | "
                                f"KRW {p_amount_krw_val:,.0f} (~${p_amount_usd_val:,.2f}) "
                                f"| 봇이 다음 실행 시 자동 반영"
                            )
                            st.rerun()
                        else:
                            st.error(f"저장 실패: {err}")

        with col_p2:
            st.markdown("**납부 기록**")
            if _tax_load_err:
                st.error(f"시트 로드 실패: {_tax_load_err}")
            elif not _tax_payments:
                st.info("아직 기록된 납부가 없습니다.")
            else:
                _df_pay = pd.DataFrame([{
                    '납부일':      p['date'],
                    '대상 연도':   p['for_year'],
                    '금액 ($)':    f"${p['amount_usd']:,.2f}",
                    '금액 (KRW)':  f"{p['amount_krw']:,.0f}",
                    '환율':        f"{p['fx_rate']:.0f}",
                    '메모':        p['note'],
                } for p in _tax_payments])
                st.dataframe(_df_pay, use_container_width=True, hide_index=True)
                st.caption(
                    f"누적 납부: **${_total_paid_usd:,.2f}** "
                    f"({len(_tax_payments)}건) — 봇이 가상 잔고에서 차감"
                )

                _del_dates = ['(선택 안 함)'] + [p['date'] for p in _tax_payments]
                _del_target = st.selectbox("삭제할 기록 (납부일 기준)", _del_dates,
                                            key='p_tax_delete_target')
                if _del_target != '(선택 안 함)':
                    if st.button(f"{_del_target} 기록 삭제",
                                  use_container_width=True):
                        ok, err = delete_tax_payment(_del_target)
                        if ok:
                            st.success("삭제 완료")
                            st.rerun()
                        else:
                            st.error(f"삭제 실패: {err}")

    # ── 계좌 현황 (양도세 차감은 토글 ON 일 때만) ──────────────
    st.subheader("내 계좌 현황")
    _tax_apply_now = bool(st.session_state.get('p_tax_apply_to_bot', False))
    _eff_paid_usd = _total_paid_usd if _tax_apply_now else 0.0
    _adjusted_total = float(last['총자산']) - _eff_paid_usd
    _adjusted_cash  = float(last['현금'])    - _eff_paid_usd
    _adjusted_cash_pct = (_adjusted_cash / _adjusted_total * 100
                          if _adjusted_total > 0 else 0)
    _adjusted_pnl_pct = (_adjusted_total / st_cap - 1) * 100 if st_cap > 0 else 0
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("보유 수량",  f"{last['보유수량']:,} 주")
    a2.metric("보유 현금",  f"${_adjusted_cash:,.2f}",
              f"-양도세 ${_eff_paid_usd:,.0f}" if _eff_paid_usd > 0 else None)
    a3.metric("수익률" + (" (세후)" if _eff_paid_usd > 0 else ""),
              f"{_adjusted_pnl_pct:+.2f}%",
              f"세전 {last['수익률']}" if _eff_paid_usd > 0 else None)
    a4.metric("평가 금액",  f"${last['보유수량'] * latest_tqqq:,.2f}")
    a5.metric("현금 비중",  f"{_adjusted_cash_pct:.2f}%")
    if _eff_paid_usd > 0:
        st.caption(
            f"누적 양도세 납부 ${_eff_paid_usd:,.2f} 가 현금에서 차감되어 표시됩니다 "
            f"(봇 차감 토글 ON)."
        )
    elif _total_paid_usd > 0 and not _tax_apply_now:
        st.caption(
            f"양도세 납부 기록 ${_total_paid_usd:,.2f} 이 있으나 "
            f"*봇 차감 토글이 OFF* 이라 화면/봇 모두 세전 잔고를 표시 중입니다."
        )
    with st.expander("상세 매매 로그", expanded=True):
        st.dataframe(log_df.iloc[::-1], use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# TAB 2 — 백테스트 분석
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("위대리 전략 전체 기간 백테스트")
    st.caption("기본 2010년부터. '2010년 이전 합성 데이터'를 켜면 닷컴(2000)·2008까지 확장됩니다.")
    OPT_PARAMS = dict(hc=0.06, lc=-0.06, sH=2.0, sM=0.3, sL=0.2, bH=1.0, bM=0.6, bL=2.0)
    bt_hc, bt_lc = OPT_PARAMS['hc'], OPT_PARAMS['lc']
    bt_sH, bt_sM, bt_sL = OPT_PARAMS['sH'], OPT_PARAMS['sM'], OPT_PARAMS['sL']
    bt_bH, bt_bM, bt_bL = OPT_PARAMS['bH'], OPT_PARAMS['bM'], OPT_PARAMS['bL']

    with st.container(border=True):
        st.markdown("**백테스트 설정**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            bt_cap = st.number_input("초기 자본 ($)", value=10_000, step=1000,
                                     key='p_bt_cap')
        with col_b:
            _synth_on = bool(st.session_state.get('p_bt_use_synth', False))
            _bt_min = datetime(1999, 3, 10).date() if _synth_on else datetime(2010, 2, 11).date()
            bt_start_date = st.date_input(
                "검증 시작일",
                value=datetime.strptime("2010-01-01", "%Y-%m-%d").date(),
                min_value=_bt_min, max_value=datetime.now().date(),
                key='p_bt_start')
        with col_c:
            bt_end_date = st.date_input(
                "검증 종료일", value=datetime.now().date(),
                key='p_bt_end')
        bt_cash_ratio = st.slider(
            "초기 현금 비중 (%)", min_value=0, max_value=100,
            value=45, step=1, key='p_bt_cash',
            help="시작 시점 현금 vs 주식 비중. 기본 45% (검증된 균형점). "
                 "낮출수록 초기 stock 노출 up -> 강세장 수익 up but 첫 약세장 MDD up"
        ) / 100
        # ── v5.2: 200일 게이트 + 프리-2010 합성 데이터 토글 ──
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            bt_use_gate = st.checkbox(
                "200일 추세 게이트 적용", value=False, key='p_bt_use_gate',
                help="QQQ가 200일선 이탈 시 전량 현금화 후 재진입 대기. "
                     "닷컴/2008/2022 레짐 붕괴 꼬리를 절단(강세장 CAGR 일부 반납).")
            bt_compare_gate = st.checkbox(
                "게이트 ON/OFF 동시 비교", value=False, key='p_bt_compare_gate',
                help="게이트 적용/미적용을 동시에 백테스트해 나란히 비교")
        with gcol2:
            bt_use_synth = st.checkbox(
                "2010년 이전 합성 데이터 사용", value=False, key='p_bt_use_synth',
                help="TQQQ 상장(2010) 이전을 QQQ 3배 합성으로 채워 닷컴(2000-02)·2008까지 검증. "
                     "※ 변동성 잠식 정확·금융비용 근사 → 절대수익보다 게이트 유무 상대비교 용도.\n"
                     "체크 후 '검증 시작일'을 1999-03 까지 선택 가능 (재실행 시 반영).")
        st.caption(
            f"전략 파라미터: 최적화 기본값 고정 "
            f"(hc=+-6%, 매도 H/M/L = 2.0/0.3/0.2, 매수 H/M/L = 1.0/0.6/2.0)"
        )

    # ── v4.8: OOS-검증된 추가 신호 ──
    with st.expander("v4.8 OOS-검증 개선 (Task A + Task B)", expanded=False):
        v48_bt_on = st.checkbox(
            "v4.8 개선 적용", value=True, key='p_bt_v48_enable',
            help="OOS 워크포워드 7-fold +0.75%p CAGR robust 검증. 끄면 v4.7 동작"
        )
        v48c1, v48c2 = st.columns(2)
        with v48c1:
            bt_dec_scale = st.number_input(
                "12월 매도 축소율 (dec_sell_scale)", min_value=0.0, max_value=1.0,
                value=0.75, step=0.05, key='p_bt_dec_scale',
                disabled=not v48_bt_on,
                help="OOS 최적값 0.75"
            )
            bt_ma_w = st.number_input(
                "MA 윈도우 (주)", min_value=4, max_value=26, value=13, step=1,
                key='p_bt_ma_w', disabled=not v48_bt_on,
                help="OOS 최적값 13주"
            )
        with v48c2:
            bt_thr = st.number_input(
                "추세 임계값 xMA", min_value=1.0, max_value=1.20,
                value=1.08, step=0.01, key='p_bt_thr',
                disabled=not v48_bt_on,
                help="QQQ가 MA x 이 값 이상이면 멜트업. OOS 최적값 1.08"
            )
            bt_mult = st.number_input(
                "HIGH 매도 축소율", min_value=0.0, max_value=1.0,
                value=0.70, step=0.05, key='p_bt_mult',
                disabled=not v48_bt_on,
                help="OOS 최적값 0.70"
            )
    v48_kwargs = dict(
        dec_sell_scale       = bt_dec_scale if v48_bt_on else 1.0,
        ma_window_weeks      = bt_ma_w     if v48_bt_on else 0,
        trend_threshold      = bt_thr      if v48_bt_on else 99.0,
        sell_rate_multiplier = bt_mult     if v48_bt_on else 1.0,
    )
    gate_kwargs = dict(use_gate=bt_use_gate, gate_sma_days=200)
    # ── 거래비용 + 양도세 옵션 ─────────────────────────────────
    with st.expander("거래비용 & 양도세 (현실 시뮬레이션)", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**거래 수수료**")
            apply_comm = st.checkbox("수수료 적용", value=False, key='p_apply_comm',
                                     help="매수/매도 시점에 수수료 차감")
            comm_buy_pct  = st.number_input("매수 수수료 (%)", value=0.015, step=0.001, format="%.3f",
                                            key='p_comm_buy', disabled=not apply_comm)
            comm_sell_pct = st.number_input("매도 수수료 + SEC fee (%)", value=0.01706, step=0.001, format="%.5f",
                                            key='p_comm_sell', disabled=not apply_comm,
                                            help="기본 0.01706% = 증권사 0.015% + SEC fee 0.00206%")
        with cc2:
            st.markdown("**양도세 (한국 거주자)**")
            apply_tax = st.checkbox("양도세 적용", value=False, key='p_apply_tax',
                                    help="직전 연도 실현이익에 대해 인출 전략에 따라 정산")
            tax_dedu  = st.number_input("연간 공제 (USD)", value=1923.0, step=100.0,
                                        key='p_tax_dedu', disabled=not apply_tax,
                                        help="기본 $1923 ~ 2.5M KRW / FX 1300 — FX 변동 시 조정")
            tax_rate_pct = st.number_input("세율 (%)", value=22.0, step=0.5,
                                           key='p_tax_rate', disabled=not apply_tax)
            tax_strategy_label = st.selectbox(
                "양도세 인출 전략",
                ["A - 1월/5월 50/50 분할 (기본)",
                 "B - 5월 일괄",
                 "C - 1월 일괄"],
                key='p_tax_strategy', disabled=not apply_tax,
                help="A 분할 전략은 세금 부담을 후반기로 분산해 그 사이 기간의 운용 자산이 늘어남"
            )
            tax_strategy = tax_strategy_label[0]
            tax_rebal = st.checkbox("매도-only 리밸런싱 (옵션 B)", value=True,
                                    key='p_tax_rebal', disabled=not apply_tax,
                                    help="납부 후 cash 가 부족할 때만 stock 일부 매도. cash 과다 시 강제 매수는 안 함.")
        col_cm1, col_cm2 = st.columns(2)
        compare_costs = col_cm1.checkbox(
            "Gross vs Net 비교 (적용 ON vs OFF)",
            value=False, key='p_compare_costs',
            help="거래비용/세금 적용/미적용 동시 백테스트")
        compare_strategies = col_cm2.checkbox(
            "3가지 인출 전략 동시 비교 (A/B/C)",
            value=False, key='p_compare_strategies', disabled=not apply_tax,
            help="A=1월/5월 50/50, B=5월 일괄, C=1월 일괄 동시 비교")

    # v5.2: 합성 데이터 토글에 따라 소스 선택 (라이브 df 는 실측; 백테스트만 합성 허용)
    df_source = load_wedaeri_data(extended=True) if bt_use_synth else df
    df_bt = df_source[df_source['Date'] <= pd.to_datetime(bt_end_date)].reset_index(drop=True)
    if bt_use_synth and 'synthetic' in df_source.columns and df_source['synthetic'].any():
        _syn_end = df_source.loc[df_source['synthetic'], 'Date'].max()
        st.info(
            f"⚠️ 합성 TQQQ 구간 포함 (~{_syn_end.strftime('%Y-%m')}까지, QQQ 3배 근사). "
            f"변동성 잠식은 정확하나 금융비용은 근사치입니다 — 절대수익보다 "
            f"**게이트 유무 상대비교**로 해석하세요.",
            icon="🧪")

    with st.spinner("백테스트 계산 중..."):
        cost_kwargs = dict(
            apply_commission = apply_comm,
            comm_buy   = comm_buy_pct  / 100,
            comm_sell  = comm_sell_pct / 100,
            apply_tax  = apply_tax,
            tax_deduction_usd = tax_dedu,
            tax_rate   = tax_rate_pct / 100,
            tax_rebalance = tax_rebal,
            tax_strategy = tax_strategy,
        )
        bt_cur = run_full_backtest(
            df_bt, bt_cap, bt_cash_ratio,
            hc=bt_hc, lc=bt_lc,
            sH=bt_sH, sM=bt_sM, sL=bt_sL,
            bH=bt_bH, bM=bt_bM, bL=bt_bL,
            start_date=bt_start_date,
            **cost_kwargs,
            **v48_kwargs,
            **gate_kwargs,
        )
        # v5.2: 게이트 ON/OFF 동시 비교 → 반대 상태도 계산
        bt_alt = None
        if bt_compare_gate:
            bt_alt = run_full_backtest(
                df_bt, bt_cap, bt_cash_ratio,
                hc=bt_hc, lc=bt_lc,
                sH=bt_sH, sM=bt_sM, sL=bt_sL,
                bH=bt_bH, bM=bt_bM, bL=bt_bL,
                start_date=bt_start_date,
                **cost_kwargs,
                **v48_kwargs,
                use_gate=(not bt_use_gate), gate_sma_days=200,
            )
        bt_gross = None
        if compare_costs and (apply_comm or apply_tax):
            bt_gross = run_full_backtest(
                df_bt, bt_cap, bt_cash_ratio,
                hc=bt_hc, lc=bt_lc,
                sH=bt_sH, sM=bt_sM, sL=bt_sL,
                bH=bt_bH, bM=bt_bM, bL=bt_bL,
                start_date=bt_start_date,
                apply_commission=False, apply_tax=False,
                **v48_kwargs,
                **gate_kwargs,
            )

        bt_strategies = None
        if compare_strategies and apply_tax:
            bt_strategies = {}
            for code in ['A', 'B', 'C']:
                kw = dict(cost_kwargs); kw['tax_strategy'] = code
                bt_strategies[code] = run_full_backtest(
                    df_bt, bt_cap, bt_cash_ratio,
                    hc=bt_hc, lc=bt_lc,
                    sH=bt_sH, sM=bt_sM, sL=bt_sL,
                    bH=bt_bH, bM=bt_bM, bL=bt_bL,
                    start_date=bt_start_date,
                    **kw,
                    **v48_kwargs,
                    **gate_kwargs,
                )
    if bt_cur is None:
        st.warning("백테스트 데이터가 부족합니다.")
        st.stop()
    dates = pd.to_datetime(bt_cur['dates'])
    ev = bt_cur['eval']
    n_h = int((ev >= bt_hc).sum()); n_l = int((ev <= bt_lc).sum())
    n_m = len(ev) - n_h - n_l
    with st.expander("Eval(시장평가) 진단 — 티어 배분 확인", expanded=False):
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("중앙값",  f"{float(np.median(ev)):.2%}")
        d2.metric("p10",    f"{float(np.percentile(ev,10)):.2%}")
        d3.metric("p90",    f"{float(np.percentile(ev,90)):.2%}")
        d4.metric("HIGH 비중", f"{n_h/len(ev):.1%}")
        d5.metric("LOW 비중",  f"{n_l/len(ev):.1%}")
        st.caption(
            f"HIGH {n_h}주 / MID {n_m}주 / LOW {n_l}주 | "
            f"스프레드시트 기준 약 HIGH 29% / MID 55% / LOW 17%"
        )
    st.markdown("### 성과 지표")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CAGR",         f"{bt_cur['cagr']:.2%}")
    m2.metric("MDD",          f"{bt_cur['mdd']:.2%}", delta_color="inverse")
    m3.metric("Calmar 비율",  f"{bt_cur['cal']:.2f}")
    m4.metric("Sortino 비율", f"{bt_cur['sor']:.2f}")
    m5.metric("최종 자산",    f"${bt_cur['final']:,.0f}",
              f"초기 ${bt_cur['init']:,.0f}")
    st.caption(
        f"백테스트 기간: {dates[0].strftime('%Y.%m.%d')} ~ {dates[-1].strftime('%Y.%m.%d')}"
        f"  ({bt_cur['years']:.1f}년) | 초기 자본 ${bt_cap:,} | 초기 현금 {bt_cash_ratio:.0%} | "
        f"게이트 {'ON' if bt_use_gate else 'OFF'} / 수수료 {'ON' if apply_comm else 'OFF'} / 양도세 {'ON' if apply_tax else 'OFF'}"
    )
    # ── v5.2: 게이트 ON/OFF 비교 표 ─────────────────────────────
    if bt_compare_gate and bt_alt is not None:
        st.markdown("### 게이트 ON/OFF 비교")
        _on  = bt_cur if bt_use_gate else bt_alt
        _off = bt_alt if bt_use_gate else bt_cur
        cmp_g = pd.DataFrame({
            '구분':   ['게이트 ON', '게이트 OFF', '차이(ON-OFF)'],
            'CAGR':   [f"{_on['cagr']:.2%}", f"{_off['cagr']:.2%}",
                       f"{(_on['cagr']-_off['cagr'])*100:+.2f}%p"],
            'MDD':    [f"{_on['mdd']:.2%}",  f"{_off['mdd']:.2%}",
                       f"{(_on['mdd']-_off['mdd'])*100:+.2f}%p"],
            'Calmar': [f"{_on['cal']:.2f}",  f"{_off['cal']:.2f}",
                       f"{_on['cal']-_off['cal']:+.2f}"],
            '최종 자산': [f"${_on['final']:,.0f}", f"${_off['final']:,.0f}",
                          f"${_on['final']-_off['final']:+,.0f}"],
        })
        st.dataframe(cmp_g, use_container_width=True, hide_index=True)
        st.caption(
            "게이트 ON은 위기 MDD를 크게 줄이는 대신 강세장 CAGR을 반납합니다 "
            "(검증: 위험조정 우월 전략이 아니라 선택적 재난보험 — 뉴노멀을 믿으면 OFF, "
            "2022급 재발 대비면 ON)."
        )
    # ── 거래비용/세금 요약 + Gross vs Net 비교 ─────────────────
    if apply_comm or apply_tax:
        st.markdown("### 거래비용 & 양도세 요약")
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("누적 수수료",    f"${bt_cur['cum_commission']:,.0f}")
        cs2.metric("누적 양도세",    f"${bt_cur['cum_tax']:,.0f}")
        total_cost = bt_cur['cum_commission'] + bt_cur['cum_tax']
        cs3.metric("총 비용 합계",   f"${total_cost:,.0f}")
        cs4.metric("최종 자산 대비", f"{total_cost / max(bt_cur['final'], 1) * 100:.2f}%")

        if bt_gross is not None:
            st.markdown("**Gross (비용/세금 0) vs Net (현재 설정)**")
            gn = pd.DataFrame({
                '구분': ['Gross', 'Net (현재)', '차이'],
                'CAGR': [f"{bt_gross['cagr']:.2%}", f"{bt_cur['cagr']:.2%}",
                         f"{(bt_cur['cagr']-bt_gross['cagr'])*100:+.2f}%p"],
                'MDD':  [f"{bt_gross['mdd']:.2%}",  f"{bt_cur['mdd']:.2%}",
                         f"{(bt_cur['mdd']-bt_gross['mdd'])*100:+.2f}%p"],
                'Calmar': [f"{bt_gross['cal']:.2f}", f"{bt_cur['cal']:.2f}",
                           f"{bt_cur['cal']-bt_gross['cal']:+.2f}"],
                '최종 자산': [f"${bt_gross['final']:,.0f}", f"${bt_cur['final']:,.0f}",
                              f"${bt_cur['final']-bt_gross['final']:+,.0f}"],
            })
            st.dataframe(gn, use_container_width=True, hide_index=True)

        if apply_tax and bt_cur.get('tax_events'):
            st.markdown("**개별 납부 이벤트 내역**")
            tax_df = pd.DataFrame([{
                '납부일':       e['date'],
                '대상 연도':    e['prev_year'],
                '연 실현이익':  f"${e['gain']:,.0f}",
                '연 과세표준':  f"${e['taxable']:,.0f}",
                '이번 납부액':  f"${e['tax']:,.0f}",
                '예정 금액':    f"${e.get('orig_tax', e['tax']):,.0f}",
                '세금 충당 매도': f"{e['sold_for_tax']}주",
            } for e in bt_cur['tax_events']])
            st.dataframe(tax_df, use_container_width=True, hide_index=True)
            unpaid = bt_cur.get('unpaid_tax', 0.0)
            if unpaid > 0:
                st.warning(f"미납 세액 ${unpaid:,.0f} — 백테스트 종료 시점에 예정 납부일이 도래하지 않음. "
                           f"실제 운용에서는 이후 납부됨.")

        if bt_strategies is not None:
            st.markdown("### 양도세 인출 전략 3종 비교")
            srows = []
            for code in ['A', 'B', 'C']:
                r = bt_strategies[code]
                meta = TAX_SCHEDULES[code]
                srows.append({
                    '전략':    f"{code}. {meta['name']}",
                    'CAGR':   f"{r['cagr']:.2%}",
                    'MDD':    f"{r['mdd']:.2%}",
                    'Calmar': f"{r['cal']:.2f}",
                    'Sortino':f"{r['sor']:.2f}",
                    '최종 자산': f"${r['final']:,.0f}",
                    '누적 세금': f"${r['cum_tax']:,.0f}",
                    '미납': f"${r.get('unpaid_tax', 0):,.0f}",
                })
            st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)

            best_code = max(bt_strategies.keys(),
                            key=lambda c: bt_strategies[c]['final'])
            best_r = bt_strategies[best_code]
            worst_code = min(bt_strategies.keys(),
                             key=lambda c: bt_strategies[c]['final'])
            worst_r = bt_strategies[worst_code]
            diff_pct = (best_r['final'] / worst_r['final'] - 1) * 100
            st.success(
                f"**최종 자산 1위: 전략 {best_code} ({TAX_SCHEDULES[best_code]['name']})** "
                f"— ${best_r['final']:,.0f}\n\n"
                f"최하: 전략 {worst_code} ({TAX_SCHEDULES[worst_code]['name']}) "
                f"— ${worst_r['final']:,.0f} (1위 대비 {-diff_pct:.2f}% 적음)"
            )

            fig_strat = go.Figure()
            colors = {'A':'#60a5fa','B':'#22c55e','C':'#fbbf24'}
            for code in ['A', 'B', 'C']:
                r = bt_strategies[code]
                meta = TAX_SCHEDULES[code]
                fig_strat.add_trace(go.Scatter(
                    x=pd.to_datetime(r['dates']), y=r['eq'],
                    name=f"{code}. {meta['name']}",
                    line=dict(color=colors[code], width=2),
                ))
            fig_strat.update_layout(
                title='인출 전략별 — OOS 누적 자산 곡선 (log)',
                yaxis=dict(title='자산 ($)', type='log'),
                height=380, **CHART_LAYOUT
            )
            apply_grid(fig_strat)
            st.plotly_chart(fig_strat, use_container_width=True)

            st.caption(
                "분할 납부 (A/C) 는 후반 납부분이 *그 사이 기간 동안 운용 중* 이라 복리 효과로 "
                "최종 자산이 일괄(B) 보다 약간 높을 수 있어요. 다만 2~5월 시장 변동성에 노출되니 "
                "단기 MDD 도 함께 비교하세요."
            )

    st.markdown("### 성과 비교 (위대리 vs B&H)")
    compare = pd.DataFrame({
        '전략':      ['위대리', 'TQQQ B&H'],
        'CAGR':     [f"{bt_cur['cagr']:.2%}",  f"{bt_cur['bh_cagr']:.2%}"],
        'MDD':      [f"{bt_cur['mdd']:.2%}",   f"{bt_cur['bh_mdd']:.2%}"],
        'Calmar':   [f"{bt_cur['cal']:.2f}",   "—"],
        'Sortino':  [f"{bt_cur['sor']:.2f}",   "—"],
        '최종 자산': [f"${bt_cur['final']:,.0f}", f"${bt_cur['bh_eq'][-1]:,.0f}"],
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### 누적 자산 곡선 (로그 스케일)")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=dates, y=bt_cur['eq'], name='위대리' + (' (게이트 ON)' if bt_use_gate else ''),
        line=dict(color='#60a5fa', width=2.5)
    ))
    fig_eq.add_trace(go.Scatter(
        x=dates, y=bt_cur['bh_eq'], name='TQQQ B&H',
        line=dict(color='#fb923c', width=1.5, dash='dash'), opacity=0.7
    ))
    if bt_compare_gate and bt_alt is not None:
        fig_eq.add_trace(go.Scatter(
            x=dates, y=bt_alt['eq'],
            name='위대리 (게이트 ' + ('OFF' if bt_use_gate else 'ON') + ')',
            line=dict(color='#22c55e', width=1.5, dash='dot'), opacity=0.75
        ))
    if bt_gross is not None:
        fig_eq.add_trace(go.Scatter(
            x=dates, y=bt_gross['eq'], name='Gross (비용/세금 0)',
            line=dict(color='#a855f7', width=1.5, dash='dot'), opacity=0.6
        ))
    fig_eq.add_hline(
        y=bt_cap, line_color='#475569', line_dash='dot', line_width=1,
        annotation_text=f"초기자본 ${bt_cap:,}",
        annotation_font=dict(color='#94a3b8', size=10)
    )
    fig_eq.update_layout(
        title='전략별 자산 추이',
        yaxis=dict(title='자산 ($)', type='log'),
        height=400, **CHART_LAYOUT
    )
    apply_grid(fig_eq)
    st.plotly_chart(fig_eq, use_container_width=True)
    st.markdown("### 티어별 성과 분석")
    tiers_arr = np.array(bt_cur['tiers'])
    wr = bt_cur['rets'] * 100
    wr_full   = np.concatenate([[0], wr]) if len(wr) > 0 else np.array([0.0])
    tier_rows = []
    for name, icon in [('HIGH', 'H'), ('MID', 'M'), ('LOW', 'L')]:
        mask = tiers_arr == name
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        t_rets = wr_full[mask]
        tier_rows.append({
            '티어':    f'{name}',
            '발생 주': f"{cnt}주 ({cnt/len(tiers_arr):.1%})",
            '평균 수익률': f"{t_rets.mean():+.2f}%",
            '최대 상승': f"{t_rets.max():+.2f}%",
            '최대 하락': f"{t_rets.min():+.2f}%",
            '양전 비율': f"{(t_rets > 0).mean():.1%}",
        })
    if tier_rows:
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)
    st.markdown("### 연도별 상세 성과")
    if bt_cur.get('yearly'):
        yr_df = pd.DataFrame(bt_cur['yearly'])
        st.dataframe(yr_df, use_container_width=True, hide_index=True)
        yr_vals = [float(r['수익률'].replace('%', '').replace('+', '')) for r in bt_cur['yearly']]
        yr_mdds = [float(r['연간 MDD'].replace('%', '')) for r in bt_cur['yearly']]
        yr_labels = [str(r['연도']) for r in bt_cur['yearly']]
        bar_colors = ['#ef4444' if v < 0 else '#14b8a6' for v in yr_vals]
        y_max = max(yr_vals + [0]) * 1.30 + 10
        y_min = min(yr_mdds + [0]) - 12
        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(
            x=yr_labels, y=yr_vals,
            marker_color=bar_colors,
            name='Annual Return',
            text=[f"{v:+.0f}%" for v in yr_vals],
            textposition='outside',
            textfont=dict(size=10, color='#cbd5e1'),
            cliponaxis=False,
        ))
        fig_yr.add_trace(go.Scatter(
            x=yr_labels, y=yr_mdds,
            mode='lines+markers+text',
            line=dict(color='#fb923c', width=2),
            marker=dict(size=8, color='#fb923c'),
            name='MDD',
            text=[f"{v:.0f}%" for v in yr_mdds],
            textposition='bottom center',
            textfont=dict(size=9, color='#fb923c'),
            cliponaxis=False,
        ))
        fig_yr.add_hline(y=0, line_color='#475569', line_width=1)
        fig_yr.update_layout(
            title=dict(text='연도별 수익률 & MDD', y=0.98),
            yaxis=dict(title='Return / MDD (%)', range=[y_min, y_max]),
            height=420,
            margin=dict(l=10, r=10, t=70, b=20),
            paper_bgcolor=CHART_LAYOUT['paper_bgcolor'],
            plot_bgcolor=CHART_LAYOUT['plot_bgcolor'],
            font=CHART_LAYOUT['font'],
            legend=dict(orientation='h', y=1.06, font=dict(size=10)),
        )
        apply_grid(fig_yr)
        st.plotly_chart(fig_yr, use_container_width=True)

    # ── 전체 매매 로그 (분석용 완전 패널) ───────────────────
    st.markdown("### 전체 매매 로그 (분석용 완전 패널)")
    if bt_cur.get('trade_log'):
        log = bt_cur['trade_log']
        raw_df = pd.DataFrame(log)

        n_buy  = int((raw_df['action'] == 'BUY').sum())
        n_sell = int((raw_df['action'] == 'SELL').sum())
        n_hold = int((raw_df['action'] == 'HOLD').sum())
        n_tax  = int((raw_df['action'] == 'TAX').sum())
        st.caption(
            f"총 {len(raw_df):,}행 기록 "
            f"(매수 {n_buy:,} / 매도 {n_sell:,} / 관망 {n_hold:,}"
            f"{' / 양도세 ' + str(n_tax) if n_tax else ''}) "
            f"— 매주 1행 완전 패널. CSV 는 원본 숫자값(무포맷)으로 다운로드됩니다."
        )

        emoji_map = {'BUY': '🟢 매수', 'SELL': '🔴 매도',
                     'HOLD': '⚪ 관망', 'TAX': '🟡 양도세'}

        def _fmt_view(r):
            sq = int(r.get('sell_qty', 0)); bq = int(r.get('buy_qty', 0))
            rp = float(r.get('realized_pnl', 0.0))
            cm = float(r.get('commission', 0.0))
            tx = float(r.get('tax_paid', 0.0))
            return {
                'Date':      r['date'],
                'Action':    emoji_map.get(r['action'], r['action']),
                'Eval/Tier': f"{r['eval']*100:.2f}% ({r['tier']})",
                'Gate':      '보유' if int(r.get('gate_on', 1)) else '현금',
                'QQQ종가':    f"${r['qqq_close']:.2f}",
                'TQQQ종가':   f"${r['tqqq_close']:.2f}",
                '매수가':     f"${r['tqqq_close']:.2f}" if bq > 0 else "",
                '매수량':     f"{bq:,}"                if bq > 0 else "",
                '매도가':     f"${r['tqqq_close']:.2f}" if sq > 0 else "",
                '매도량':     f"{sq:,}"                if sq > 0 else "",
                '수익금':     f"${rp:,.2f}"             if rp != 0 else "",
                '수수료':     f"${cm:,.2f}"             if cm != 0 else "",
                '양도세':     f"${tx:,.2f}"             if tx != 0 else "",
                '평단가':     f"${r['avg_cost']:.2f}",
                '보유수량':   f"{int(r['shares']):,}",
                '현금':       f"${r['cash']:,.2f}",
                '총자산':     f"${r['total_asset']:,.2f}",
                '수익률':     f"{r['cum_return']*100:+.2f}%",
                'DD':        f"{r.get('drawdown', 0.0)*100:.1f}%",
                'Note':      r.get('note', ''),
            }

        col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
        with col_f1:
            action_filter = st.multiselect(
                "Action 필터",
                options=['BUY', 'SELL', 'TAX', 'HOLD'],
                default=['BUY', 'SELL', 'TAX'],
                format_func=lambda a: emoji_map.get(a, a),
                key='p_log_filter',
            )
        with col_f2:
            sort_desc = st.checkbox("최신순", value=True, key='p_log_sort')
        with col_f3:
            show_raw = st.checkbox("원본 숫자값", value=False, key='p_log_raw',
                                   help="포맷 없이 raw 숫자 컬럼 그대로 표시")

        sub = raw_df[raw_df['action'].isin(action_filter)].copy()
        if sort_desc:
            sub = sub.iloc[::-1]

        if show_raw:
            st.dataframe(sub, use_container_width=True, hide_index=True, height=600)
        else:
            view_df = pd.DataFrame([_fmt_view(r) for r in sub.to_dict('records')])
            st.dataframe(view_df, use_container_width=True, hide_index=True, height=600)
        st.caption(f"표시: {len(sub):,} 행 (전체 {len(raw_df):,} 행 중)")

        csv = raw_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="매매 로그 CSV 다운로드 (원본 숫자값 · 분석용)",
            data=csv,
            file_name=f"wedaeri_trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )

        with st.expander("로그 컬럼 설명 (분석용 데이터 사전)", expanded=False):
            st.markdown("""
매주 1행(관망 포함)씩 기록되는 완전 패널입니다. 양도세 정산 주에는 상태 행 뒤에
`action='TAX'` 행이 추가됩니다. 모든 값은 raw 숫자라 CSV 만으로 재현/검증이 가능합니다.

| 컬럼 | 의미 |
|---|---|
| `date` | 주간 마감일 (W-FRI) |
| `week` | 백테스트 시작 기준 주 인덱스 (0-base) |
| `action` | BUY / SELL / HOLD / TAX |
| `tier` | HIGH / MID / LOW (Eval 기준) |
| `eval` | QQQ 추세 대비 괴리율 (소수, 예 -0.0812 = -8.12%) |
| `qqq_close` | QQQ 종가 |
| `qqq_growth` | QQQ 5년 로그추세선의 당일 값 (Growth) |
| `qqq_ma` | QQQ 13주 이동평균 (Task B용) |
| `qqq_ma_ratio` | qqq_close / qqq_ma (멜트업 판정: >= trend_threshold) |
| `gate_on` | 200일 게이트 risk-on(1) / risk-off(0) — 게이트 미적용 백테스트에도 참고용 기록 |
| `tqqq_close` | TQQQ 종가 (= LOC 체결가) |
| `tqqq_prev` | 전주 TQQQ 종가 |
| `tqqq_wk_ret` | TQQQ 주간 수익률 (소수) |
| `signal_diff` | shares_prev x (금주-전주) — 매수/매도 규모를 결정하는 신호값 |
| `sell_rate` | 이번 주 실효 매도 배율 (모멘텀/12월 곱 반영 후) |
| `buy_rate` | 이번 주 실효 매수 배율 |
| `mom_filter` | HIGH 모멘텀 필터 발동 여부 (1/0) |
| `dec_scale` | 12월 매도 축소 적용 여부 (1/0) |
| `trade_price` | 체결가 (매매 없으면 NaN) |
| `buy_qty` / `sell_qty` | 매수/매도 체결 수량 |
| `realized_pnl` | 이 매도 이벤트의 실현손익 (수수료 반영, USD) |
| `commission` | 이 이벤트의 수수료 (USD) |
| `tax_paid` | 이 이벤트의 양도세 납부액 (USD, TAX 행) |
| `avg_cost` | 이벤트 후 가중평균 매입단가 |
| `shares` | 이벤트 후 보유 수량 |
| `cash` | 이벤트 후 현금 |
| `stock_value` | shares x tqqq_close |
| `total_asset` | cash + stock_value |
| `cum_return` | total_asset / 초기자본 - 1 (소수) |
| `peak` / `drawdown` | 로그 전체 기준 running peak 와 낙폭 (소수) |
| `note` | 비고 (양도세 대상연도, 게이트청산/재진입 등) |

**바로 써먹는 분석 예시 (pandas):**

```python
import pandas as pd
df = pd.read_csv("wedaeri_trade_log_YYYYMMDD.csv", parse_dates=["date"])

# 티어별 실현손익 분포
df[df.action=="SELL"].groupby("tier")["realized_pnl"].describe()

# 게이트가 현금(risk-off)이었던 주 확인
df[df.gate_on==0][["date","qqq_close","total_asset"]]

# 주간수익률로 CAGR/변동성 재계산해서 지표 재현 검증
w = df.drop_duplicates("date").set_index("date")["total_asset"]
ret = w.pct_change().dropna()
```

""")
# ═══════════════════════════════════════════════════════════════
# TAB 3 — 전략 로직
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 위대리 전략 완전 해설")
    st.caption("TQQQ Quantum T-Flow — QQQ 추세 기반 3-티어 자동 리밸런싱 시스템")
    st.markdown("""
### 전략 개요
**위대리 전략**은 나스닥 100(QQQ)의 5년 장기 추세와 현재 가격의 괴리를 측정해,
시장이 **과열(HIGH)** 이면 수익을 실현하고, **침체(LOW)** 이면 공격적으로 매수하는
**역추세 + 추세추종 혼합 리밸런싱 전략**입니다.
> 핵심 철학: *"시장 온도계(Eval)가 낮을수록 더 많이 사고, 높을수록 더 많이 판다."*
> 매주 금요일 LOC 주문 1회 — 감정 없이 기계적으로 실행합니다.
""")
    st.divider()
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        st.markdown("""
### Step 1. QQQ 시장 평가(Eval) 계산
**Eval**은 QQQ 현재가가 5년 장기 추세 대비 얼마나 비싸거나 싼지를 나타내는 온도계입니다.

```python
log(QQQ) = a + b x t   (t = 주의 순번)
Growth = exp(a + b x t_오늘)  # 추세선의 오늘 값
Eval   = (QQQ / Growth) - 1  # 추세 대비 괴리율
```

**Eval 해석 (현재 기본 임계값 +-6%):**
| Eval 값 | 의미 | 티어 |
|---------|------|------|
| +6% 이상 | QQQ가 추세보다 6% 이상 고평가 | HIGH |
| 0% 근처 | 추세에 부합하는 적정 가격 | MID |
| -6% 이하 | QQQ가 추세보다 6% 이상 저평가 | LOW |
""")
    with col_r:
        tc = {'HIGH': '#fbbf24', 'MID': '#60a5fa', 'LOW': '#4ade80'}.get(latest_tier, '#60a5fa')
        tb = {'HIGH': 'rgba(251,191,36,0.15)',
               'MID':  'rgba(96,165,250,0.15)',
               'LOW':  'rgba(74,222,128,0.15)'}.get(latest_tier, 'rgba(96,165,250,0.15)')
        label = {'HIGH': '과열 구간', 'MID': '중립 구간', 'LOW': '저평가 구간'}.get(latest_tier, '')
        st.markdown(f"""
<div style="background:{tb}; border:2px solid {tc}; border-radius:14px;
            padding:24px; text-align:center; margin-top:12px;">
  <p style="color:#94a3b8; margin:0; font-size:14px;">현재 시장 상태</p>
  <h1 style="color:{tc}; margin:8px 0; font-size:3rem;">{latest_tier}</h1>
  <h2 style="color:white; margin:0;">Eval = {latest_eval:.2%}</h2>
  <p style="color:#94a3b8; margin-top:10px;">QQQ가 5년 추세 대비<br>
     <b style="color:{tc};">{abs(latest_eval):.1%} {label}</b></p>
</div>
""", unsafe_allow_html=True)
        ev_recent = df.dropna(subset=['Eval']).tail(52)
        if not ev_recent.empty:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(
                x=ev_recent['Date'], y=ev_recent['Eval'] * 100,
                fill='tozeroy',
                line=dict(color='#60a5fa', width=2),
                fillcolor='rgba(96,165,250,0.2)',
                name='Eval'
            ))
            fig_e.add_hline(y=_hc_d,  line_color='#fbbf24', line_dash='dash', line_width=1,
                             annotation_text="HIGH", annotation_font=dict(color='#fbbf24', size=9))
            fig_e.add_hline(y=_lc_d,  line_color='#4ade80', line_dash='dash', line_width=1,
                             annotation_text="LOW",  annotation_font=dict(color='#4ade80', size=9))
            fig_e.add_hline(y=0,      line_color='#475569', line_dash='dot',  line_width=1)
            fig_e.update_layout(
                title='최근 1년 Eval 추이',
                yaxis_title='Eval (%)',
                height=230, showlegend=False, **CHART_LAYOUT
            )
            apply_grid(fig_e)
            st.plotly_chart(fig_e, use_container_width=True)
    st.divider()
    st.markdown("### Step 2. 3-티어 시장 분류")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"""
<div class="tier-high">
<h4>HIGH 티어 — 과열 구간</h4>
<b>조건:</b> Eval &gt;= +{_hc_d:.1f}%<br><br>
시장이 추세보다 뜨거운 상태.<br>
주가 상승 시 차익을 적극 실현해<br>
현금을 축적합니다.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right"><b>{_sH_d}x</b></td></tr>
  <tr><td>매수 배율</td><td align="right">{_bH_d}x</td></tr>
</table>
-> <b>팔아서 현금 쌓기</b>
</div>
""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
<div class="tier-mid">
<h4>MID 티어 — 중립 구간</h4>
<b>조건:</b> {_lc_d:.1f}% &lt; Eval &lt; +{_hc_d:.1f}%<br><br>
시장이 추세 근처에서 움직이는<br>
평상시 상태.<br>
균형 잡힌 비율로<br>기계적 리밸런싱.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right"><b>{_sM_d}x</b></td></tr>
  <tr><td>매수 배율</td><td align="right"><b>{_bM_d}x</b></td></tr>
</table>
-> <b>균형 유지</b>
</div>
""", unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
<div class="tier-low">
<h4>LOW 티어 — 저평가 구간</h4>
<b>조건:</b> Eval &lt;= {_lc_d:.1f}%<br><br>
시장이 추세보다 차가운 상태.<br>
역사적으로 가장 강한<br>
매수 기회 구간.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right">{_sL_d}x</td></tr>
  <tr><td>매수 배율</td><td align="right"><b>{_bL_d}x</b></td></tr>
</table>
-> <b>공격적 매수 — 핵심!</b>
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Step 3. 매매 실행 로직")
    st.markdown("**매주 금요일 장 마감 10분 전(오후 3:50) LOC 주문으로 실행합니다.**")
    st.code("""
매주 금요일 장 마감 시:
이번 주 TQQQ가 전주 대비 상승했다면:
    평가 수익   = 보유수량 x (이번 주 가격 - 전주 가격)
    매도 수량   = int(평가 수익 x 매도 배율[티어] / 현재가)
    실제 매도   = min(매도 수량, 현재 보유수량)   # 공매도 방지
    -> LOC 매도 주문 실행 -> 현금 증가
이번 주 TQQQ가 전주 대비 하락했다면:
    평가 손실   = 보유수량 x (전주 가격 - 이번 주 가격)
    매수 예산   = min(보유 현금, 평가 손실 x 매수 배율[티어])
    매수 수량   = int(매수 예산 / 현재가)
    -> LOC 매수 주문 실행 -> 주식 증가
변동 없으면:
    -> 관망 (주문 없음)
""", language='python')
    col_up, col_dn = st.columns(2)
    with col_up:
        st.info(f"""
**상승 시 매도 — 차익 실현**
평가 수익의 일정 배율만큼 매도해 현금을 확보합니다.
- **HIGH 티어** -> {_sH_d}x : 수익의 {_sH_d*100:.0f}% 어치 매도 (적극 실현)
- **MID 티어** -> {_sM_d}x : 수익의 {_sM_d*100:.0f}%만 매도 (소폭 실현)
- **LOW 티어** -> {_sL_d}x : 수익의 {_sL_d*100:.0f}%만 매도 (최소 실현)
-> 과열 구간에서 더 많이 팔아 현금을 쌓아둡니다.
-> 이 현금이 LOW 티어 발생 시 매수 탄약이 됩니다.
""")
    with col_dn:
        st.success(f"""
**하락 시 매수 — 저점 매집**
평가 손실의 배율만큼 현금을 투입해 추가 매수합니다.
- **HIGH 티어** -> {_bH_d}x : 손실의 {_bH_d*100:.0f}% 예산 매수
- **MID 티어** -> {_bM_d}x : 손실의 {_bM_d*100:.0f}% 예산 매수
- **LOW 티어** -> {_bL_d}x : 손실의 {_bL_d*100:.0f}% 예산 매수 (공격)
-> 시장이 차가울수록 더 많이 사서 평균 단가를 낮춥니다.
-> 현금이 부족하면 LOW 티어 기회를 살릴 수 없으므로
  초기 현금 비중 45% 유지를 권장합니다.
""")
    st.divider()
    st.markdown("### Step 3-b. 200일 추세 게이트 (v5.2 · 선택)")
    st.markdown("""
원본 위대리는 **절대 추세 인식이 없어** 레짐 전환(닷컴/2008/2022)과 건강한 눌림목을
구분하지 못합니다. LOW 티어 물타기(매수 2.0x)가 현금을 소진하면, 진짜 붕괴에서
나체 3x 매수후보유로 전락해 합성 기준 MDD -71~-97%가 납니다.

**200일 추세 게이트**는 이 꼬리를 자릅니다.
- 기초지수(QQQ)가 **200일 이동평균 아래로 이탈하면 전량 현금화**, 신규 매수 중단.
- QQQ가 200일선을 **회복하면 재진입**(목표 주식비중만큼 재투입).

**꼭 알아야 할 사실 (백테스트 검증 결과):**
- 게이트는 위기 MDD를 크게 줄이지만(-71~-97% → -17~-46%), **강세장 CAGR을 반납**합니다
  (실측 2010–26 Calmar 원본 ~1.3 → 게이트 ~0.66). 즉 **위험조정수익 우월 전략이 아니라
  "재난 보험"** 입니다.
- 밴드/확인지연으로 반납분을 되찾으려는 정교화는 **OOS(표본 밖)에서 기각**됐습니다
  (2020-03 V자 반등에서 재진입 실패로 오히려 손실). 그래서 순수 게이트만 채택했습니다.
- **채택은 데이터가 강제하는 게 아니라 리스크 선호 결정입니다.** "닷컴/2008급은 다시 안 온다
  (뉴노멀)"고 보면 OFF, "2022급(금리발 -80%)은 시스템위기 아니어도 온다"고 보면 ON.
  실제로 게이트는 실측 2022에서 TQQQ MDD를 -80% → -15% 로 잘랐습니다.

> Tab2 백테스트에서 **게이트 ON/OFF 동시 비교** + **2010년 이전 합성 데이터**를 켜서
> 닷컴·2008 구간의 방어 효과를 직접 확인하세요.
""")
    st.divider()
    st.markdown("### Step 4. 거래비용 & 양도세 모델")
    st.markdown("""
백테스트는 *명목 수익률* 외에 **거래 수수료** 와 **한국 거주자 양도세** 를 옵션으로 적용해
현실적인 net 수익률을 추정할 수 있습니다.

**거래 수수료**
- 매수: 0.015% (증권사 수수료)
- 매도: 0.015% + 0.00206% SEC fee = **0.01706%**
- 위대리는 주 1회 매매라 누적 영향은 작음 (~CAGR -0.05%p 수준)

**양도세 (한국 거주자)**
- 매년 12/31 기준 실현이익 산정 -> 250만원 (약 $1,923) 공제 -> 초과분에 22% 부과
- 가중평균 cost basis 방식 (한국 세법 기본 계산)
- 현금 부족 시 자동으로 일부 주식 매도해서 충당

**양도세 인출 전략 3종**
| 코드 | 패턴 | 특징 |
|---|---|---|
| **A** | 1월/5월 50/50 분할 (기본) | 후반 50%가 그 사이 운용되어 복리 up |
| B | 5월 일괄 | 1~4월 동안 자금이 묶이지 않음 |
| C | 1월 일괄 | 가장 빠른 정리, 깔끔한 회계 |

-> 백테스트에서 3종 동시 비교로 어느 전략이 최종 자산이 더 큰지 확인 가능.

**매도-only 리밸런싱 (옵션 B)**
- 양도세 차감 후 cash 비중이 *목표보다 낮으면* 일부 stock 매도해서 cash 보충
- cash 가 *과다해도 강제 매수는 안 함* (시장 타이밍 회피)
- 강제 매수 버전은 양도세 직후 (보통 1월 강세장 끝) stock 을 비싸게 사서 그 다음 약세장에서 큰 손실로 이어졌음 -> 백테스트로 검증된 후 매도-only 가 기본값
""")
    st.divider()
    st.markdown("### Step 5. 실전 운용 체크리스트")
    tip1, tip2, tip3 = st.columns(3)
    with tip1:
        st.warning("""
**주문 타이밍 (오토봇)**

봇 정상 실행 윈도우 (ET 기준):
- 금요일 04:00-09:30 (pre-market) — 어제 종가 기준
- 금요일 15:00-16:05 (near-close) — *KST 토 04:00-05:05*
- 금요일 16:05~ / 토 06:00 까지 (post-close)

-> KST 토 04:42 자동매매 패턴은 **near-close 모드** 사용
""")
    with tip2:
        st.info("""
**자본 배분 원칙**
- 초기 현금 비중 **45%** 권장 (백테스트 검증값)
- 현금 = LOW 티어 발생 시 매수 탄약
- LOW 티어 때 현금 소진 -> 기회 상실
- 양도세 적용 시 매도-only 리밸런싱 권장
""")
    with tip3:
        st.success("""
**핵심 마인드셋**
- Eval 낮을수록 = 기회, 두려워하지 않기
- 하락 시 매수 배율 낮추면 전략 붕괴
- 파라미터는 워크포워드로 검증된 디폴트 유지
- 양도세 / 수수료 적용 후 net CAGR 로 평가
""")
    st.divider()
    st.markdown("""
### 위대리 vs TQQQ 단순 보유(B&H) 핵심 비교
| 항목 | 위대리 (최적화 기본값) | TQQQ B&H |
|------|:---:|:---:|
| **연평균 수익 (Gross CAGR)** | **~31%** | ~45% |
| **평균 OOS MDD** | **~25%** | ~48% |
| **최악 OOS MDD** | -71% | -80% |
| **Calmar (수익/위험)** | **1.71** | ~0.93 |
| **심리적 안정성** | 상 | 하 |
| **주간 주문 횟수** | 1회 | 없음 |

> **TQQQ를 그냥 들고 있으면 -80% 이상의 하락을 버텨야 합니다.**
> 위대리는 약간 적은 수익으로 최대 낙폭을 **절반 수준**으로 줄여줍니다.
> 낙폭이 작으면 복리 효과도 극대화됩니다. (MDD -80% 회복엔 +400% 상승이 필요!)
> *위 수치는 walk-forward 15개 OOS 폴드 평균. in-sample 백테스트와 다를 수 있음.*

### Step 6. v4.8 OOS-검증 개선 (Task A + Task B)

v4.8 은 양도세 부담을 완화하고 강세장에서 winner를 더 태우기 위한
**두 가지 추가 신호**를 도입했습니다. 모두 **워크포워드 7-fold OOS 검증**을 통과한
robust 한 개선입니다.

**Task A — 12월 매도 축소 (dec_sell_scale)**

12월 weekly 매도 시그널의 sell rate에 0.75를 곱해 매도 강도를 25% 줄입니다.

- 효과: 12월에 실현되었을 이익이 포지션 안에 남아 다음 해로 이연 -> 양도세 22% 부담 감소
- 부수 효과: 1~5월 동안 그 자금이 계속 운용되어 복리 효과 up

**Task B — HIGH-tier 모멘텀 필터 (ma_window / trend_threshold / sell_rate_multiplier)**

시장이 HIGH 티어(Eval >= +6%) 이고 *동시에* QQQ가 13주 이동평균보다 8% 이상 위에 있을 때
(= 진짜 멜트업) HIGH 매도 배율을 0.7배로 축소해 winner를 더 오래 보유합니다.

- 일상적 HIGH 주(약한 과열)는 그대로 매도 -> 보호 매도 유지
- 진짜 멜트업(예: 2024 말~2025 초)에서만 발동 -> MDD 보존

**OOS 검증 결과**

| 항목 | IS (2017-2026) | OOS (워크포워드) |
|---|---|---|
| 베이스라인 (v4.7) | 30.51% | 25.56% |
| **v4.8 (Task A+B)** | **31.29%** | **26.31% (+0.75%p)** |

- IS 와 OOS 가산이 거의 일치(+0.78 vs +0.75) -> robust 한 alpha
- MDD 는 거의 변동 없음 (-30.45% vs -30.58%)

### Step 7. v5.1 — 앱/봇 엔진 동기화

앱과 오토봇(wedaeri_bot.py)의 주문 수량이 달랐던 원인 3가지를 수정해
두 엔진이 **완전히 동일한 주문**을 내도록 통일했습니다 (봇 방식이 기준).

1. **금요일 휴장 주 누락 (주범)** — 기존 앱은 Eval 을 일별 데이터에 금요일
   날짜로 머지한 뒤 리샘플해서, 금요일이 휴장인 주(성금요일 · 준틴스 ·
   독립기념일 등)의 Eval 이 NaN 이 되어 그 주 전체가 조용히 삭제됐습니다.
   이후 모든 주의 전주 가격 참조가 어긋나 매매 체인이 갈라졌습니다.
   -> 이제 주간 QQQ 에서 OLS/Eval 을 직접 재계산 (봇과 동일).
2. **QQQ 13주 MA 워밍업** — 기존 앱은 시작일 이후 데이터로만 MA 를 계산해
   첫 13주의 멜트업 필터 판정이 봇과 달랐습니다. -> 전체 히스토리 기준.
3. **초기 현금** — 기존 앱은 현금을 정확히 비중대로 고정하고 정수 주수
   잔액을 버렸습니다. -> 원금 - 주식매수액 (봇과 동일, 잔액 보존).

### Step 8. v5.2 — 200일 게이트 + 프리-2010 합성

- **200일 추세 게이트** (사이드바·Tab2 토글): QQQ 200일선 이탈 시 전량 현금화 →
  회복 시 재진입. 레짐 붕괴 꼬리 절단용 선택적 재난보험. 강세장 CAGR은 반납.
- **2010년 이전 합성 데이터** (Tab2 토글): TQQQ 상장(2010) 이전을 QQQ 3배 일간리밸런스
  합성으로 채워 닷컴(2000-02)·2008까지 백테스트. 변동성 잠식은 정확, 금융비용은 근사 —
  **절대수익보다 게이트 유무 상대비교** 용도. 합성 구간은 화면에 경고 배지로 표시됩니다.

### v4.9 변경 — 분석용 로그 강화
- 백테스트 매매 로그가 **매주 1행(관망 포함) 완전 패널**로 바뀌었습니다.
- 모든 값이 **raw 숫자(무포맷)** 라 CSV 다운로드만으로 전략 재현/검증이 가능합니다.
""")
