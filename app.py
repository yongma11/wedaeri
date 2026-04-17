# wedaeri_app.py — TQQQ 위대리 v4.0
# Tab1: 실전 트레이딩 | Tab2: 백테스트 분석 | Tab3: 전략 로직

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 설정 파일 (앱과 같은 폴더에 wedaeri_config.json 저장)
# ─────────────────────────────────────────────────────────────
CONFIG_FILE = Path(__file__).parent / "wedaeri_config.json"

# 앱 최초 실행 시 사용할 기본값
DEFAULT_CONFIG = {
    # 사이드바 (실전)
    'start_date': '2025-12-26',
    'cap':        108000,
    'cash':       40,       # 정수 %
    # 백테스트 전용
    'bt_cap':     20000,
    'bt_cash':    40,       # 정수 %
    'bt_start':   '2010-01-01',  # 백테스트 시작일
    # 전략 파라미터 (Tab1·Tab2 공유)
    'hc':  7.0,   'lc': -7.0,
    'sH':  1.5,   'sM':  0.6,   'sL': 0.35,
    'bH':  0.4,   'bM':  0.6,   'bL': 2.0,
}

def load_config() -> dict:
    """저장된 JSON 설정 파일 로드 (없으면 기본값 반환)"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding='utf-8') as f:
                saved = json.load(f)
            return {**DEFAULT_CONFIG, **saved}   # 새 키는 기본값으로 보완
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(ss) -> None:
    """현재 session_state 값을 JSON 파일에 저장"""
    # p_start 는 date 객체일 수도 있고 문자열일 수도 있음 → 항상 문자열로 변환
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
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────
# 1. 페이지 설정 & 스타일
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="위대리 Quantum T-Flow v4.0", layout="wide")

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
# session_state 초기화 (앱 최초 실행 시 JSON → session_state)
# ─────────────────────────────────────────────────────────────
if '_cfg_loaded' not in st.session_state:
    _cfg = load_config()
    ss   = st.session_state
    ss['_cfg_loaded'] = True
    # date_input 은 date 객체를 요구 — 문자열로 저장된 값을 변환
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
    st.header("⚙️ 위대리 기본 설정")
    with st.container(border=True):
        # key= 사용 → session_state['p_*'] 와 자동 연결
        # value= 는 key 가 이미 session_state 에 있으면 무시되므로 생략
        st_start      = st.date_input("투자 시작일", key='p_start')
        st_cap        = st.number_input("시작 원금 ($)", step=1000.0, key='p_cap')
        st_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, key='p_cash') / 100

        col_ref, col_save = st.columns(2)
        with col_ref:
            if st.button("🔄 데이터 갱신", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_save:
            if st.button("💾 설정 저장", use_container_width=True):
                save_config(st.session_state)
                st.toast("✅ 설정이 저장되었습니다!", icon="💾")

    st.divider()
    # 현재 적용 중인 파라미터를 session_state에서 동적으로 표시
    _p = st.session_state
    _hc_d = _p.get('p_hc', 7.0); _lc_d = _p.get('p_lc', -7.0)
    _sH_d = _p.get('p_sH', 1.5); _sM_d = _p.get('p_sM', 0.6); _sL_d = _p.get('p_sL', 0.35)
    _bH_d = _p.get('p_bH', 0.4); _bM_d = _p.get('p_bM', 0.6); _bL_d = _p.get('p_bL', 2.0)
    st.markdown(f"""
**📌 현재 적용 파라미터**

| 티어 | Eval | 매도× | 매수× |
|------|------|-------|-------|
| HIGH | ≥ +{_hc_d:.1f}% | {_sH_d} | {_bH_d} |
| MID  | 중간 | {_sM_d} | {_bM_d} |
| LOW  | ≤ {_lc_d:.1f}% | {_sL_d} | {_bL_d} |

*(백테스트 탭에서 변경·저장 가능)*
""")

# ─────────────────────────────────────────────────────────────
# 3. 데이터 로딩 — 주간 Expanding Window OLS (스프레드시트 방식)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_wedaeri_data():
    """QQQ + TQQQ 로딩 & 주간 Expanding OLS로 Growth/Eval 계산

    핵심: 스프레드시트처럼 2010년부터 주간 데이터 기준 확장 창(최대 260주=5년)
    으로 OLS를 계산합니다. 2000년 데이터로 1260일 창을 쓰면 2010년 초
    Growth가 닷컴·금융위기 영향을 받아 Eval이 완전히 달라집니다.
    """
    try:
        raw = yf.download(["QQQ", "TQQQ"], start="2010-01-01",
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

        # ── 주간 QQQ — W-FRI 리샘플로 주의 마지막 거래일 기준 ─────
        # weekday==4 고정 필터 대신 resample('W-FRI').last() 사용:
        # 금요일이 휴장(공휴일)인 주는 목요일 등 실제 마지막 거래일을 사용.
        # 주 순번 t = 1,2,3,... 사용 (ordinal 날짜보다 수치 안정)
        # 창: 첫 260주는 expanding, 이후 fixed 260주
        qqq_wkly = (df.set_index('Date')[['QQQ']]
                    .resample('W-FRI').last()
                    .dropna()
                    .reset_index())
        n = len(qqq_wkly)
        W = 260   # 5년 = 260주

        t = np.arange(1, n + 1, dtype=float)
        y = np.log(qqq_wkly['QQQ'].values.astype(float))

        # prefix sums (인덱스 0 = 0, 인덱스 i = 누적합 1..i)
        ps_t  = np.zeros(n + 1); ps_t[1:]  = np.cumsum(t)
        ps_y  = np.zeros(n + 1); ps_y[1:]  = np.cumsum(y)
        ps_t2 = np.zeros(n + 1); ps_t2[1:] = np.cumsum(t ** 2)
        ps_ty = np.zeros(n + 1); ps_ty[1:] = np.cumsum(t * y)

        growth_wkly = np.empty(n)
        for i in range(n):
            w     = min(i + 1, W)
            end   = i + 1        # prefix-sum 끝 인덱스 (포함)
            start = end - w      # prefix-sum 시작 인덱스 (불포함)

            if w == 1:           # 데이터 1개: 추세 = 가격 자체
                growth_wkly[i] = float(qqq_wkly['QQQ'].iloc[i])
                continue

            s_t  = ps_t[end]  - ps_t[start]
            s_y  = ps_y[end]  - ps_y[start]
            s_t2 = ps_t2[end] - ps_t2[start]
            s_ty = ps_ty[end] - ps_ty[start]
            denom = w * s_t2 - s_t ** 2
            if denom == 0:
                growth_wkly[i] = float(qqq_wkly['QQQ'].iloc[i])
                continue
            b = (w * s_ty - s_t * s_y) / denom
            a = (s_y - b * s_t) / w
            growth_wkly[i] = np.exp(a + b * t[i])

        qqq_wkly['Growth'] = growth_wkly
        qqq_wkly['Eval']   = qqq_wkly['QQQ'] / qqq_wkly['Growth'] - 1

        # 일별 데이터에 주간 Growth/Eval 병합 (금요일 날짜 기준)
        df = df.merge(qqq_wkly[['Date', 'Growth', 'Eval']], on='Date', how='left')
        return df

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 3-b. 실시간 종가 주입 (캐시 밖 — 장 마감 후 즉시 반영)
# ─────────────────────────────────────────────────────────────
def inject_live_price(df: pd.DataFrame) -> pd.DataFrame:
    """장 마감 후 최신 종가를 df에 주입합니다.
    - 오늘(또는 가장 최근 거래일)의 TQQQ/QQQ 실시간 가격을 df 마지막 행에 반영
    - Eval이 없으면 직전 주의 Eval을 재사용 (OLS는 주 단위로 거의 안 변함)
    - 토요일 아침처럼 캐시가 금요일 장 마감 전 데이터인 경우를 커버
    """
    try:
        live_tqqq = float(yf.Ticker("TQQQ").fast_info['last_price'])
        live_qqq  = float(yf.Ticker("QQQ").fast_info['last_price'])

        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        # 주말이면 직전 금요일을 타겟 날짜로 사용
        wd = today.weekday()
        if wd == 5:    # 토요일
            target = today - pd.Timedelta(days=1)
        elif wd == 6:  # 일요일
            target = today - pd.Timedelta(days=2)
        else:
            target = today

        last_df_date = df['Date'].iloc[-1]

        if target > last_df_date:
            # 타겟 날짜가 df에 없음 → 새 행 추가 (Eval은 직전 값 재사용)
            last_eval   = float(df['Eval'].dropna().iloc[-1])
            last_growth = float(df['Growth'].dropna().iloc[-1])
            new_row = pd.DataFrame({
                'Date': [target], 'TQQQ': [live_tqqq], 'QQQ': [live_qqq],
                'Eval': [last_eval], 'Growth': [last_growth]
            })
            df = pd.concat([df, new_row], ignore_index=True).reset_index(drop=True)
        else:
            # 마지막 행 가격만 업데이트 (Eval은 이미 계산되어 있음)
            df.loc[df.index[-1], 'TQQQ'] = live_tqqq
            df.loc[df.index[-1], 'QQQ']  = live_qqq
            # Eval이 NaN이면 직전 값으로 채움
            if pd.isna(df.loc[df.index[-1], 'Eval']):
                df.loc[df.index[-1], 'Eval']   = float(df['Eval'].dropna().iloc[-2])
                df.loc[df.index[-1], 'Growth'] = float(df['Growth'].dropna().iloc[-2])
    except Exception:
        pass
    return df


# ─────────────────────────────────────────────────────────────
# 4. 실전 시뮬레이션 (Tab 1용 — 지정 시작일부터)
# ─────────────────────────────────────────────────────────────
def run_wedaeri_sim(data, start_dt, init_cap, cash_ratio,
                    hc=0.07, lc=-0.07,
                    sH=1.5, sM=0.6, sL=0.35,
                    bH=0.4, bM=0.6, bL=2.0):
    sim  = data[data['Date'] >= pd.to_datetime(start_dt)].copy()
    # resample('W-FRI').last() : 금요일 휴장 주는 그 주 마지막 거래일 자동 사용
    wkly = (sim.set_index('Date')[['TQQQ', 'Eval']]
               .resample('W-FRI').last()
               .dropna()
               .reset_index())
    if wkly.empty or len(wkly) < 2:
        return pd.DataFrame()

    cash   = init_cap * cash_ratio
    shares = int((init_cap * (1 - cash_ratio)) / wkly['TQQQ'].iloc[0])
    logs   = []

    for i in range(len(wkly)):
        p    = float(wkly.loc[i, 'TQQQ'])
        ev   = float(wkly.loc[i, 'Eval'])
        tier = 'HIGH' if ev >= hc else ('LOW' if ev <= lc else 'MID')
        sr   = {'HIGH': sH, 'MID': sM, 'LOW': sL}[tier]
        br   = {'HIGH': bH, 'MID': bM, 'LOW': bL}[tier]

        action, disp_qty = "관망", 0
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
            '액션':     action,
            '주문수량': disp_qty,
            '보유수량': shares,
            '현금':     round(cash, 2),
            '총자산':   round(total, 2),
            '수익률':   f"{(total/init_cap - 1)*100:.2f}%",
        })
    return pd.DataFrame(logs)


# ─────────────────────────────────────────────────────────────
# 5. 전체 기간 백테스트 (Tab 2용)
# ─────────────────────────────────────────────────────────────
def run_full_backtest(data, init_cap=20_000, cash_ratio=0.40,
                      hc=0.07, lc=-0.07,
                      sH=1.5,  sM=0.6,  sL=0.35,
                      bH=0.4,  bM=0.6,  bL=2.0,
                      start_date=None):
    # resample('W-FRI').last() : 금요일 휴장 주는 그 주 마지막 거래일 자동 사용
    wkly = (data.set_index('Date')[['TQQQ', 'Eval']]
                .resample('W-FRI').last()
                .dropna()
                .reset_index())
    # 백테스트 시작일 필터
    if start_date is not None:
        wkly = wkly[wkly['Date'] >= pd.to_datetime(start_date)].reset_index(drop=True)
    if len(wkly) < 2:   # 최소 2주 필요 (첫 주 초기화 + 1주 매매)
        return None

    P     = wkly['TQQQ'].values.astype(float)
    EV    = wkly['Eval'].values.astype(float)
    dates = wkly['Date'].values
    N     = len(wkly)
    # 실제 날짜 기반 연수 계산 (시작일 변경 시 정확성 향상)
    span_days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
    YEARS = max(span_days / 365.25, N / 52)

    cash   = init_cap * cash_ratio
    shares = int((init_cap * (1 - cash_ratio)) / P[0])
    eq     = np.empty(N)
    tiers  = []
    eq[0]  = cash + shares * P[0]

    for i in range(N):
        p  = P[i]
        ev = EV[i]
        tier = 'HIGH' if ev >= hc else ('LOW' if ev <= lc else 'MID')
        tiers.append(tier)
        if i == 0:
            continue
        diff = shares * (p - P[i-1])
        if diff > 0:
            qty = int(min(round(diff * {'HIGH': sH, 'MID': sM, 'LOW': sL}[tier] / p), shares))
            shares -= qty; cash += qty * p
        elif diff < 0:
            qty = int(min(cash, abs(diff) * {'HIGH': bH, 'MID': bM, 'LOW': bL}[tier]) / p)
            shares += qty; cash -= qty * p
        eq[i] = cash + shares * p

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

    # ── 연도별 성과 계산 ──────────────────────────────────────
    yearly_df = pd.DataFrame({'date': pd.to_datetime(dates), 'eq': eq})
    yearly_df['year'] = yearly_df['date'].dt.year
    yearly_rows = []
    prev_end = init_cap
    for yr, grp in yearly_df.groupby('year'):
        start_eq = prev_end
        end_eq   = float(grp['eq'].iloc[-1])
        ret_pct  = (end_eq / start_eq - 1) * 100 if start_eq > 0 else 0
        # 연 내 peak-to-trough MDD
        yr_peak = np.maximum.accumulate(grp['eq'].values)
        yr_mdd  = float((grp['eq'].values / yr_peak - 1).min()) * 100
        yearly_rows.append({
            '연도': int(yr),
            '수익률': f"{ret_pct:+.1f}%",
            '연간 MDD': f"{yr_mdd:.1f}%",
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
    }


# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
with st.spinner("📡 데이터 로딩 중 (최초 1회만 시간이 걸립니다)..."):
    df = load_wedaeri_data()

# 장 마감 후 최신 종가 주입 — 캐시와 무관하게 항상 실행
# 토요일 아침: 금요일 종가가 캐시에 없어도 즉시 반영
df = inject_live_price(df)

if df.empty:
    st.error("데이터 로드 실패. 잠시 후 새로고침 해주세요.")
    st.stop()

# 전략 파라미터는 session_state p_* 에서 읽어옴 (Tab2 프리셋 변경 즉시 반영)
_ss = st.session_state
log_df = run_wedaeri_sim(
    df, st_start, st_cap, st_cash_ratio,
    hc  = _ss.get('p_hc', 7.0)  / 100,
    lc  = _ss.get('p_lc', -7.0) / 100,
    sH  = _ss.get('p_sH', 1.5),
    sM  = _ss.get('p_sM', 0.6),
    sL  = _ss.get('p_sL', 0.35),
    bH  = _ss.get('p_bH', 0.4),
    bM  = _ss.get('p_bM', 0.6),
    bL  = _ss.get('p_bL', 2.0),
)

tqqq_series = df['TQQQ'].dropna()
latest_tqqq = float(tqqq_series.iloc[-1]) if not tqqq_series.empty else 0.0
eval_series  = df['Eval'].dropna()
latest_eval  = float(eval_series.iloc[-1]) if not eval_series.empty else 0.0
_hc_rt = _ss.get('p_hc', 7.0) / 100
_lc_rt = _ss.get('p_lc', -7.0) / 100
latest_tier  = 'HIGH' if latest_eval >= _hc_rt else ('LOW' if latest_eval <= _lc_rt else 'MID')

st.title("🚀 TQQQ [위대리] v4.0 : 균형형 트레이딩 시스템")
tab1, tab2, tab3 = st.tabs(["🔥 실전 트레이딩", "📊 백테스트 분석", "📘 전략 로직"])


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
    c4.metric("매매 주기",    "금요일 (주간 LOC)")

    st.subheader(f"📝 금주 장 마감(LOC) 주문표 ({datetime.now().strftime('%Y-%m-%d')})")
    b_col, s_col = st.columns(2)
    with b_col:
        val = f"수량: {last['주문수량']} 주" if last['액션'] == "매수" else "대기 (신호 없음)"
        st.markdown(
            f'<div class="order-card-buy"><h4>📉 LOC 매수 주문</h4>'
            f'<h1 style="color:#188038;">{val}</h1></div>',
            unsafe_allow_html=True)
    with s_col:
        val = f"수량: {abs(last['주문수량'])} 주" if last['액션'] == "매도" else "대기 (신호 없음)"
        st.markdown(
            f'<div class="order-card-sell"><h4>📈 LOC 매도 주문</h4>'
            f'<h1>{val}</h1></div>',
            unsafe_allow_html=True)

    st.divider()
    st.subheader("💰 내 계좌 현황")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("보유 수량",  f"{last['보유수량']:,} 주")
    a2.metric("보유 현금",  f"${last['현금']:,.2f}")
    a3.metric("수익률",     last['수익률'])
    a4.metric("평가 금액",  f"${last['보유수량'] * latest_tqqq:,.2f}")
    a5.metric("현금 비중",  f"{last['현금'] / last['총자산'] * 100:.2f}%")

    with st.expander("📋 상세 매매 로그", expanded=True):
        st.dataframe(log_df.iloc[::-1], use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — 백테스트 분석
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 위대리 전략 전체 기간 백테스트")
    st.caption("TQQQ 상장 첫 주(2010년 2월)부터 현재까지의 전략 시뮬레이션")

    # ── 파라미터 패널 ─────────────────────────────────────────
    with st.expander("⚙️ 백테스트 파라미터 설정", expanded=True):

        preset = st.selectbox(
            "파라미터 프리셋 선택",
            ["📋 현재 파라미터  (CAGR ~42%, MDD ~32%)",
             "🏆 최적화 파라미터 (CAGR ~43%, MDD ~31%)",
             "🛡️ 안정형 파라미터 (CAGR ~36%, MDD ~24%)",
             "✏️ 직접 설정"],
            index=0
        )

        PRESETS = {
            # 스크린샷 기준 — hc 7%, lc -7%, init_cash 40%
            "현재":  dict(hc=0.07, lc=-0.07, sH=1.5, sM=0.6,  sL=0.35, bH=0.4, bM=0.6, bL=2.0),
            # 최적화 결과 — MDD 35% 이내 최대 CAGR (Grid Search)
            "최적화": dict(hc=0.06, lc=-0.06, sH=2.0, sM=0.3,  sL=0.2,  bH=1.0, bM=0.6, bL=2.0),
            # 안정형 — MDD 25% 이내 최적
            "안정형": dict(hc=0.06, lc=-0.10, sH=2.0, sM=0.6,  sL=0.33, bH=1.0, bM=1.0, bL=2.0),
        }

        key_map = {"📋 현재": "현재", "🏆 최적화": "최적화", "🛡️ 안정형": "안정형"}
        pkey    = next((v for k, v in key_map.items() if k in preset), None)

        # ── 프리셋 선택 시 공유 session_state(p_*) 강제 갱신 ──────
        # p_* 키는 Tab1 실전 트레이딩에도 그대로 적용됩니다.
        _ss = st.session_state
        if pkey and _ss.get('_last_preset') != pkey:
            _ss['_last_preset'] = pkey
            v = PRESETS[pkey]
            _ss['p_hc'] = v['hc'] * 100   # 슬라이더는 % 단위
            _ss['p_lc'] = v['lc'] * 100
            _ss['p_sH'] = v['sH']
            _ss['p_sM'] = v['sM']
            _ss['p_sL'] = v['sL']
            _ss['p_bH'] = v['bH']
            _ss['p_bM'] = v['bM']
            _ss['p_bL'] = v['bL']

        P_DEF = PRESETS[pkey] if pkey else {
            'hc': _ss.get('p_hc', 7.0) / 100,
            'lc': _ss.get('p_lc', -7.0) / 100,
            'sH': _ss.get('p_sH', 1.5), 'sM': _ss.get('p_sM', 0.6),
            'sL': _ss.get('p_sL', 0.35),
            'bH': _ss.get('p_bH', 0.4), 'bM': _ss.get('p_bM', 0.6),
            'bL': _ss.get('p_bL', 2.0),
        }

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**기본 설정**")
            bt_start_date = st.date_input("백테스트 시작일", key='p_bt_start')
            bt_cap        = st.number_input("초기 자본 ($)", value=20_000, step=1000, key='p_bt_cap')
            bt_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100,
                                      int(_ss.get('p_bt_cash', 40)), key='p_bt_cash') / 100
            st.markdown("**시장 평가 기준**")
            # p_hc/p_lc 는 % 단위로 저장 → 슬라이더가 그대로 표시
            bt_hc = st.slider("HIGH 기준 Eval ≥ (%)", 1.0, 20.0,
                               float(P_DEF['hc']*100), 0.5, key='p_hc') / 100
            bt_lc = st.slider("LOW 기준  Eval ≤ (%)", -20.0, -1.0,
                               float(P_DEF['lc']*100), 0.5, key='p_lc') / 100

        with p2:
            st.markdown("**매도 배율** (상승 시 차익실현 강도)")
            bt_sH = st.slider("매도 HIGH ×", 0.1, 5.0,  float(P_DEF['sH']), 0.05, key='p_sH')
            bt_sM = st.slider("매도 MID  ×", 0.1, 3.0,  float(P_DEF['sM']), 0.05, key='p_sM')
            bt_sL = st.slider("매도 LOW  ×", 0.05, 2.0, float(P_DEF['sL']), 0.05, key='p_sL')

        with p3:
            st.markdown("**매수 배율** (하락 시 추가 매수 강도)")
            bt_bH = st.slider("매수 HIGH ×", 0.1, 3.0,  float(P_DEF['bH']), 0.05, key='p_bH')
            bt_bM = st.slider("매수 MID  ×", 0.1, 3.0,  float(P_DEF['bM']), 0.05, key='p_bM')
            bt_bL = st.slider("매수 LOW  ×", 0.5, 10.0, float(P_DEF['bL']), 0.25, key='p_bL')

        # ── 안내 메시지 + 저장 버튼 ───────────────────────────────
        msg_col, btn_col = st.columns([3, 1])
        with msg_col:
            if pkey:
                vals = PRESETS[pkey]
                st.info(
                    f"💡 **{pkey} 프리셋** — "
                    f"hc={vals['hc']:.1%} / lc={vals['lc']:.1%} | "
                    f"매도 {vals['sH']}/{vals['sM']}/{vals['sL']} | "
                    f"매수 {vals['bH']}/{vals['bM']}/{vals['bL']}  "
                    f"→ 슬라이더·실전 탭 모두 자동 반영됩니다."
                )
            else:
                st.info("💡 슬라이더를 조정하면 **실전 트레이딩 탭**에도 즉시 반영됩니다.")
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 파라미터 저장", use_container_width=True):
                save_config(st.session_state)
                st.toast("✅ 파라미터가 저장되었습니다! 다음 실행 시에도 유지됩니다.", icon="💾")

    # ── 백테스트 실행 ─────────────────────────────────────────
    with st.spinner("🔄 백테스트 계산 중..."):
        bt_cur = run_full_backtest(
            df, bt_cap, bt_cash_ratio,
            hc=bt_hc, lc=bt_lc,
            sH=bt_sH, sM=bt_sM, sL=bt_sL,
            bH=bt_bH, bM=bt_bM, bL=bt_bL,
            start_date=bt_start_date,
        )
        bt_opt = run_full_backtest(
            df, bt_cap, bt_cash_ratio,
            hc=0.06, lc=-0.06,
            sH=2.0, sM=0.3, sL=0.2,
            bH=1.0, bM=0.6, bL=2.0,
            start_date=bt_start_date,
        )

    if bt_cur is None:
        st.warning("백테스트 데이터가 부족합니다.")
        st.stop()

    dates = pd.to_datetime(bt_cur['dates'])

    # ── Eval 진단 (계산 검증용) ───────────────────────────────
    ev = bt_cur['eval']
    n_h = int((ev >= bt_hc).sum()); n_l = int((ev <= bt_lc).sum())
    n_m = len(ev) - n_h - n_l
    with st.expander("🔎 Eval(시장평가) 진단 — 티어 배분 확인", expanded=False):
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

    # ── 성과 지표 ─────────────────────────────────────────────
    st.markdown("### 📈 성과 지표")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CAGR",         f"{bt_cur['cagr']:.2%}",
              f"최적화 {bt_opt['cagr']:.2%}")
    m2.metric("MDD",          f"{bt_cur['mdd']:.2%}",
              f"최적화 {bt_opt['mdd']:.2%}", delta_color="inverse")
    m3.metric("Calmar 비율",  f"{bt_cur['cal']:.2f}",
              f"최적화 {bt_opt['cal']:.2f}")
    m4.metric("Sortino 비율", f"{bt_cur['sor']:.2f}",
              f"최적화 {bt_opt['sor']:.2f}")
    m5.metric("최종 자산",    f"${bt_cur['final']:,.0f}",
              f"초기 ${bt_cur['init']:,.0f}")

    st.caption(
        f"백테스트 기간: {dates[0].strftime('%Y.%m.%d')} ~ {dates[-1].strftime('%Y.%m.%d')}"
        f"  ({bt_cur['years']:.1f}년) | 초기 자본 ${bt_cap:,} | 초기 현금 {bt_cash_ratio:.0%}"
    )

    # ── 3-way 비교표 ──────────────────────────────────────────
    st.markdown("### 📋 3-way 성과 비교")
    compare = pd.DataFrame({
        '전략':      ['내 설정', '최적화 파라미터', 'TQQQ B&H'],
        'CAGR':     [f"{bt_cur['cagr']:.2%}",  f"{bt_opt['cagr']:.2%}",  f"{bt_cur['bh_cagr']:.2%}"],
        'MDD':      [f"{bt_cur['mdd']:.2%}",   f"{bt_opt['mdd']:.2%}",   f"{bt_cur['bh_mdd']:.2%}"],
        'Calmar':   [f"{bt_cur['cal']:.2f}",   f"{bt_opt['cal']:.2f}",   "—"],
        'Sortino':  [f"{bt_cur['sor']:.2f}",   f"{bt_opt['sor']:.2f}",   "—"],
        '최종 자산': [f"${bt_cur['final']:,.0f}", f"${bt_opt['final']:,.0f}",
                     f"${bt_cur['bh_eq'][-1]:,.0f}"],
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.divider()

    # ── 자산 곡선 ─────────────────────────────────────────────
    st.markdown("### 📈 누적 자산 곡선 (로그 스케일)")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=dates, y=bt_cur['eq'], name='내 설정',
        line=dict(color='#60a5fa', width=2.5)
    ))
    fig_eq.add_trace(go.Scatter(
        x=dates, y=bt_opt['eq'], name='최적화 파라미터',
        line=dict(color='#4ade80', width=2.5)
    ))
    fig_eq.add_trace(go.Scatter(
        x=dates, y=bt_cur['bh_eq'], name='TQQQ B&H',
        line=dict(color='#fb923c', width=1.5, dash='dash'), opacity=0.7
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

    # ── MDD 곡선 + Eval 분포 ──────────────────────────────────
    col_dd, col_ev = st.columns(2)

    with col_dd:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dates, y=bt_cur['dd'] * 100,
            fill='tozeroy', name='내 설정',
            line=dict(color='#f87171', width=1.5),
            fillcolor='rgba(248,113,113,0.2)'
        ))
        fig_dd.add_trace(go.Scatter(
            x=dates, y=bt_opt['dd'] * 100,
            name='최적화', line=dict(color='#4ade80', width=1.5)
        ))
        fig_dd.add_hline(
            y=bt_cur['mdd']*100, line_color='#f87171', line_dash='dash',
            annotation_text=f"MDD {bt_cur['mdd']:.1%}",
            annotation_font=dict(color='#f87171', size=10)
        )
        fig_dd.update_layout(
            title='낙폭(Drawdown) 비교',
            yaxis_title='낙폭 (%)', height=300, **CHART_LAYOUT
        )
        apply_grid(fig_dd)
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_ev:
        ev_vals = bt_cur['eval']
        n_high  = int((ev_vals >= bt_hc).sum())
        n_low   = int((ev_vals <= bt_lc).sum())
        n_mid   = len(ev_vals) - n_high - n_low

        fig_ev = go.Figure()
        fig_ev.add_trace(go.Histogram(
            x=ev_vals * 100, nbinsx=40,
            name='Eval', marker_color='#60a5fa', opacity=0.75
        ))
        fig_ev.add_vline(x=bt_hc*100, line_color='#fbbf24', line_dash='dash',
                          annotation_text=f"HIGH ≥{bt_hc:.1%}",
                          annotation_font=dict(color='#fbbf24', size=10))
        fig_ev.add_vline(x=bt_lc*100, line_color='#4ade80', line_dash='dash',
                          annotation_text=f"LOW ≤{bt_lc:.1%}",
                          annotation_font=dict(color='#4ade80', size=10))
        fig_ev.update_layout(
            title=(f"Eval 분포  |  "
                   f"HIGH {n_high/len(ev_vals):.1%} / "
                   f"MID {n_mid/len(ev_vals):.1%} / "
                   f"LOW {n_low/len(ev_vals):.1%}"),
            xaxis_title='QQQ 시장평가 (%)',
            height=300, showlegend=False, **CHART_LAYOUT
        )
        apply_grid(fig_ev)
        st.plotly_chart(fig_ev, use_container_width=True)

    # ── 주간 수익률 분포 ──────────────────────────────────────
    st.markdown("### 📊 주간 수익률 분포")

    wr = bt_cur['rets'] * 100
    if len(wr) < 2:
        st.info("⚠️ 주간 데이터가 부족합니다. 백테스트 시작일을 더 앞으로 설정해 주세요.")
    else:
        pos_pct = (wr > 0).mean() * 100

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=wr, nbinsx=60,
            marker_color=['#4ade80' if r >= 0 else '#f87171' for r in wr],
            name='주간 수익률'
        ))
        fig_ret.add_vline(x=0, line_color='white', line_dash='dash', line_width=1)
        fig_ret.add_vline(x=float(np.mean(wr)), line_color='#fbbf24', line_dash='dot',
                           annotation_text=f"평균 {np.mean(wr):.2f}%",
                           annotation_font=dict(color='#fbbf24', size=10))
        fig_ret.update_layout(
            title=f'주간 수익률 분포  |  양전 비율 {pos_pct:.1f}%  |  최대 낙주 {float(wr.min()):.1f}%',
            xaxis_title='주간 수익률 (%)',
            height=260, showlegend=False, **CHART_LAYOUT
        )
        apply_grid(fig_ret)
        st.plotly_chart(fig_ret, use_container_width=True)

    # ── 티어별 성과 ───────────────────────────────────────────
    st.markdown("### 🎯 티어별 성과 분석")

    tiers_arr = np.array(bt_cur['tiers'])
    wr_full   = np.concatenate([[0], wr]) if len(wr) > 0 else np.array([0.0])  # 첫 주 0 패딩

    tier_rows = []
    for name, icon in [('HIGH', '🟡'), ('MID', '🔵'), ('LOW', '🟢')]:
        mask = tiers_arr == name
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        t_rets = wr_full[mask]
        tier_rows.append({
            '티어':    f'{icon} {name}',
            '발생 주': f"{cnt}주 ({cnt/len(tiers_arr):.1%})",
            '평균 수익률': f"{t_rets.mean():+.2f}%",
            '최대 상승': f"{t_rets.max():+.2f}%",
            '최대 하락': f"{t_rets.min():+.2f}%",
            '양전 비율': f"{(t_rets > 0).mean():.1%}",
        })

    if tier_rows:
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

    # ── 연도별 상세 성과 ──────────────────────────────────────
    st.markdown("### 📅 연도별 상세 성과")
    if bt_cur.get('yearly'):
        yr_df = pd.DataFrame(bt_cur['yearly'])
        # 수익률에 따라 행 색상 (양수=초록, 음수=빨강)
        def color_ret(val):
            try:
                v = float(val.replace('%', '').replace('+', ''))
                color = '#22c55e' if v >= 0 else '#f87171'
            except Exception:
                color = ''
            return f'color: {color}; font-weight: bold'

        styled = yr_df.style.applymap(color_ret, subset=['수익률'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # 연도별 수익률 막대 차트
        yr_vals = [float(r['수익률'].replace('%', '').replace('+', '')) for r in bt_cur['yearly']]
        yr_labels = [str(r['연도']) for r in bt_cur['yearly']]
        colors = ['#4ade80' if v >= 0 else '#f87171' for v in yr_vals]

        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(
            x=yr_labels, y=yr_vals,
            marker_color=colors,
            name='연간 수익률',
            text=[f"{v:+.1f}%" for v in yr_vals],
            textposition='outside',
            textfont=dict(size=10, color='#cbd5e1'),
        ))
        fig_yr.add_hline(y=0, line_color='#475569', line_width=1)
        fig_yr.update_layout(
            title='연도별 수익률 (%)',
            yaxis_title='수익률 (%)',
            height=280, showlegend=False, **CHART_LAYOUT
        )
        apply_grid(fig_yr)
        st.plotly_chart(fig_yr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — 전략 로직
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📘 위대리 전략 완전 해설")
    st.caption("TQQQ Quantum T-Flow — QQQ 추세 기반 3-티어 자동 리밸런싱 시스템")

    # ── 전략 개요 ─────────────────────────────────────────────
    st.markdown("""
### 🎯 전략 개요

**위대리 전략**은 나스닥 100(QQQ)의 5년 장기 추세와 현재 가격의 괴리를 측정해,
시장이 **과열(HIGH)** 이면 수익을 실현하고, **침체(LOW)** 이면 공격적으로 매수하는
**역추세 + 추세추종 혼합 리밸런싱 전략**입니다.

> 💡 핵심 철학: *"시장 온도계(Eval)가 낮을수록 더 많이 사고, 높을수록 더 많이 판다."*
> 매주 금요일 LOC 주문 1회 — 감정 없이 기계적으로 실행합니다.
""")

    st.divider()

    # ── Step 1: Eval 계산 ─────────────────────────────────────
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown("""
### 📐 Step 1. QQQ 시장 평가(Eval) 계산

**Eval**은 QQQ 현재가가 5년 장기 추세 대비 얼마나 비싸거나 싼지를 나타내는 온도계입니다.

```python
# 5년(1260거래일) 롤링 log-선형 회귀
log(QQQ) = a + b × t   (t = 날짜의 순번)

Growth = exp(a + b × t_오늘)  ← 추세선의 오늘 값
Eval   = (QQQ / Growth) - 1  ← 추세 대비 괴리율
```

**왜 단순 이동평균이 아닌 log-선형 회귀인가?**

| 방식 | 계산 | 문제점 |
|------|------|--------|
| 단순 5년 이동평균 | 과거 가격 평균 | 상승장에서 항상 현재가 > 평균 → Eval 항상 양수(과열) |
| **log-선형 회귀** | 추세선을 오늘 시점으로 외삽 | ✅ 현실적인 추세 대비 위치 측정 |

**Eval 해석:**

| Eval 값 | 의미 | 티어 |
|---------|------|------|
| +10% 이상 | QQQ가 추세보다 10% 고평가 | HIGH |
| 0% 근처 | 추세에 부합하는 적정 가격 | MID |
| −10% 이하 | QQQ가 추세보다 10% 저평가 | LOW |
""")

    with col_r:
        # 현재 시장 상태 카드
        tc = {'HIGH': '#fbbf24', 'MID': '#60a5fa', 'LOW': '#4ade80'}.get(latest_tier, '#60a5fa')
        tb = {'HIGH': 'rgba(251,191,36,0.15)',
               'MID':  'rgba(96,165,250,0.15)',
               'LOW':  'rgba(74,222,128,0.15)'}.get(latest_tier, 'rgba(96,165,250,0.15)')
        label = {'HIGH': '과열 구간', 'MID': '중립 구간', 'LOW': '저평가 구간'}.get(latest_tier, '')

        st.markdown(f"""
<div style="background:{tb}; border:2px solid {tc}; border-radius:14px;
            padding:24px; text-align:center; margin-top:12px;">
  <p style="color:#94a3b8; margin:0; font-size:14px;">📊 현재 시장 상태</p>
  <h1 style="color:{tc}; margin:8px 0; font-size:3rem;">{latest_tier}</h1>
  <h2 style="color:white; margin:0;">Eval = {latest_eval:.2%}</h2>
  <p style="color:#94a3b8; margin-top:10px;">QQQ가 5년 추세 대비<br>
     <b style="color:{tc};">{abs(latest_eval):.1%} {label}</b></p>
</div>
""", unsafe_allow_html=True)

        # 최근 Eval 추이 차트
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
            fig_e.add_hline(y=7.0,  line_color='#fbbf24', line_dash='dash', line_width=1,
                             annotation_text="HIGH", annotation_font=dict(color='#fbbf24', size=9))
            fig_e.add_hline(y=-7.0, line_color='#4ade80', line_dash='dash', line_width=1,
                             annotation_text="LOW",  annotation_font=dict(color='#4ade80', size=9))
            fig_e.add_hline(y=0,    line_color='#475569', line_dash='dot',  line_width=1)
            fig_e.update_layout(
                title='최근 1년 Eval 추이',
                yaxis_title='Eval (%)',
                height=230, showlegend=False, **CHART_LAYOUT
            )
            apply_grid(fig_e)
            st.plotly_chart(fig_e, use_container_width=True)

    st.divider()

    # ── Step 2: 3-티어 시스템 ─────────────────────────────────
    st.markdown("### 🎚️ Step 2. 3-티어 시장 분류")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
<div class="tier-high">
<h4>🟡 HIGH 티어 — 과열 구간</h4>
<b>조건:</b> Eval ≥ +7.0%<br><br>
시장이 추세보다 뜨거운 상태.<br>
주가 상승 시 차익을 적극 실현해<br>
현금을 축적합니다.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right"><b>1.5× (기본)</b></td></tr>
  <tr><td>매수 배율</td><td align="right">0.4×</td></tr>
</table>
→ <b>팔아서 현금 쌓기</b>
</div>
""", unsafe_allow_html=True)

    with t2:
        st.markdown("""
<div class="tier-mid">
<h4>🔵 MID 티어 — 중립 구간</h4>
<b>조건:</b> −7% < Eval < +7.0%<br><br>
시장이 추세 근처에서 움직이는<br>
평상시 상태.<br>
균형 잡힌 비율로<br>기계적 리밸런싱.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right"><b>0.6×</b></td></tr>
  <tr><td>매수 배율</td><td align="right"><b>0.6×</b></td></tr>
</table>
→ <b>균형 유지</b>
</div>
""", unsafe_allow_html=True)

    with t3:
        st.markdown("""
<div class="tier-low">
<h4>🟢 LOW 티어 — 저평가 구간</h4>
<b>조건:</b> Eval ≤ −7%<br><br>
시장이 추세보다 차가운 상태.<br>
역사적으로 가장 강한<br>
매수 기회 구간.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right">0.35×</td></tr>
  <tr><td>매수 배율</td><td align="right"><b>2.0× ★</b></td></tr>
</table>
→ <b>공격적 매수 — 핵심!</b>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Step 3: 매매 로직 ─────────────────────────────────────
    st.markdown("### ⚙️ Step 3. 매매 실행 로직")
    st.markdown("**매주 금요일 장 마감 10분 전(오후 3:50) LOC 주문으로 실행합니다.**")

    st.code("""
매주 금요일 장 마감 시:

이번 주 TQQQ가 전주 대비 상승했다면:
    평가 수익   = 보유수량 × (이번 주 가격 - 전주 가격)
    매도 수량   = int(평가 수익 × 매도 배율[티어] / 현재가)
    실제 매도   = min(매도 수량, 현재 보유수량)   ← 공매도 방지
    → LOC 매도 주문 실행 → 현금 증가

이번 주 TQQQ가 전주 대비 하락했다면:
    평가 손실   = 보유수량 × (전주 가격 - 이번 주 가격)
    매수 예산   = min(보유 현금, 평가 손실 × 매수 배율[티어])
    매수 수량   = int(매수 예산 / 현재가)
    → LOC 매수 주문 실행 → 주식 증가

변동 없으면:
    → 관망 (주문 없음)
""", language='python')

    # 상승/하락 시 설명
    col_up, col_dn = st.columns(2)
    with col_up:
        st.info("""
**📈 상승 시 매도 — 차익 실현**

평가 수익의 일정 배율만큼 매도해 현금을 확보합니다.

- **HIGH 티어** → 1.5× : 수익의 1.5배 어치 매도 (적극 실현)
- **MID 티어** → 0.6× : 수익의 60%만 매도 (적정 실현)
- **LOW 티어** → 0.35× : 수익의 35%만 매도 (최소 실현)

→ 과열 구간에서 더 많이 팔아 현금을 쌓아둡니다.
→ 이 현금이 LOW 티어 발생 시 매수 탄약이 됩니다.
""")

    with col_dn:
        st.success("""
**📉 하락 시 매수 — 저점 매집**

평가 손실의 배율만큼 현금을 투입해 추가 매수합니다.

- **HIGH 티어** → 0.4× : 손실의 40% 예산 매수 (소극)
- **MID 티어** → 0.6× : 손실의 60% 예산 매수 (적정)
- **LOW 티어** → 2.0× : 손실의 200% 예산 매수 (공격 ★)

→ 시장이 차가울수록 더 많이 사서 평균 단가를 낮춥니다.
→ 현금이 부족하면 LOW 티어 기회를 살릴 수 없으므로
  초기 현금 비중 40% 유지를 권장합니다.
""")

    st.divider()

    # ── 파라미터 비교표 ───────────────────────────────────────
    st.markdown("### 📋 현재 vs 최적화 파라미터 비교")

    param_df = pd.DataFrame({
        '파라미터':    ['HIGH 기준 (Eval ≥)', 'LOW 기준 (Eval ≤)',
                       '매도 HIGH', '매도 MID', '매도 LOW',
                       '매수 HIGH', '매수 MID', '매수 LOW ★',
                       '— 결과 —', 'CAGR', 'MDD', 'Calmar'],
        '현재 설정':   ['+7.0%', '−7.0%',
                       '1.5×', '0.6×', '0.35×',
                       '0.4×', '0.6×', '2.0×',
                       '', '~42%', '~32%', '~1.32'],
        '최적화 설정': ['+6.0%', '−6.0%',
                       '2.0×', '0.3×', '0.2×',
                       '1.0×', '0.6×', '2.0×',
                       '', '~43%', '~31%', '~1.40'],
        '변화 핵심':   ['기준 하향 → 더 빠른 차익', 'LOW 범위 축소',
                       '과열 시 더 적극 실현', '중립 매도 줄임', 'LOW일 때 최소 실현',
                       '과열 시 매수 강화', '중립 매수 소폭 감소', '저점 시 공격 유지',
                       '', '+1%p 개선', '1%p 개선', '0.08 개선'],
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 실전 팁 ───────────────────────────────────────────────
    st.markdown("### 💡 실전 운용 체크리스트")

    tip1, tip2, tip3 = st.columns(3)
    with tip1:
        st.warning("""
**⏰ 주문 타이밍**

- 매주 **금요일** 장 마감 10분 전
- 오후 3:50 기준 LOC 주문 입력
- 목요일에 미리 계산 후 금요일 확인
- 시장 휴장 시 전날(목요일) 대체
""")
    with tip2:
        st.info("""
**💰 자본 배분 원칙**

- 초기 현금 비중 **40~50%** 권장
- 현금 = LOW 티어 발생 시 매수 탄약
- LOW 티어 때 현금 소진 → 기회 상실
- 포지션 비중도 주기적으로 점검
""")
    with tip3:
        st.success("""
**🧠 핵심 마인드셋**

- Eval 낮을수록 = 기회, 두려워하지 않기
- 하락 시 매수 배율 낮추면 전략 붕괴
- 파라미터는 월 1회 검토, 잦은 변경 ×
- B&H 대비 MDD 1/3 = 위대리의 핵심 가치
""")

    # ── B&H 비교 강조 ─────────────────────────────────────────
    st.divider()
    st.markdown("""
### 📊 위대리 vs TQQQ 단순 보유(B&H) 핵심 비교

| 항목 | 위대리 (현재) | 위대리 (최적화) | TQQQ B&H |
|------|:---:|:---:|:---:|
| **연평균 수익(CAGR)** | ~42% | **~43%** | ~40% |
| **최대 낙폭(MDD)** | **~32%** | **~31%** | ~80% |
| **Calmar (수익/위험)** | 1.32 | **1.40** | ~0.50 |
| **심리적 안정성** | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| **주간 주문 횟수** | 1회 | 1회 | 없음 |

> **TQQQ를 그냥 들고 있으면 -80% 이상의 하락을 버텨야 합니다.**
> 위대리는 비슷한 수익을 내면서 최대 낙폭을 **1/3 수준**으로 줄여줍니다.
> 낙폭이 작으면 복리 효과도 극대화됩니다. (MDD -80% 회복엔 +400% 상승이 필요!)
""")
