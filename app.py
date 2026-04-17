# wedaeri_app.py — TQQQ 위대리 v4.0
# Tab1: 실전 트레이딩 | Tab2: 백테스트 분석 | Tab3: 전략 로직

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        st_start      = st.date_input("투자 시작일", datetime.strptime("2025-12-26", "%Y-%m-%d"))
        st_cap        = st.number_input("시작 원금 ($)", value=108000.00, step=1000.0)
        st_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, 40) / 100
        if st.button("🔄 데이터 갱신", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.markdown("""
**📌 현재 파라미터**

| 티어 | Eval 조건 | 매도× | 매수× |
|------|----------|-------|-------|
| HIGH | ≥ +5.5%  | 1.5   | 0.5   |
| MID  | 중간      | 0.6   | 0.6   |
| LOW  | ≤ −7%   | 0.33  | 2.0   |

**📌 최적화 파라미터**

| 티어 | Eval 조건 | 매도× | 매수× |
|------|----------|-------|-------|
| HIGH | ≥ +5.5%  | 2.0   | 0.5   |
| MID  | 중간      | 0.6   | 1.0   |
| LOW  | ≤ −10%  | 0.5   | 5.0   |

CAGR +9.5%p, MDD 개선
""")

# ─────────────────────────────────────────────────────────────
# 3. 데이터 로딩 — O(n) Rolling OLS polyfit
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_wedaeri_data():
    """QQQ + TQQQ 로딩 & O(n) rolling OLS로 Growth/Eval 계산"""
    try:
        raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01",
                          auto_adjust=True, progress=False)
        df = (raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw).copy()
        df = df.reset_index()
        df.columns = [c.strip() for c in df.columns]
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        df = df.dropna(subset=['QQQ']).reset_index(drop=True)

        # O(n) Rolling OLS: log(QQQ) = a + b*t  →  Growth = exp(a + b*t_i)
        W  = 1260
        t  = df['Date'].map(pd.Timestamp.toordinal).values.astype(float)
        y  = np.log(df['QQQ'].values.astype(float))

        cum_t  = np.cumsum(np.insert(t,    0, 0))
        cum_y  = np.cumsum(np.insert(y,    0, 0))
        cum_t2 = np.cumsum(np.insert(t**2, 0, 0))
        cum_ty = np.cumsum(np.insert(t*y,  0, 0))

        growth = np.full(len(df), np.nan)
        for i in range(W, len(df)):
            s_t  = cum_t[i]  - cum_t[i-W]
            s_y  = cum_y[i]  - cum_y[i-W]
            s_t2 = cum_t2[i] - cum_t2[i-W]
            s_ty = cum_ty[i] - cum_ty[i-W]
            denom = W * s_t2 - s_t**2
            if denom == 0:
                continue
            b = (W * s_ty - s_t * s_y) / denom
            a = (s_y - b * s_t) / W
            growth[i] = np.exp(a + b * t[i])

        df['Growth'] = growth
        df['Eval']   = df['QQQ'] / df['Growth'] - 1
        return df

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 4. 실전 시뮬레이션 (Tab 1용 — 지정 시작일부터)
# ─────────────────────────────────────────────────────────────
def run_wedaeri_sim(data, start_dt, init_cap, cash_ratio,
                    hc=0.055, lc=-0.07,
                    sH=1.5, sM=0.6, sL=0.33,
                    bH=0.5, bM=0.6, bL=2.0):
    sim  = data[data['Date'] >= pd.to_datetime(start_dt)].copy()
    wkly = (sim[sim['Date'].dt.weekday == 4]
            .dropna(subset=['Eval', 'TQQQ'])
            .reset_index(drop=True))
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
def run_full_backtest(data, init_cap=20_000, cash_ratio=0.50,
                      hc=0.055, lc=-0.07,
                      sH=1.5,  sM=0.6,  sL=0.33,
                      bH=0.5,  bM=0.6,  bL=2.0):
    wkly = (data[data['Date'].dt.weekday == 4]
            .dropna(subset=['Eval', 'TQQQ'])
            .reset_index(drop=True))
    if wkly.empty:
        return None

    P     = wkly['TQQQ'].values.astype(float)
    EV    = wkly['Eval'].values.astype(float)
    dates = wkly['Date'].values
    N     = len(wkly)
    YEARS = N / 52

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
    neg   = rets[rets < 0]
    dstd  = neg.std() * np.sqrt(52) if len(neg) > 1 else 1e-9
    ann_r = (1 + rets.mean()) ** 52 - 1
    sor   = ann_r / dstd
    cal   = cagr / abs(mdd) if mdd != 0 else 0

    bh_peak = np.maximum.accumulate(P)
    return {
        'cagr': cagr,  'mdd': mdd,   'cal': cal,  'sor': sor,
        'final': eq[-1], 'init': init_cap, 'years': YEARS,
        'eq': eq, 'dd': dd, 'dates': dates,
        'tiers': tiers, 'prices': P, 'eval': EV,
        'bh_eq':   P / P[0] * init_cap,
        'bh_cagr': (P[-1] / P[0]) ** (1 / YEARS) - 1,
        'bh_mdd':  (P / bh_peak - 1).min(),
        'rets': rets,
    }


# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
with st.spinner("📡 데이터 로딩 중 (최초 1회만 시간이 걸립니다)..."):
    df = load_wedaeri_data()

if df.empty:
    st.error("데이터 로드 실패. 잠시 후 새로고침 해주세요.")
    st.stop()

log_df = run_wedaeri_sim(df, st_start, st_cap, st_cash_ratio)

tqqq_series = df['TQQQ'].dropna()
latest_tqqq = float(tqqq_series.iloc[-1]) if not tqqq_series.empty else 0.0
eval_series  = df['Eval'].dropna()
latest_eval  = float(eval_series.iloc[-1]) if not eval_series.empty else 0.0
latest_tier  = 'HIGH' if latest_eval >= 0.055 else ('LOW' if latest_eval <= -0.07 else 'MID')

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
            ["📋 현재 파라미터  (CAGR ~38%, MDD ~31%)",
             "🏆 최적화 파라미터 (CAGR ~48%, MDD ~28%)",
             "🛡️ 안정형 파라미터 (MDD 중시)",
             "✏️ 직접 설정"],
            index=0
        )

        PRESETS = {
            "현재":  dict(hc=0.055, lc=-0.07,  sH=1.5, sM=0.6, sL=0.33, bH=0.5, bM=0.6, bL=2.0),
            "최적화": dict(hc=0.055, lc=-0.10,  sH=2.0, sM=0.6, sL=0.5,  bH=0.5, bM=1.0, bL=5.0),
            "안정형": dict(hc=0.08,  lc=-0.10,  sH=2.0, sM=0.6, sL=0.5,  bH=0.5, bM=0.6, bL=5.0),
        }

        key_map = {"📋 현재": "현재", "🏆 최적화": "최적화", "🛡️ 안정형": "안정형"}
        pkey    = next((v for k, v in key_map.items() if k in preset), None)
        P_DEF   = PRESETS[pkey] if pkey else PRESETS["현재"]

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**기본 설정**")
            bt_cap        = st.number_input("초기 자본 ($)", value=20_000, step=1000, key='bt_cap')
            bt_cash_ratio = st.slider("초기 현금 비중 (%)", 0, 100, 50, key='bt_cr') / 100
            st.markdown("**시장 평가 기준**")
            bt_hc = st.slider("HIGH 기준 Eval ≥ (%)", 1.0, 20.0,
                               P_DEF['hc']*100, 0.5, key='bt_hc') / 100
            bt_lc = st.slider("LOW 기준  Eval ≤ (%)", -20.0, -1.0,
                               P_DEF['lc']*100, 0.5, key='bt_lc') / 100

        with p2:
            st.markdown("**매도 배율** (상승 시 차익실현 강도)")
            bt_sH = st.slider("매도 HIGH ×", 0.1, 5.0,  P_DEF['sH'], 0.1, key='bt_sH')
            bt_sM = st.slider("매도 MID  ×", 0.1, 3.0,  P_DEF['sM'], 0.1, key='bt_sM')
            bt_sL = st.slider("매도 LOW  ×", 0.1, 2.0,  P_DEF['sL'], 0.05, key='bt_sL')

        with p3:
            st.markdown("**매수 배율** (하락 시 추가 매수 강도)")
            bt_bH = st.slider("매수 HIGH ×", 0.1, 3.0,  P_DEF['bH'], 0.1, key='bt_bH')
            bt_bM = st.slider("매수 MID  ×", 0.1, 3.0,  P_DEF['bM'], 0.1, key='bt_bM')
            bt_bL = st.slider("매수 LOW  ×", 0.5, 10.0, P_DEF['bL'], 0.5, key='bt_bL')

        if pkey:
            vals = PRESETS[pkey]
            st.info(
                f"💡 **{pkey} 프리셋** — "
                f"hc={vals['hc']:.1%} / lc={vals['lc']:.1%} | "
                f"매도 {vals['sH']}/{vals['sM']}/{vals['sL']} | "
                f"매수 {vals['bH']}/{vals['bM']}/{vals['bL']}  "
                f"→ 슬라이더가 자동 반영됩니다."
            )

    # ── 백테스트 실행 ─────────────────────────────────────────
    with st.spinner("🔄 백테스트 계산 중..."):
        bt_cur = run_full_backtest(
            df, bt_cap, bt_cash_ratio,
            hc=bt_hc, lc=bt_lc,
            sH=bt_sH, sM=bt_sM, sL=bt_sL,
            bH=bt_bH, bM=bt_bM, bL=bt_bL,
        )
        bt_opt = run_full_backtest(
            df, bt_cap, bt_cash_ratio,
            hc=0.055, lc=-0.10,
            sH=2.0, sM=0.6, sL=0.5,
            bH=0.5,  bM=1.0, bL=5.0,
        )

    if bt_cur is None:
        st.warning("백테스트 데이터가 부족합니다.")
        st.stop()

    dates = pd.to_datetime(bt_cur['dates'])

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
        title=f'주간 수익률 분포  |  양전 비율 {pos_pct:.1f}%  |  최대 낙주 {wr.min():.1f}%',
        xaxis_title='주간 수익률 (%)',
        height=260, showlegend=False, **CHART_LAYOUT
    )
    apply_grid(fig_ret)
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── 티어별 성과 ───────────────────────────────────────────
    st.markdown("### 🎯 티어별 성과 분석")

    tiers_arr = np.array(bt_cur['tiers'])
    wr_full   = np.concatenate([[0], wr])  # 첫 주 0 패딩

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
            fig_e.add_hline(y=5.5,  line_color='#fbbf24', line_dash='dash', line_width=1,
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
<b>조건:</b> Eval ≥ +5.5%<br><br>
시장이 추세보다 뜨거운 상태.<br>
주가 상승 시 차익을 적극 실현해<br>
현금을 축적합니다.<br><br>
<table width="100%">
  <tr><td>매도 배율</td><td align="right"><b>1.5× (기본)</b></td></tr>
  <tr><td>매수 배율</td><td align="right">0.5×</td></tr>
</table>
→ <b>팔아서 현금 쌓기</b>
</div>
""", unsafe_allow_html=True)

    with t2:
        st.markdown("""
<div class="tier-mid">
<h4>🔵 MID 티어 — 중립 구간</h4>
<b>조건:</b> −7% < Eval < +5.5%<br><br>
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
  <tr><td>매도 배율</td><td align="right">0.33×</td></tr>
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
- **LOW 티어** → 0.33× : 수익의 33%만 매도 (최소 실현)

→ 과열 구간에서 더 많이 팔아 현금을 쌓아둡니다.
→ 이 현금이 LOW 티어 발생 시 매수 탄약이 됩니다.
""")

    with col_dn:
        st.success("""
**📉 하락 시 매수 — 저점 매집**

평가 손실의 배율만큼 현금을 투입해 추가 매수합니다.

- **HIGH 티어** → 0.5× : 손실의 50% 예산 매수 (소극)
- **MID 티어** → 0.6× : 손실의 60% 예산 매수 (적정)
- **LOW 티어** → 2.0× : 손실의 200% 예산 매수 (공격 ★)

→ 시장이 차가울수록 더 많이 사서 평균 단가를 낮춥니다.
→ 현금이 부족하면 LOW 티어 기회를 살릴 수 없으므로
  초기 현금 비중 40~50% 유지를 권장합니다.
""")

    st.divider()

    # ── 파라미터 비교표 ───────────────────────────────────────
    st.markdown("### 📋 현재 vs 최적화 파라미터 비교")

    param_df = pd.DataFrame({
        '파라미터':    ['HIGH 기준 (Eval ≥)', 'LOW 기준 (Eval ≤)',
                       '매도 HIGH', '매도 MID', '매도 LOW',
                       '매수 HIGH', '매수 MID', '매수 LOW ★',
                       '— 결과 —', 'CAGR', 'MDD', 'Calmar'],
        '현재 설정':   ['+5.5%', '−7.0%',
                       '1.5×', '0.6×', '0.33×',
                       '0.5×', '0.6×', '2.0×',
                       '', '~38%', '~31%', '~1.24'],
        '최적화 설정': ['+5.5%', '−10.0%',
                       '2.0×', '0.6×', '0.5×',
                       '0.5×', '1.0×', '5.0× ★',
                       '', '~48%', '~28%', '~1.70'],
        '변화 핵심':   ['동일', 'LOW 범위 확장 → MDD 개선',
                       '과열 시 더 적극 실현', '동일', 'LOW일 때 최소 실현',
                       '동일', '중립 시 매수 강화', '저점 시 초공격적 매수',
                       '', '+9.5%p 개선', '3%p 개선', '0.46 개선'],
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
| **연평균 수익(CAGR)** | ~38% | **~48%** | ~40% |
| **최대 낙폭(MDD)** | **~31%** | **~28%** | ~80% |
| **Calmar (수익/위험)** | 1.24 | **1.70** | ~0.50 |
| **심리적 안정성** | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| **주간 주문 횟수** | 1회 | 1회 | 없음 |

> **TQQQ를 그냥 들고 있으면 -80% 이상의 하락을 버텨야 합니다.**
> 위대리는 비슷한 수익을 내면서 최대 낙폭을 **1/3 수준**으로 줄여줍니다.
> 낙폭이 작으면 복리 효과도 극대화됩니다. (MDD -80% 회복엔 +400% 상승이 필요!)
""")
