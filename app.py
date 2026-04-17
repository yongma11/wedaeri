"""
위대리 Quantum T-Flow v3.0 — 버그 수정판
=========================================
수정 목록:
  Fix ① st.secrets AttrDict 불변 객체 → dict() 복사 후 수정
  Fix ② gspread.authorize() deprecated → service_account_from_dict() 사용
  Fix ③ MultiIndex else 분기 컬럼명 오류 → 예외 발생으로 명확히 처리
  Fix ④ tz-aware vs tz-naive 날짜 비교 오류 → tz_localize(None) 통일
  Fix ⑤ polyfit 루프 try/except 누락 → 추가
  Fix ⑥ Eval NaN 시 티어 오판 → pd.isna() 명시 체크
  Fix ⑦ 매도 수량 round() 혼용 → int() 단일화import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="위대리 Quantum T-Flow v3.0", layout="wide")

# [사용자 설정]
SHEET_KEY = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE = '2025-12-26'
INITIAL_CAP = 108000
INITIAL_CASH_RATIO = 0.40

# 2. 데이터 엔진 (에러 방어 로직 강화)
@st.cache_data(ttl=3600)
def load_market_data():
    try:
        # 데이터 다운로드
        df_raw = yf.download(["QQQ", "TQQQ"], start="2000-01-01", auto_adjust=True, progress=False)
        
        # 데이터가 비어있는지 확인
        if df_raw.empty:
            return pd.DataFrame()

        # Multi-Index 대응
        df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df = df_close.dropna().reset_index()
        
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        
        # 데이터가 1260개보다 적으면 성장성 계산 불가
        if len(df) < 1260:
            return df

        # 성장성 계산 (5년 로그 회귀)
        results = [np.nan] * len(df)
        date_nums = df['Date'].map(pd.Timestamp.toordinal).values
        values = df['QQQ'].values
        for i in range(1260, len(df)):
            fit = np.polyfit(date_nums[i-1260:i], np.log(values[i-1260:i]), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        
        df['Growth'] = results
        df['Eval'] = (df['QQQ'] / df['Growth']) - 1
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {e}")
        return pd.DataFrame()

# 3. 메인 실행 부분
df = load_market_data()

# [핵심 수정] 데이터가 있는지 먼저 확인
if df.empty or 'Date' not in df.columns or len(df) == 0:
    st.error("⚠️ 야후 파이낸스에서 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
else:
    # 실시간 날짜 비교 (에러 발생했던 지점 수정)
    today_ts = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    last_date_ts = pd.to_datetime(df['Date'].iloc[-1]) # 이제 데이터가 있음이 보장됨

    # ... 이후 시뮬레이션 및 차트 로직 실행 ...
    st.success(f"✅ 데이터 로드 완료 (최신 날짜: {last_date_ts.strftime('%Y-%m-%d')})")
    
    # 상단 요약 지표 예시
    sim_df = df[df['Date'] >= pd.to_datetime(START_DATE)].copy()
    if not sim_df.empty:
        st.write(f"📊 시뮬레이션 시작일: {START_DATE}")
        # (여기에 이전에 작성한 시뮬레이션 및 차트 코드를 이어서 붙여넣으세요)
        st.line_chart(sim_df.set_index('Date')['TQQQ'])
    else:
        st.warning(f"설정한 시작일({START_DATE}) 이후의 데이터가 아직 없습니다.")
  추가: 평단가 실시간 갱신, 실시간 현재가 반영, 보유 현황 표시
"""

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# ── 1. 페이지 설정 ────────────────────────────────────────────────────────────
st.set_page_config(page_title="위대리 Quantum T-Flow v3.0", layout="wide")

# ── 2. 핵심 파라미터 ──────────────────────────────────────────────────────────
SHEET_KEY          = '1s8XX-8PUAWyWOHOwst2W-b99pQo1_aFtLVg5uTD_HMI'
START_DATE         = '2025-12-26'
INITIAL_CAP        = 108_000        # 시작 원금 ($)
INITIAL_CASH_RATIO = 0.40           # 초기 현금 비중 (40%)

SETTINGS = {
    'high_cut'   : 0.055,           # Eval 상단 컷 → HIGH 티어
    'low_cut'    : -0.07,           # Eval 하단 컷 → LOW  티어
    'sell_ratios': {'HIGH': 1.5, 'MID': 0.6, 'LOW': 0.33},  # 상승 시 매도 배율
    'buy_ratios' : {'HIGH': 0.5, 'MID': 0.6, 'LOW': 2.0},   # 하락 시 매수 배율
}

# ── 3. 구글 시트 인증 ─────────────────────────────────────────────────────────
def get_gspread_client():
    try:
        # ✅ Fix ①: dict()로 복사 → AttrDict 불변 객체 수정 오류(TypeError) 해결
        creds_info = dict(st.secrets["gcp_service_account"])
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

        # ✅ Fix ②: gspread.authorize() deprecated → service_account_from_dict 사용
        return gspread.service_account_from_dict(
            creds_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
    except Exception as e:
        st.error(f"❌ 구글 인증 실패 ({e})")
        return None


# ── 4. 데이터 수집 및 지표 계산 ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_and_process_data():
    end_dt = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df_raw = yf.download(
        ["QQQ", "TQQQ"],
        start="2000-01-01",
        end=end_dt,
        auto_adjust=True,
        progress=False,
    )

    # ✅ Fix ③: MultiIndex 구조 강제 확인 → else 분기의 잘못된 컬럼명 오류 방지
    if not isinstance(df_raw.columns, pd.MultiIndex):
        raise ValueError(
            "yfinance 응답 구조가 예상과 다릅니다. "
            "QQQ·TQQQ 두 티커를 동시에 조회해야 합니다."
        )
    df = df_raw['Close'].copy().dropna().reset_index()
    if 'Date' not in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)

    # ✅ Fix ④: tz-aware → tz-naive 변환 (날짜 비교 TypeError 방지)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # 5년(≈1260 거래일) 롤링 로그 회귀 → Growth 추세선 계산
    date_nums = df['Date'].map(pd.Timestamp.toordinal).values
    qqq_vals  = df['QQQ'].values
    results   = [np.nan] * len(df)

    # ✅ Fix ⑤: try/except 추가 → 데이터 이상치로 인한 앱 크래시 방지
    for i in range(1260, len(df)):
        try:
            fit        = np.polyfit(date_nums[i-1260:i], np.log(qqq_vals[i-1260:i]), 1)
            results[i] = np.exp(fit[1] + fit[0] * date_nums[i])
        except Exception:
            pass

    df['Growth'] = results
    df['Eval']   = (df['QQQ'] / df['Growth']) - 1
    return df


# ── 5. 실시간 현재가 조회 ────────────────────────────────────────────────────
def get_live_prices():
    """TQQQ·QQQ 실시간(또는 직전 종가) 가격 반환. 조회 실패 시 None."""
    try:
        tqqq_fi = yf.Ticker("TQQQ").fast_info
        qqq_fi  = yf.Ticker("QQQ").fast_info
        t_p = tqqq_fi.get('last_price') or tqqq_fi.get('previousClose')
        q_p = qqq_fi.get('last_price')  or qqq_fi.get('previousClose')
        return (float(t_p) if t_p else None), (float(q_p) if q_p else None)
    except Exception:
        return None, None


# ── 6. 시뮬레이션 엔진 ───────────────────────────────────────────────────────
def run_simulation(weekly):
    """
    매주 금요일 기준 TQQQ 매매 시뮬레이션.

    전략 요약:
    - Eval = (QQQ / Growth) - 1 로 시장 과열/침체 판단
    - 가격 상승 시: 보유 평가이익 × sell_ratio 만큼의 금액에 해당하는 주수 매도
      (HIGH 티어일수록 더 공격적 매도)
    - 가격 하락 시: 보유 평가손실 × buy_ratio 만큼의 금액에 해당하는 주수 매수
      (LOW 티어일수록 더 공격적 매수)

    반환: (총자산 이력 리스트, 액션 로그 리스트, 최종 보유주수, 최종 현금, 최종 평단가)
    """
    cash      = INITIAL_CAP * INITIAL_CASH_RATIO
    shares    = int((INITIAL_CAP * (1 - INITIAL_CASH_RATIO)) / weekly['TQQQ'].iloc[0])
    avg_price = float(weekly['TQQQ'].iloc[0])
    history   = []
    action_log = []

    for i in range(len(weekly)):
        p      = float(weekly.loc[i, 'TQQQ'])
        m_eval = weekly.loc[i, 'Eval']

        # ✅ Fix ⑥: pd.isna() 명시 체크 → NaN 비교 시 묵시적 MID 오판 방지
        if pd.isna(m_eval):
            tier = 'MID'
        elif m_eval >= SETTINGS['high_cut']:
            tier = 'HIGH'
        elif m_eval <= SETTINGS['low_cut']:
            tier = 'LOW'
        else:
            tier = 'MID'

        label = "보유"

        if i > 0:
            prev_p = float(weekly.loc[i-1, 'TQQQ'])
            diff   = shares * (p - prev_p)   # 보유 주식 평가금액 변화

            if diff > 0:
                # 상승 → 일부 매도
                # ✅ Fix ⑦: round() 제거 → int() 단일화, 부동소수점 반올림 불일치 해소
                q = min(int(diff * SETTINGS['sell_ratios'][tier] / p), shares)
                if q > 0:
                    shares -= q
                    cash   += q * p
                    label   = f"매도 {q}주"

            elif diff < 0:
                # 하락 → 일부 매수
                spend = min(cash, abs(diff) * SETTINGS['buy_ratios'][tier])
                q     = int(spend / p)
                if q > 0:
                    # ✅ 평단가 갱신 (기존 코드 누락 → 추가)
                    avg_price  = (avg_price * shares + p * q) / (shares + q)
                    shares    += q
                    cash      -= q * p
                    label      = f"매수 {q}주"

        total = cash + shares * p
        history.append(total)
        action_log.append({
            '날짜'    : weekly.loc[i, 'Date'].strftime('%Y-%m-%d'),
            'TQQQ'   : round(p, 2),
            'Eval'   : f"{m_eval:.2%}" if not pd.isna(m_eval) else '-',
            'Tier'   : tier,
            '액션'   : label,
            '주수'   : shares,
            '평단가' : round(avg_price, 2),
            '현금'   : round(cash, 2),
            '총자산' : round(total, 2),
        })

    return history, action_log, shares, cash, avg_price


# ════════════════════════════════════════════════════════════════════════════════
# 메인 대시보드
# ════════════════════════════════════════════════════════════════════════════════
st.title("🚀 위대리 Quantum T-Flow v3.0")
st.caption(
    f"기준일: {START_DATE}  |  원금: ${INITIAL_CAP:,.0f}  |"
    f"  초기현금: {INITIAL_CASH_RATIO:.0%}  |"
    f"  마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# 데이터 로드
with st.spinner("데이터 불러오는 중..."):
    df = fetch_and_process_data()

# 실시간 가격 반영 (캐시된 df의 최신 행에 덮어쓰기)
tqqq_live, qqq_live = get_live_prices()
today_ts = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))

if tqqq_live:
    if today_ts > df['Date'].iloc[-1]:
        # 오늘 데이터가 yfinance에 아직 없으면 새 행 추가
        new_row = pd.DataFrame({
            'Date'  : [today_ts],
            'QQQ'   : [qqq_live or float(df['QQQ'].iloc[-1])],
            'TQQQ'  : [tqqq_live],
            'Growth': [df['Growth'].iloc[-1]],
            'Eval'  : [df['Eval'].iloc[-1]],
        })
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        # 오늘 행이 있으면 TQQQ 가격만 실시간값으로 갱신
        df.loc[df.index[-1], 'TQQQ'] = tqqq_live

# 금요일 데이터 필터링
start_ts = pd.Timestamp(START_DATE)
weekly = (
    df[df['Date'] >= start_ts]
    .loc[lambda d: d['Date'].dt.weekday == 4]
    .copy()
    .reset_index(drop=True)
)

if weekly.empty:
    st.warning("⚠️ 기준일 이후 금요일 데이터가 없습니다. START_DATE를 확인하세요.")
    st.stop()

# 시뮬레이션 실행
history, log, final_shares, final_cash, final_avg = run_simulation(weekly)
weekly['Total_Asset'] = history

# 현재 상태 계산
cur_asset  = weekly['Total_Asset'].iloc[-1]
profit_abs = cur_asset - INITIAL_CAP
profit_pct = (cur_asset / INITIAL_CAP - 1) * 100

raw_eval   = df['Eval'].iloc[-1]
eval_str   = f"{raw_eval:.2%}" if not pd.isna(raw_eval) else "계산 중"
tier_label = (
    'HIGH 🔴' if (not pd.isna(raw_eval) and raw_eval >= SETTINGS['high_cut']) else
    'LOW  🟢' if (not pd.isna(raw_eval) and raw_eval <= SETTINGS['low_cut'])  else
    'MID  🟡'
)

# ── 메트릭 카드 ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 자산",         f"${cur_asset:,.2f}",   f"${profit_abs:+,.2f}")
c2.metric("누적 수익률",     f"{profit_pct:.2f}%")
c3.metric("시장 평가(Eval)", eval_str)
c4.metric("현재 Tier",       tier_label)
c5.metric("TQQQ 현재가",     f"${tqqq_live:.2f}" if tqqq_live else "N/A")

# ── 자산 성장 차트 ────────────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weekly['Date'],
    y=weekly['Total_Asset'],
    name="총 자산",
    line=dict(color='#00FFCC', width=2),
    hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>',
))
fig.add_hline(
    y=INITIAL_CAP,
    line_dash="dash",
    line_color="#888888",
    annotation_text=f"원금 ${INITIAL_CAP:,.0f}",
    annotation_position="bottom right",
)
fig.update_layout(
    title="위대리 자산 성장 곡선",
    template="plotly_dark",
    yaxis_title="자산 ($)",
    xaxis_title="날짜",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── 현재 보유 현황 ────────────────────────────────────────────────────────────
st.subheader("📈 현재 보유 현황")
h1, h2, h3 = st.columns(3)
h1.metric("보유 주수", f"{final_shares}주")
h2.metric("평단가",    f"${final_avg:.2f}")
h3.metric("보유 현금", f"${final_cash:,.2f}")

# ── 주간 매매 이력 ────────────────────────────────────────────────────────────
with st.expander("📋 주간 매매 이력 보기"):
    st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

# ── 구글 시트 실시간 주문표 ───────────────────────────────────────────────────
st.divider()
st.subheader("📊 실시간 구글 시트 주문표")

if st.button("🔄 시트 새로고침"):
    st.rerun()

gc = get_gspread_client()
if gc:
    try:
        order_data = gc.open_by_key(SHEET_KEY).worksheet("위대리").get('L4:O4')
        if order_data:
            st.table(pd.DataFrame(order_data, columns=["액션", "방법", "가격", "수량"]))
        else:
            st.info("시트에 주문 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"시트 데이터 로드 오류: {e}")
