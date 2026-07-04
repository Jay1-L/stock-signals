# -*- coding: utf-8 -*-
"""
추세추종 매수 신호 스캐너 v3 — 전체 시장 스캔 (KRX 차단 우회)
한국 종목 목록: KRX 정보데이터시스템 -> KIND 공시사이트 -> 내장 목록 순서로 시도
미국: S&P500 전체 + 나스닥 주요종목
"""
import io
import json
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# ── 설정 ──
HOLD_PERIODS = [2, 3, 5, 10, 20, 60, 126, 252]
LOOKBACK_YEARS = 8
MA_PROXIMITY = 0.025
BOX_WINDOW = 40
BOX_TIGHTNESS = 0.12
MIN_SAMPLES = 8
MIN_TURNOVER_KRW = 500_000_000   # 한국 주식 최소 일평균 거래대금 5억 (초소형/유동성 없는 종목 제외)
WORKERS = 8

ETF_KR = [
    ("069500", "KODEX 200"), ("396500", "TIGER 차이나반도체FACTSET"),
    ("413600", "SOL 글로벌AI반도체탑픽액티브"), ("446770", "ACE 글로벌반도체TOP4 Plus"),
    ("305720", "KODEX 2차전지산업"), ("091160", "KODEX 반도체"),
    ("133690", "TIGER 미국나스닥100"), ("360750", "TIGER 미국S&P500"),
]
ETF_US = [
    ("SPY", "SPDR S&P 500"), ("QQQ", "Invesco QQQ"),
    ("SMH", "VanEck Semiconductor"), ("SOXL", "Direxion Semi Bull 3X"),
    ("TLT", "iShares 20Y Treasury"), ("IWM", "iShares Russell 2000"),
    ("GLD", "SPDR Gold"),
]
# 나스닥100 중 S&P500에 없는 주요 종목 (보완용)
EXTRA_US = [
    ("ARM", "Arm Holdings"), ("PDD", "PDD Holdings"), ("MELI", "MercadoLibre"),
    ("TEAM", "Atlassian"), ("DDOG", "Datadog"), ("ZS", "Zscaler"),
    ("MRVL", "Marvell"), ("ASML", "ASML"), ("AZN", "AstraZeneca"),
]
# 최후 폴백용 한국 대형주 (모든 목록 소스 실패 시)
FALLBACK_KR = [
    ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"), ("005380", "현대차"), ("000270", "기아"),
    ("068270", "셀트리온"), ("035420", "NAVER"), ("012450", "한화에어로스페이스"),
    ("042660", "한화오션"), ("267260", "HD현대일렉트릭"), ("034020", "두산에너빌리티"),
]


def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

MA_DEFS = [
    ("SMA10", sma, 10, "단기"), ("SMA20", sma, 20, "단기"),
    ("SMA50", sma, 50, "장기"), ("SMA200", sma, 200, "장기"),
    ("EMA10", ema, 10, "단기"), ("EMA21", ema, 21, "장기"),
    ("EMA30", ema, 30, "장기"),
]


def backtest(close, entries, hold):
    idx = np.where(entries.values)[0]
    idx = idx[idx + hold < len(close)]
    if len(idx) < MIN_SAMPLES:
        return None
    buy = close.values[idx]
    sell = close.values[idx + hold]
    ret = sell / buy - 1
    wins, losses = ret[ret > 0], ret[ret <= 0]
    win_rate = len(wins) / len(ret) * 100
    avg_p = wins.mean() * 100 if len(wins) else 0.0
    avg_l = losses.mean() * 100 if len(losses) else 0.0
    pl = abs(avg_p / avg_l) if avg_l != 0 else (999.99 if avg_p > 0 else 0)
    return dict(days=hold, win=round(win_rate, 1), n=int(len(ret)),
                avgP=round(avg_p, 1), avgL=round(avg_l, 1), pl=round(pl, 2))


def detect_signals(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    today = len(df) - 1
    out = []

    for label, fn, n, term in MA_DEFS:
        if len(df) < n + 30:
            continue
        ma = fn(c, n)
        rising = ma > ma.shift(5)
        near = (c / ma - 1).abs() <= MA_PROXIMITY
        cond = (near & rising).fillna(False)
        if cond.iloc[today]:
            zone = (ma.iloc[today] * (1 - MA_PROXIMITY), ma.iloc[today] * (1 + MA_PROXIMITY))
            out.append(("avg", f"{label} {term} 근접", cond, zone))

    if len(df) > 60:
        hh20 = h.rolling(20).max().shift(1)
        cond20 = (c > hh20).fillna(False)
        if cond20.iloc[today]:
            zone = (hh20.iloc[today], hh20.iloc[today] * 1.05)
            out.append(("high", "20일신고가 돌파", cond20, zone))
        ath = c.cummax().shift(1)
        cond_ath = (c > ath).fillna(False)
        if cond_ath.iloc[today]:
            zone = (ath.iloc[today], ath.iloc[today] * 1.05)
            out.append(("high", "역대최고 돌파", cond_ath, zone))

    if len(df) > BOX_WINDOW + 20:
        box_hi = h.rolling(BOX_WINDOW).max().shift(1)
        box_lo = l.rolling(BOX_WINDOW).min().shift(1)
        tight = (box_hi / box_lo - 1) <= BOX_TIGHTNESS
        cond = ((c > box_hi) & tight).fillna(False)
        if cond.iloc[today]:
            zone = (box_hi.iloc[today], box_hi.iloc[today] * 1.05)
            out.append(("box", "박스돌파 단기", cond, zone))
    return out


def analyze_ticker(name, df, country, asset, mcap=None):
    if df is None or len(df) < 80:
        return None
    df = df.dropna(subset=["Close"]).copy()
    if df["Close"].iloc[-30:].nunique() < 5:   # 거래정지 등 제외
        return None
    # 한국 주식: 유동성 필터 (일평균 거래대금)
    if country == "KR" and asset == "stock" and "Volume" in df.columns:
        turnover = (df["Close"] * df["Volume"]).iloc[-20:].mean()
        if turnover < MIN_TURNOVER_KRW:
            return None
    close = df["Close"]
    signals = detect_signals(df)
    if not signals:
        return None

    best = None
    for sig_type, sig_label, cond, zone in signals:
        periods = []
        for h in HOLD_PERIODS:
            r = backtest(close, cond, h)
            if r:
                periods.append(r)
        if not periods:
            continue
        top = max(periods, key=lambda r: r["win"] * min(r["pl"], 10))
        score = top["win"] * min(top["pl"], 10)
        if best is None or score > best["score"]:
            best = dict(sig=sig_type, label=sig_label, top=top,
                        periods=periods, zone=zone, score=score)
    if best is None:
        return None

    last = float(close.iloc[-1])
    lo, hi = best["zone"]
    momentum = float(close.iloc[-1] / close.iloc[-min(252, len(close) - 1)] - 1)
    return dict(
        name=name, country=country, asset=asset,
        sig=best["sig"],
        desc=best["label"] + " · " + str(best["top"]["days"]) + "일 보유",
        close=round(last, 2), zone=[round(lo, 2), round(hi, 2)],
        inZone=bool(lo <= last <= hi),
        stats=best["top"], periods=best["periods"],
        mcap=mcap, _mom=momentum,
    )


def finalize(results):
    if not results:
        return []
    moms = np.array([r["_mom"] for r in results])
    for r in results:
        r["rs"] = round(float((moms < r["_mom"]).mean() * 100), 1)
        del r["_mom"]
    # 시총 있는 종목 우선, 그 다음 RS순
    results.sort(key=lambda r: (-(r["mcap"] or 0), -r["rs"]))
    return results


# ──────────────────────────────────────────────
# 한국 종목 목록: 3중 폴백
# ──────────────────────────────────────────────
def kr_list_from_krx():
    """1순위: KRX 정보데이터시스템 (해외 IP 차단 가능성 있음)"""
    import FinanceDataReader as fdr
    krx = fdr.StockListing("KRX")
    krx = krx[krx["Market"].isin(["KOSPI", "KOSDAQ"])]
    out = []
    for r in krx.itertuples():
        name = str(r.Name)
        if "스팩" in name or name.endswith(("우", "우B", "우C", "1우", "2우B", "3우B")):
            continue
        mcap = (getattr(r, "Marcap", 0) or 0) / 1400  # 달러 환산
        out.append((str(r.Code).zfill(6), name, mcap if mcap > 0 else None))
    return out


def kr_list_from_krx_desc():
    """1.5순위: KRX-DESC (공시사이트 KIND 경유 상장법인 목록, 차단 덜함)"""
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX-DESC")
    sym_col = "Symbol" if "Symbol" in df.columns else "Code"
    out = []
    for r in df.itertuples():
        name = str(getattr(r, "Name", ""))
        code = str(getattr(r, sym_col, "")).zfill(6)
        if not name or not code.isdigit():
            continue
        if "스팩" in name:
            continue
        out.append((code, name, None))
    return out


def kr_list_from_kind():
    """2순위: KIND 상장법인목록 다운로드 (차단이 덜함)"""
    import requests
    url = ("https://kind.krx.co.kr/corpgeneral/corpList.do"
           "?method=download&searchType=13")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
    res = requests.get(url, headers=headers, timeout=60)
    res.raise_for_status()
    html = res.content.decode("euc-kr", errors="ignore")
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    out = []
    for r in df.itertuples():
        name = str(getattr(r, "회사명", ""))
        code = str(getattr(r, "종목코드", "")).zfill(6)
        if not name or not code.isdigit():
            continue
        if "스팩" in name:
            continue
        out.append((code, name, None))  # KIND에는 시총 정보 없음
    return out


def build_universe():
    universe = []  # (ticker, name, country, asset, mcap_usd)
    meta = {"krSource": None, "krListed": 0, "usListed": 0, "warnings": []}

    # ── 한국: 3중 폴백 ──
    kr = None
    sources = [("KRX", kr_list_from_krx),
               ("KRX-DESC", kr_list_from_krx_desc),
               ("KIND", kr_list_from_kind)]
    for src_name, fn in sources:
        try:
            kr = fn()
            if kr and len(kr) > 100:
                print(f"한국 목록 소스: {src_name} ({len(kr)}개)")
                meta["krSource"] = src_name
                break
            kr = None
        except Exception as e:
            print(f"{src_name} 목록 실패: {str(e)[:120]}")
    if kr is None:
        print("모든 한국 목록 소스 실패 -> 내장 대형주 목록 사용")
        kr = [(c, n, None) for c, n in FALLBACK_KR]
        meta["krSource"] = "내장목록"
        meta["warnings"].append(
            f"한국 전체 종목 목록을 불러오지 못해 대형주 {len(FALLBACK_KR)}개만 스캔했어요. "
            "코스피·코스닥 전체가 반영되지 않은 결과입니다.")
    if meta["krSource"] == "KRX-DESC" or meta["krSource"] == "KIND":
        meta["warnings"].append(
            "시가총액 정보를 제공하지 않는 목록 소스를 사용해 한국 종목의 시총 필터가 동작하지 않아요.")
    meta["krListed"] = len(kr)
    for code, name, mcap in kr:
        universe.append((code, name, "KR", "stock", mcap))

    # ── 미국: S&P500 + 보완 종목 ──
    us_seen = set()
    try:
        import FinanceDataReader as fdr
        sp = fdr.StockListing("S&P500")
        sym_col = "Symbol" if "Symbol" in sp.columns else sp.columns[0]
        name_col = "Name" if "Name" in sp.columns else sp.columns[1]
        for r in sp.itertuples():
            sym = str(getattr(r, sym_col))
            if sym not in us_seen:
                us_seen.add(sym)
                universe.append((sym, str(getattr(r, name_col)), "US", "stock", None))
    except Exception as e:
        print("S&P500 목록 실패:", str(e)[:120])
        meta["warnings"].append("미국 S&P500 목록을 불러오지 못해 일부 종목만 스캔했어요.")
    for sym, name in EXTRA_US:
        if sym not in us_seen:
            us_seen.add(sym)
            universe.append((sym, name, "US", "stock", None))
    meta["usListed"] = len(us_seen)
    print(f"미국 종목: {len(us_seen)}개")

    # ── ETF ──
    for code, name in ETF_KR:
        universe.append((code, name, "KR", "etf", None))
    for sym, name in ETF_US:
        universe.append((sym, name, "US", "etf", None))
    return universe, meta


def fetch_and_analyze(item):
    import FinanceDataReader as fdr
    ticker, name, country, asset, mcap = item
    start = (dt.date.today() - dt.timedelta(days=365 * LOOKBACK_YEARS)).isoformat()
    try:
        df = fdr.DataReader(ticker, start)
        return analyze_ticker(name, df, country, asset, mcap)
    except Exception:
        return None


def main():
    print("유니버스 구성 중...")
    universe, meta = build_universe()
    meta["scanned"] = len(universe)
    print(f"총 {len(universe)}종목 스캔 시작 (동시 {WORKERS}개)")

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(fetch_and_analyze, item) for item in universe]
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                results.append(r)
            if done % 200 == 0:
                print(f"  진행 {done}/{len(universe)} · 신호 {len(results)}개")

    results = finalize(results)
    payload = dict(
        asOf=dt.date.today().isoformat(),
        count=len(results),
        meta=meta,
        stocks=results,
    )
    with open("signals.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"완료: {len(results)}종목 신호 -> signals.json")


if __name__ == "__main__":
    main()
