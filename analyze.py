# -*- coding: utf-8 -*-
"""
추세추종 매수 신호 스캐너 v2 — 전체 시장 스캔
1) 한국: KOSPI + KOSDAQ 전체 상장종목 (스팩/우선주 제외) + 주요 ETF
2) 미국: S&P500 전체 + 나스닥100 + 주요 ETF
3) 신호 탐지: 이평 근접(avg) / 박스 돌파(box) / 신고가 돌파(high)
4) 과거 전체 백테스트 -> 승률, 평균이익/손실, 손익비, 표본수
5) 종합 RS 계산 후 signals.json 저장
"""
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
MIN_MCAP_KRW = 100_000_000_000   # 한국 최소 시총 1000억 (동전주/초소형주 제외)
USDKRW = 1400                     # 시총 필터용 대략 환율
WORKERS = 8                       # 동시 다운로드 수

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
    if df["Close"].iloc[-30:].nunique() < 5:  # 거래정지 등 비정상 종목 제외
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
    results.sort(key=lambda r: -(r["mcap"] or 0))
    return results


def build_universe():
    """한국 전체(KOSPI+KOSDAQ) + 미국(S&P500+나스닥100) + ETF 목록 생성"""
    import FinanceDataReader as fdr
    universe = []  # (ticker, name, country, asset, mcap_usd)

    # ── 한국: 전체 상장종목 ──
    try:
        krx = fdr.StockListing("KRX")
        krx = krx[krx["Market"].isin(["KOSPI", "KOSDAQ"])]
        for r in krx.itertuples():
            name = str(r.Name)
            # 스팩, 우선주, 리츠 인프라 등 제외
            if "스팩" in name or name.endswith(("우", "우B", "우C", "1우", "2우B", "3우B")):
                continue
            mcap_krw = getattr(r, "Marcap", 0) or 0
            if mcap_krw < MIN_MCAP_KRW:
                continue
            universe.append((str(r.Code), name, "KR", "stock", mcap_krw / USDKRW))
        print(f"한국 종목: {sum(1 for u in universe if u[2]=='KR')}개")
    except Exception as e:
        print("KRX 목록 실패:", e)

    # ── 미국: S&P500 + 나스닥100 ──
    us_seen = set()
    for listing in ["S&P500", "NASDAQ100"]:
        try:
            df = fdr.StockListing(listing)
            sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            name_col = "Name" if "Name" in df.columns else df.columns[1]
            for r in df.itertuples():
                sym = str(getattr(r, sym_col))
                if sym in us_seen:
                    continue
                us_seen.add(sym)
                universe.append((sym, str(getattr(r, name_col)), "US", "stock", None))
        except Exception as e:
            print(listing, "목록 실패:", e)
    print(f"미국 종목: {len(us_seen)}개")

    # ── ETF ──
    for code, name in ETF_KR:
        universe.append((code, name, "KR", "etf", None))
    for sym, name in ETF_US:
        universe.append((sym, name, "US", "etf", None))
    return universe


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
    universe = build_universe()
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
        stocks=results,
    )
    with open("signals.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"완료: {len(results)}종목 신호 -> signals.json")


if __name__ == "__main__":
    main()
