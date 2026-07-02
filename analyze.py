# -*- coding: utf-8 -*-
"""
추세추종 매수 신호 스캐너
1) 한국(KRX) + 미국 종목/ETF 일봉 다운로드
2) 신호 탐지: 이평 근접(avg) / 박스 돌파(box) / 신고가 돌파(high)
3) 과거 전체 백테스트 -> 승률, 평균이익/손실, 손익비, 표본수
4) 종합 RS 계산 후 signals.json 저장
"""
import json
import sys
import datetime as dt
import numpy as np
import pandas as pd

UNIVERSE_KR = [
    ("005930", "삼성전자", "KR", "stock"),
    ("000660", "SK하이닉스", "KR", "stock"),
    ("373220", "LG에너지솔루션", "KR", "stock"),
    ("207940", "삼성바이오로직스", "KR", "stock"),
    ("005380", "현대차", "KR", "stock"),
    ("000270", "기아", "KR", "stock"),
    ("068270", "셀트리온", "KR", "stock"),
    ("035420", "NAVER", "KR", "stock"),
    ("105560", "KB금융", "KR", "stock"),
    ("055550", "신한지주", "KR", "stock"),
    ("012450", "한화에어로스페이스", "KR", "stock"),
    ("042660", "한화오션", "KR", "stock"),
    ("009540", "HD한국조선해양", "KR", "stock"),
    ("329180", "HD현대중공업", "KR", "stock"),
    ("267260", "HD현대일렉트릭", "KR", "stock"),
    ("006400", "삼성SDI", "KR", "stock"),
    ("051910", "LG화학", "KR", "stock"),
    ("035720", "카카오", "KR", "stock"),
    ("015760", "한국전력", "KR", "stock"),
    ("034020", "두산에너빌리티", "KR", "stock"),
    ("000150", "두산", "KR", "stock"),
    ("006800", "미래에셋증권", "KR", "stock"),
    ("086520", "에코프로", "KR", "stock"),
    ("247540", "에코프로비엠", "KR", "stock"),
    ("196170", "알테오젠", "KR", "stock"),
    ("217590", "TMC", "KR", "stock"),
    ("083450", "GST", "KR", "stock"),
    ("090460", "BH", "KR", "stock"),
    ("023160", "태광", "KR", "stock"),
    ("059090", "미코", "KR", "stock"),
    ("095340", "ISC", "KR", "stock"),
    ("240810", "원익IPS", "KR", "stock"),
    ("403870", "HPSP", "KR", "stock"),
    ("058470", "리노공업", "KR", "stock"),
    ("039030", "이오테크닉스", "KR", "stock"),
    ("098460", "고영", "KR", "stock"),
    ("036930", "주성엔지니어링", "KR", "stock"),
    ("131970", "두산테스나", "KR", "stock"),
    ("069500", "KODEX 200", "KR", "etf"),
    ("396500", "TIGER 차이나반도체FACTSET", "KR", "etf"),
    ("413600", "SOL 글로벌AI반도체탑픽액티브", "KR", "etf"),
    ("446770", "ACE 글로벌반도체TOP4 Plus", "KR", "etf"),
    ("305720", "KODEX 2차전지산업", "KR", "etf"),
    ("091160", "KODEX 반도체", "KR", "etf"),
]
UNIVERSE_US = [
    ("AAPL", "Apple", "US", "stock"),
    ("MSFT", "Microsoft", "US", "stock"),
    ("NVDA", "NVIDIA", "US", "stock"),
    ("GOOGL", "Alphabet", "US", "stock"),
    ("AMZN", "Amazon", "US", "stock"),
    ("META", "Meta", "US", "stock"),
    ("TSLA", "Tesla", "US", "stock"),
    ("AVGO", "Broadcom", "US", "stock"),
    ("AMD", "AMD", "US", "stock"),
    ("TSM", "TSMC", "US", "stock"),
    ("LLY", "Eli Lilly", "US", "stock"),
    ("JPM", "JPMorgan", "US", "stock"),
    ("V", "Visa", "US", "stock"),
    ("PLTR", "Palantir", "US", "stock"),
    ("VRT", "Vertiv", "US", "stock"),
    ("SPY", "SPDR S&P 500", "US", "etf"),
    ("QQQ", "Invesco QQQ", "US", "etf"),
    ("SMH", "VanEck Semiconductor", "US", "etf"),
    ("SOXL", "Direxion Semi Bull 3X", "US", "etf"),
    ("TLT", "iShares 20Y Treasury", "US", "etf"),
]

HOLD_PERIODS = [2, 3, 5, 10, 20, 60, 126, 252]
LOOKBACK_YEARS = 8
MA_PROXIMITY = 0.025
BOX_WINDOW = 40
BOX_TIGHTNESS = 0.12
MIN_SAMPLES = 8


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


def download_all(full=False):
    import FinanceDataReader as fdr
    start = (dt.date.today() - dt.timedelta(days=365 * LOOKBACK_YEARS)).isoformat()
    universe = list(UNIVERSE_KR) + list(UNIVERSE_US)

    if full:
        try:
            k200 = fdr.StockListing("KOSPI200")
            universe += [(r.Code, r.Name, "KR", "stock") for r in k200.itertuples()]
        except Exception as e:
            print("KOSPI200 목록 실패:", e)
        try:
            sp = fdr.StockListing("S&P500").head(100)
            universe += [(r.Symbol, r.Name, "US", "stock") for r in sp.itertuples()]
        except Exception as e:
            print("S&P 목록 실패:", e)

    seen, results = set(), []
    for ticker, name, country, asset in universe:
        if ticker in seen:
            continue
        seen.add(ticker)
        try:
            df = fdr.DataReader(ticker, start)
            mcap = None
            try:
                if country == "US":
                    import yfinance as yf
                    mcap = yf.Ticker(ticker).fast_info.get("market_cap")
            except Exception:
                pass
            r = analyze_ticker(name, df, country, asset, mcap)
            if r:
                results.append(r)
                print("  OK " + name + ": " + r["desc"])
        except Exception as e:
            print("  FAIL " + name + ": " + str(e))
    return results


def main():
    full = "--full" in sys.argv
    print("데이터 다운로드 및 신호 스캔 중...")
    results = finalize(download_all(full))
    payload = dict(
        asOf=dt.date.today().isoformat(),
        count=len(results),
        stocks=results,
    )
    with open("signals.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("완료: " + str(len(results)) + "종목 신호 -> signals.json")


if __name__ == "__main__":
    main()
