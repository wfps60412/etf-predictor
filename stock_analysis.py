"""
stock_analysis.py  v5.0  ⚡ 加速版
三項優化：
  A. 外資+投信合併成一次 API call（省 15 次 request）
  B. 並行抓取（ThreadPoolExecutor，4 個 thread 同時跑）→ 速度提升 3~4x
  C. 增量更新：資料庫存在時只補抓新日期，從 20min → 1~2min
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
import urllib3
import ssl
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 設定區
# ==========================================
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoid2ZwczYwNDEyIiwiZW1haWwiOiJ3ZnBzNjA0MTJAZ21haWwuY29tIn0.QmlmQSgB5NG721tueiYOoYTIj-6nGfPw9sRTu0hDsjw"
FINMIND_API   = "https://api.finmindtrade.com/api/v4/data"
FINMIND_HDR   = {"Authorization": f"Bearer {FINMIND_TOKEN}", "User-Agent": "Mozilla/5.0"}

INDIVIDUAL_STOCKS = {
    # ── 原有 15 支 ────────────────────────────────────────────
    "1210":"大成",  "1216":"統一",  "1513":"中興電",
    "2323":"中環",  "2353":"宏碁",  "2374":"佳能",  "2376":"技嘉",
    "2409":"友達",  "3374":"精材",  "3481":"群創",  "4904":"遠傳",
    "6477":"安集",  "9914":"美利達","9921":"巨大",  "9933":"中鼎",
    # ── 新增 8 支（台灣各產業前二大）─────────────────────────
    "2330":"台積電","2303":"聯電",   "2454":"聯發科",
    "2882":"國泰金","2603":"長榮",   "2317":"鴻海",
    "2412":"中華電","1301":"台塑",
}
BENCHMARK_CODE = "0050"
BENCHMARK_NAME = "元大台灣50"
FILE_NAME      = "個股歷史分析資料庫.xlsx"
YEARS_BACK     = 4
DAYS_BACK      = 365 * YEARS_BACK
MAX_WORKERS    = 4       # 並行 thread 數（避免 rate limit，建議 3~5）
SLEEP_BASE     = 0.3     # 每次 API 後最短等待秒數（優化A大幅減少呼叫次數後可降低）


# ==========================================
# FinMind 通用查詢（附 retry）
# ==========================================
def finmind_get(dataset, stock_id, start_date, end_date=None, retries=2):
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    for attempt in range(retries + 1):
        try:
            r = requests.get(FINMIND_API, params=params, headers=FINMIND_HDR, timeout=30)
            r.raise_for_status()
            j = r.json()
            if j.get("status") == 200 and j.get("data"):
                return pd.DataFrame(j["data"])
            if j.get("status") != 200:
                print(f"      ℹ️  [{dataset}] status={j.get('status')}: {j.get('msg','no data')}")
            return pd.DataFrame()
        except requests.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries:
                wait = 10 * (attempt + 1)
                print(f"      ⚠️  Rate limit，等待 {wait}s 後重試...")
                time.sleep(wait)
            else:
                print(f"      ⚠️  [{dataset}] HTTP {e.response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"      ⚠️  [{dataset}] 錯誤: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# ==========================================
# 1. 宏觀指標
# ==========================================
def get_macro_data(start_date):
    print("🌐 正在抓取宏觀指標 (VIX, 匯率, S&P500, DXY)...")
    try:
        raw = yf.download(
            ["TWD=X", "^VIX", "^GSPC", "DX-Y.NYB"],
            start=start_date, progress=False
        )["Close"]
        if not raw.empty:
            raw = raw.rename(columns={
                "TWD=X": "台幣匯率", "^VIX": "VIX指數",
                "^GSPC": "SP500",   "DX-Y.NYB": "DXY",
            })
            raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")
            raw["SP500_ret"] = raw["SP500"].pct_change(1).round(6)
            raw["DXY_ret"]   = raw["DXY"].pct_change(1).round(6)
            print("   ✅ yfinance 宏觀指標成功（含 SP500_ret / DXY_ret）")
            return raw[["台幣匯率", "VIX指數", "SP500_ret", "DXY_ret"]]
    except Exception as e:
        print(f"   ⚠️  yfinance 失敗: {e}")
    return pd.DataFrame()


# ==========================================
# 2. 股價
# ==========================================
def _parse_finmind_price(df, label):
    if df.empty:
        return pd.DataFrame()
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        return pd.DataFrame()
    vol_col = next((c for c in df.columns if "volume" in c.lower()), None)
    for required in ["open", "max", "min", "close", "spread"]:
        if required not in df.columns:
            return pd.DataFrame()
    try:
        cols = [date_col, "open", "max", "min", "close", "spread"]
        out  = df[cols + ([vol_col] if vol_col else [])].copy()
        if vol_col:
            out.columns = ["日期", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交量"]
        else:
            out.columns = ["日期", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差"]
            out["成交量"] = 0
        out["日期"] = out["日期"].astype(str).str[:10]
        out["資料來源"] = label
        return out
    except Exception as e:
        print(f"      ⚠️  欄位解析失敗 ({label}): {e}")
        return pd.DataFrame()


def get_price_data(code, start_date):
    for suffix in [".TW", ".TWO"]:
        try:
            raw = yf.download(f"{code}{suffix}", start=start_date,
                              progress=False, multi_level_index=False,
                              auto_adjust=True)
            if not raw.empty:
                raw = raw.reset_index()
                raw["漲跌價差"] = raw["Close"].diff().round(2).fillna(0)
                df = raw.rename(columns={
                    "Date": "日期", "Open": "開盤價", "High": "最高價",
                    "Low": "最低價", "Close": "收盤價", "Volume": "成交量",
                })[["日期", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交量"]].copy()
                df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
                df["資料來源"] = f"yfinance_adj{suffix}"
                return df
        except Exception:
            continue
    raw = finmind_get("TaiwanStockPrice", code, start_date)
    out = _parse_finmind_price(raw, "FinMind_Raw")
    if not out.empty:
        print(f"      ⚠️  {code} 使用未還原股價")
        return out
    return pd.DataFrame()


# ==========================================
# 3. 大戶指標（ADL）
# ==========================================
def calc_adl(df_price, code, start_date):
    for suffix in [".TW", ".TWO"]:
        try:
            raw = yf.download(f"{code}{suffix}", start=start_date,
                              progress=False, multi_level_index=False,
                              auto_adjust=True)
            if not raw.empty and len(raw) > 5:
                cl, hi = raw["Close"].squeeze(), raw["High"].squeeze()
                lo, vol = raw["Low"].squeeze(), raw["Volume"].squeeze()
                mfm = ((cl - lo) - (hi - cl)) / (hi - lo).replace(0, 0.01)
                adl = (mfm * vol).cumsum().reset_index()
                adl.columns = ["日期", "大戶指標"]
                adl["日期"] = pd.to_datetime(adl["日期"]).dt.strftime("%Y-%m-%d")
                return df_price[["日期"]].merge(adl, on="日期", how="left")["大戶指標"]
        except Exception:
            continue
    hi, lo = df_price["最高價"], df_price["最低價"]
    mfm = ((df_price["收盤價"] - lo) - (hi - df_price["收盤價"])) / (hi - lo).replace(0, 0.01)
    return (mfm * df_price["成交量"]).cumsum()


# ==========================================
# 4+5. 外資 + 投信（合併成一次 API call）★優化A
# ==========================================
def get_institutional_data(code, start_date):
    """
    原本 get_foreign_data + get_trust_data 各呼叫一次 TaiwanStockInstitutionalInvestorsBuySell
    現在合併成一次 API call，同時解析外資連買天數和投信持股比
    節省 15 次 API request（每支省 1 次）
    """
    df_hold = finmind_get("TaiwanStockShareholding", code, start_date)
    foreign_hold = pd.DataFrame(columns=["日期", "外資持股比"])
    if not df_hold.empty:
        ratio_col = next(
            (c for c in df_hold.columns if "foreignInvestorHoldRatio" in c),
            next((c for c in df_hold.columns if "ratio" in c.lower()), None)
        )
        if ratio_col:
            foreign_hold = df_hold[["date", ratio_col]].copy()
            foreign_hold.columns = ["日期", "外資持股比"]
            foreign_hold["日期"] = foreign_hold["日期"].astype(str).str[:10]
    time.sleep(SLEEP_BASE)

    # ★ 一次 API call 同時取外資連買天數 + 投信持股比
    df_inst = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", code, start_date)
    foreign_streak  = pd.DataFrame(columns=["日期", "外資連買天數"])
    trust_data      = pd.DataFrame(columns=["日期", "投信持股比"])

    if not df_inst.empty and "name" in df_inst.columns:
        df_inst["日期"] = df_inst["date"].astype(str).str[:10]

        # 外資連買天數
        mask_f = df_inst["name"].apply(lambda n: "外資" in str(n))
        sub_f  = df_inst[mask_f].copy()
        if not sub_f.empty:
            sub_f["net"] = (pd.to_numeric(sub_f["buy"],  errors="coerce").fillna(0) -
                            pd.to_numeric(sub_f["sell"], errors="coerce").fillna(0))
            sub_f = sub_f.groupby("日期", as_index=False)["net"].sum().sort_values("日期")
            streaks, cur = [], 0
            for net in sub_f["net"]:
                if net > 0:   cur = cur + 1 if cur >= 0 else 1
                elif net < 0: cur = cur - 1 if cur <= 0 else -1
                streaks.append(cur)
            sub_f["外資連買天數"] = streaks
            foreign_streak = sub_f[["日期", "外資連買天數"]]

        # 投信持股比
        KEYWORDS = ["投信", "Investment Trust", "Trust"]
        mask_t = df_inst["name"].apply(lambda n: any(kw in str(n) for kw in KEYWORDS))
        sub_t  = df_inst[mask_t].copy()
        if not sub_t.empty:
            sub_t["net"] = (pd.to_numeric(sub_t["buy"],  errors="coerce").fillna(0) -
                            pd.to_numeric(sub_t["sell"], errors="coerce").fillna(0))
            sub_t = sub_t.groupby("日期", as_index=False)["net"].sum()
            sub_t["投信持股比"] = sub_t["net"].cumsum()
            trust_data = sub_t[["日期", "投信持股比"]]

    return foreign_hold, foreign_streak, trust_data


# ==========================================
# 6. 融資餘額變化率
# ==========================================
def get_margin_data(code, start_date):
    df = finmind_get("TaiwanStockMarginPurchaseShortSale", code, start_date)
    if df.empty:
        return pd.DataFrame(columns=["日期", "融資變化率"])
    df["日期"] = df["date"].astype(str).str[:10]
    col = next((c for c in df.columns if "MarginPurchase" in c and "Balance" in c), None)
    if col is None:
        return pd.DataFrame(columns=["日期", "融資變化率"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("日期").reset_index(drop=True)
    df["融資變化率"] = df[col].pct_change(5).round(4)
    return df[["日期", "融資變化率"]].dropna()


# ==========================================
# 7. 本益比 / 淨值比
# ==========================================
def get_valuation_data(code, start_date):
    df = finmind_get("TaiwanStockPER", code, start_date)
    if df.empty:
        return pd.DataFrame(columns=["日期", "PER", "PBR"])
    df["日期"] = df["date"].astype(str).str[:10]
    per_col = next((c for c in df.columns if c.lower() in ["per", "p/e"]), None)
    pbr_col = next((c for c in df.columns if c.lower() in ["pbr", "p/b"]), None)
    cols = {"日期": df["日期"]}
    cols["PER"] = pd.to_numeric(df[per_col], errors="coerce") if per_col else np.nan
    cols["PBR"] = pd.to_numeric(df[pbr_col], errors="coerce") if pbr_col else np.nan
    return pd.DataFrame(cols)


# ==========================================
# 8. 月營收
# ==========================================
def get_revenue_data(code, start_date):
    start_ext = (pd.Timestamp(start_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    df = finmind_get("TaiwanStockMonthRevenue", code, start_ext)
    if df.empty or "revenue" not in df.columns:
        return pd.DataFrame(columns=["日期", "營收年增率", "營收月增率"])
    df["date"]    = pd.to_datetime(df["date"])
    df            = df.sort_values("date").reset_index(drop=True)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    if "revenue_year" in df.columns and "revenue_month" in df.columns:
        df["_y"] = pd.to_numeric(df["revenue_year"],  errors="coerce")
        df["_m"] = pd.to_numeric(df["revenue_month"], errors="coerce")
        rev_map  = df.set_index(["_y", "_m"])["revenue"].to_dict()
        def yoy(row):
            prev = rev_map.get((row["_y"] - 1, row["_m"]), np.nan)
            return round((row["revenue"] / prev) - 1, 4) if prev and prev != 0 else np.nan
        df["營收年增率"] = df.apply(yoy, axis=1)
    else:
        df["營收年增率"] = (df["revenue"] / df["revenue"].shift(12) - 1).round(4)
    df["營收月增率"] = (df["revenue"] / df["revenue"].shift(1) - 1).round(4)
    out = df[df["date"] >= pd.Timestamp(start_date)][["date", "營收年增率", "營收月增率"]].copy()
    out.columns = ["日期", "營收年增率", "營收月增率"]
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    return out.dropna(subset=["營收年增率"])


# ==========================================
# 9. 大盤參考（0050）
# ==========================================
def get_benchmark_data(start_date):
    print(f"\n📊 抓取大盤參考 ({BENCHMARK_CODE} {BENCHMARK_NAME})...")
    df = get_price_data(BENCHMARK_CODE, start_date)
    if df is None or df.empty:
        print("   ❌ 大盤資料取得失敗，RSP 將全部補 NaN")
        return pd.DataFrame()
    if "資料來源" in df.columns:
        df = df.drop(columns=["資料來源"])
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    bench = df.set_index("日期")[["收盤價"]].copy()
    bench["大盤_收盤"] = bench["收盤價"]
    bench["大盤_MA20"] = bench["收盤價"].rolling(20).mean()
    result = bench[["大盤_收盤", "大盤_MA20"]]
    print(f"   ✅ 大盤資料: {len(result)} 筆  {result.index[0]} ~ {result.index[-1]}")
    return result


# ==========================================
# 10. 增量更新輔助：取得現有資料庫最後日期 ★優化C
# ==========================================
def get_existing_last_dates():
    """
    讀取現有資料庫各分頁的最後日期
    回傳 dict: {code: last_date_str} 或 {} (若無資料庫)
    """
    if not os.path.exists(FILE_NAME):
        return {}
    try:
        xl    = pd.ExcelFile(FILE_NAME)
        dates = {}
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(FILE_NAME, sheet_name=sheet, usecols=["日期"])
                if not df.empty:
                    dates[sheet.strip()] = str(df["日期"].max())[:10]
            except Exception:
                pass
        return dates
    except Exception:
        return {}


# ==========================================
# 11. 整合單一個股（可在 thread 中執行）
# ==========================================
def build_stock_df(code, name, start_date, benchmark_df=None):
    """
    回傳 df 或 None
    start_date 已在呼叫端根據增量模式調整
    """
    print(f"\n{'='*52}")
    print(f"  🚀 {code} {name}  (起始: {start_date})")
    print(f"{'='*52}")

    df = get_price_data(code, start_date)
    if df is None or df.empty:
        print(f"   ❌ {code} 無法取得股價，跳過。")
        return None
    source = df.pop("資料來源").iloc[0] if "資料來源" in df.columns else "?"
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    print(f"   ✅ 股價: {len(df)} 筆（{source}）")

    df["大戶指標"] = calc_adl(df, code, start_date)
    print("   ✅ 大戶指標: 完成")

    # RSP
    if benchmark_df is not None and not benchmark_df.empty:
        bench_series = benchmark_df[["大盤_收盤"]].reset_index()
        bench_series.columns = ["日期", "大盤_收盤"]
        df_rsp  = df[["日期", "收盤價"]].merge(bench_series, on="日期", how="left")
        rsp_vals = (df_rsp["收盤價"] / df_rsp["大盤_收盤"].replace(0, np.nan)).round(4)
        df["RSP"] = rsp_vals.values
        nz, miss = df["RSP"].notna().sum(), df["RSP"].isna().sum()
        print(f"   ✅ RSP: {nz} 筆有值" + (f"（{miss} 筆無對應）" if miss else ""))
    else:
        df["RSP"] = np.nan

    # ★優化A：外資+投信合併一次 API
    df_fhold, df_fstreak, df_trust = get_institutional_data(code, start_date)
    if not df_fhold.empty:
        df = df.merge(df_fhold, on="日期", how="left")
        df["外資持股比"] = df["外資持股比"].ffill().fillna(0)
        print(f"   ✅ 外資持股比: {(df['外資持股比']!=0).sum()} 筆非零")
    else:
        df["外資持股比"] = 0
    if not df_fstreak.empty:
        df = df.merge(df_fstreak, on="日期", how="left")
        df["外資連買天數"] = df["外資連買天數"].ffill().fillna(0)
        print("   ✅ 外資連買天數: 完成")
    else:
        df["外資連買天數"] = 0
    if not df_trust.empty:
        df = df.merge(df_trust, on="日期", how="left")
        df["投信持股比"] = df["投信持股比"].ffill().fillna(0)
        print(f"   ✅ 投信持股比: {(df['投信持股比']!=0).sum()} 筆非零")
    else:
        df["投信持股比"] = 0
    time.sleep(SLEEP_BASE)

    df_margin = get_margin_data(code, start_date)
    if not df_margin.empty:
        df = df.merge(df_margin, on="日期", how="left")
        df["融資變化率"] = df["融資變化率"].ffill().fillna(0)
        print(f"   ✅ 融資變化率: {(df['融資變化率']!=0).sum()} 筆非零")
    else:
        df["融資變化率"] = 0
    time.sleep(SLEEP_BASE)

    df_val = get_valuation_data(code, start_date)
    if not df_val.empty:
        df = df.merge(df_val, on="日期", how="left")
        df["PER"] = df["PER"].ffill().fillna(0)
        df["PBR"] = df["PBR"].ffill().fillna(0)
        print(f"   ✅ PER/PBR: {(df['PER']!=0).sum()} 筆非零")
    else:
        df["PER"] = 0; df["PBR"] = 0
    time.sleep(SLEEP_BASE)

    df_rev = get_revenue_data(code, start_date)
    if not df_rev.empty:
        df_rev["月份"] = df_rev["日期"].str[:7]
        for col in ["營收年增率", "營收月增率"]:
            monthly = df_rev.drop_duplicates("月份").set_index("月份")[col]
            df[col]  = df["日期"].str[:7].map(monthly).ffill().fillna(0)
        print(f"   ✅ 營收 YoY/MoM: {(df['營收年增率']!=0).sum()} 筆非零")
    else:
        df["營收年增率"] = 0; df["營收月增率"] = 0

    df = df[df["收盤價"] > 0].sort_values("日期").reset_index(drop=True)

    # 衍生特徵（不需要 VIX/匯率的部分）
    # StockType: 0=防禦型, 1=混合型, 2=景氣循環型（依業務特性，非 Beta 值）
    DEFENSIVE = {"4904","1216","1210","9933","2882","2412"}           # 電信、食品、工程、金融
    MIXED     = {"2353","1513"}                                        # 宏碁（混合PC）、中興電
    CYCLICAL  = {"9914","9921","2376","3374","2374","2409","3481","2323","6477",
                 "2330","2303","2454","2603","2317","1301"}             # 新增：半導體/IC/航運/鴻海/石化
    if code in DEFENSIVE:   df["StockType"] = 0
    elif code in CYCLICAL:  df["StockType"] = 2
    else:                   df["StockType"] = 1  # 混合型
    type_name = ["防禦型","混合型","景氣循環型"][int(df["StockType"].iloc[0])]
    print(f"   ✅ 股票類型: {type_name}")

    if benchmark_df is not None and not benchmark_df.empty:
        bench_ret = benchmark_df[["大盤_收盤"]].reset_index()
        bench_ret.columns = ["日期","大盤_收盤"]
        bench_ret["大盤_ret"] = bench_ret["大盤_收盤"].pct_change()
        df_b = df[["日期","收盤價"]].copy()
        df_b["stock_ret"] = df_b["收盤價"].pct_change()
        df_b = df_b.merge(bench_ret[["日期","大盤_ret"]], on="日期", how="left")
        cov60 = df_b["stock_ret"].rolling(60).cov(df_b["大盤_ret"])
        var60 = df_b["大盤_ret"].rolling(60).var().replace(0, np.nan)
        df["rolling_beta_60d"] = (cov60 / var60).round(4).ffill().fillna(1.0)
        df["beta_regime"]      = df["rolling_beta_60d"].rank(pct=True).round(4)
        print("   ✅ 動態Beta: 完成")
    else:
        df["rolling_beta_60d"] = 1.0
        df["beta_regime"]      = 0.5

    df["rev_acc3m"]   = df["營收年增率"].rolling(3).mean().round(4).fillna(0)
    df["rev_accel"]   = (df["營收年增率"] - df["營收年增率"].shift(3)).round(4).fillna(0)
    df["rev_mom_acc"] = df["營收月增率"].rolling(3).mean().round(4).fillna(0)
    print("   ✅ 營收動能: 完成")

    per_ma60  = df["PER"].replace(0, np.nan).rolling(60).mean()
    pbr_ma60  = df["PBR"].replace(0, np.nan).rolling(60).mean()
    per_std60 = df["PER"].replace(0, np.nan).rolling(60).std().replace(0, 1e-9)
    pbr_std60 = df["PBR"].replace(0, np.nan).rolling(60).std().replace(0, 1e-9)
    df["per_zscore"] = ((df["PER"] - per_ma60) / per_std60).round(4).fillna(0)
    df["pbr_zscore"] = ((df["PBR"] - pbr_ma60) / pbr_std60).round(4).fillna(0)
    df["per_trend"]  = (df["PER"] - df["PER"].shift(20)).round(4).fillna(0)
    print("   ✅ 估值動能: 完成")

    # 財務報表三表（季度指標）
    df = merge_financials(df, code, start_date)

    return df


# ==========================================
# 12. 大盤參考分頁
# ==========================================

# ==========================================
# 財務報表三表（季度指標）
# ==========================================
def get_financial_statements(code, start_date):
    """
    從 FinMind 抓三張財報，衍生 8 個季度指標：
      gross_margin, gross_margin_qoq, op_margin, eps_qoq,
      inventory_days, receivable_days,
      operating_cf_ratio, capex_ratio
    回傳 DataFrame（index=日期字串, columns=指標名）
    季度資料以 ffill 補到日線
    """
    print(f"   抓取財務報表（損益/資產負債/現金流）...")

    def fm_get(dataset, extra_start=None):
        s = extra_start or start_date
        try:
            r = requests.get(FINMIND_API, params={
                "dataset":    dataset,
                "data_id":    code,
                "start_date": s,
            }, headers=FINMIND_HDR, timeout=20)
            r.raise_for_status()
            j = r.json()
            if j.get("status") == 200 and j.get("data"):
                return pd.DataFrame(j["data"])
        except Exception as e:
            print(f"      ⚠️  {dataset}: {e}")
        return pd.DataFrame()

    # 往前多抓 2 季確保有前期可比較
    back_start = (pd.Timestamp(start_date) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    df_is  = fm_get("TaiwanStockFinancialStatements", back_start)   # 損益表
    time.sleep(random.uniform(0.4, 0.8))
    df_bs  = fm_get("TaiwanStockBalanceSheet",        back_start)   # 資產負債表
    time.sleep(random.uniform(0.4, 0.8))
    df_cf  = fm_get("TaiwanStockCashFlowsStatement",  back_start)   # 現金流量表
    time.sleep(random.uniform(0.4, 0.8))

    def pivot(df, types):
        """把 long format 轉成 wide，每個 date × type"""
        if df.empty:
            return pd.DataFrame()
        df = df[df["type"].isin(types)].copy()
        df["date"]  = df["date"].astype(str).str[:10]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.pivot_table(index="date", columns="type",
                               values="value", aggfunc="first")

    # 損益表欄位
    IS_TYPES = ["Revenue", "GrossProfit", "CostOfGoodsSold", "OperatingIncome", "EPS"]
    is_wide  = pivot(df_is, IS_TYPES)

    # 資產負債表欄位
    BS_TYPES = ["Inventories", "AccountsReceivableNet", "TotalAssets"]
    bs_wide  = pivot(df_bs, BS_TYPES)

    # 現金流量表欄位
    CF_TYPES = ["CashFlowsFromOperatingActivities", "CashProvidedByInvestingActivities"]
    cf_wide  = pivot(df_cf, CF_TYPES)

    # 合併三表
    merged = pd.concat([is_wide, bs_wide, cf_wide], axis=1)
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_index()
    eps_ok = ("EPS" in merged.columns)

    # 衍生指標計算
    rev  = merged.get("Revenue",         pd.Series(dtype=float))
    cogs = merged.get("CostOfGoodsSold", pd.Series(dtype=float))
    gp   = merged.get("GrossProfit",     pd.Series(dtype=float))
    oi   = merged.get("OperatingIncome", pd.Series(dtype=float))
    inv  = merged.get("Inventories",     pd.Series(dtype=float))
    ar   = merged.get("AccountsReceivableNet", pd.Series(dtype=float))
    ocf  = merged.get("CashFlowsFromOperatingActivities",  pd.Series(dtype=float))
    icf  = merged.get("CashProvidedByInvestingActivities", pd.Series(dtype=float))

    result = pd.DataFrame(index=merged.index)
    rev_safe = rev.replace(0, np.nan)

    result["gross_margin"]       = (gp / rev_safe).round(4)
    result["op_margin"]          = (oi / rev_safe).round(4)
    result["gross_margin_qoq"]   = result["gross_margin"].diff(1).round(4)
    result["eps_qoq"]            = (merged["EPS"].diff(1).round(4)
                                    if eps_ok else pd.Series(0.0, index=merged.index))
    # 存貨天數：Inventories / (COGS/90)
    cogs_safe = cogs.replace(0, np.nan)
    result["inventory_days"]     = (inv / (cogs_safe / 90)).round(1)
    # 應收天數：AR / (Revenue/90)
    result["receivable_days"]    = (ar / (rev_safe / 90)).round(1)
    # 營業現金流率
    result["operating_cf_ratio"] = (ocf / rev_safe).round(4)
    # 資本支出強度（取絕對值，投資活動通常為負）
    result["capex_ratio"]        = (icf.abs() / rev_safe).round(4)

    # 只保留 start_date 之後的資料
    result = result[result.index >= start_date].copy()
    return result


def merge_financials(df_stock, code, start_date):
    """
    將季度財務指標 ffill 對齊到日線 DataFrame
    """
    fin_cols = ["gross_margin","gross_margin_qoq","op_margin","eps_qoq",
                "inventory_days","receivable_days","operating_cf_ratio","capex_ratio"]

    fin = get_financial_statements(code, start_date)
    if fin.empty:
        print(f"      ⚠️  財務報表無資料，補零")
        for c in fin_cols:
            df_stock[c] = 0.0
        return df_stock

    # 對齊到日線：季度資料在當季結束日有值，其餘 ffill
    df_stock = df_stock.set_index("日期")
    for col in fin_cols:
        if col in fin.columns:
            df_stock[col] = fin[col].reindex(df_stock.index).ffill().fillna(0)
        else:
            df_stock[col] = 0.0

    nz = sum((df_stock[c] != 0).sum() for c in fin_cols)
    print(f"   ✅ 財務指標 (8個): 非零值共 {nz} 筆")
    return df_stock.reset_index()

def build_benchmark_df(start_date, macro_df):
    print(f"\n{'='*52}")
    print(f"  📈 {BENCHMARK_CODE} {BENCHMARK_NAME}（大盤參考，不納入個股訓練）")
    print(f"{'='*52}")
    df = get_price_data(BENCHMARK_CODE, start_date)
    if df is None or df.empty:
        return None
    if "資料來源" in df.columns:
        df = df.drop(columns=["資料來源"])
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    df["大戶指標"] = calc_adl(df, BENCHMARK_CODE, start_date)
    for col in ["外資持股比","外資連買天數","投信持股比","融資變化率","PER","PBR",
                "營收年增率","營收月增率","RSP",
                "StockType","rolling_beta_60d","beta_regime","market_stress",
                "rev_acc3m","rev_accel","rev_mom_acc",
                "per_zscore","pbr_zscore","per_trend",
                "beta_sp500_static",
                "gross_margin","gross_margin_qoq","op_margin","eps_qoq",
                "inventory_days","receivable_days","operating_cf_ratio","capex_ratio"]:
        df[col] = 0
    if not macro_df.empty:
        df = df.set_index("日期").join(macro_df, how="left").reset_index()
        for c in ["台幣匯率", "VIX指數", "SP500_ret", "DXY_ret"]:
            if c in df.columns:
                df[c] = df[c].ffill()
    final_cols = FINAL_COLS + ["RSP"]
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0
    return df[final_cols].fillna(0)


# ==========================================
# 欄位順序
# ==========================================
FINAL_COLS = [
    "日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量",
    "大戶指標","外資持股比","外資連買天數","投信持股比","融資變化率",
    "PER","PBR","營收年增率","營收月增率","台幣匯率","VIX指數",
    "SP500_ret","DXY_ret",
    "StockType","rolling_beta_60d","beta_regime","market_stress",
    "beta_sp500_static",
    "rev_acc3m","rev_accel","rev_mom_acc",
    "per_zscore","pbr_zscore","per_trend",
    "gross_margin","gross_margin_qoq","op_margin","eps_qoq",
    "inventory_days","receivable_days","operating_cf_ratio","capex_ratio",
]

INDUSTRY_SIGNALS_FILE = "industry_signals.xlsx"


# ==========================================
# 主流程（並行 + 增量）
# ==========================================
def main():
    today      = datetime.now()
    full_start = (today - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    # ★優化C：增量模式檢查
    existing_dates = get_existing_last_dates()
    is_incremental = bool(existing_dates)

    if is_incremental:
        # 全部股票的最早「上次最後日期」決定補抓起點（取最保守的昨天）
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        min_last  = min(existing_dates.values()) if existing_dates else full_start
        # 往前多抓 30 天確保滾動指標（MA20 等）計算正確
        incr_start = (pd.Timestamp(min_last) - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
        print(f"📅 增量更新模式（上次最早: {min_last}，補抓起點: {incr_start}）\n")
    else:
        incr_start = full_start
        print(f"📅 全量建置模式，資料區間：{full_start} ~ {today.strftime('%Y-%m-%d')}（{YEARS_BACK} 年）\n")
        # 全量模式才刪除舊檔
        if os.path.exists(FILE_NAME):
            try:
                os.remove(FILE_NAME)
                print(f"🗑️  已刪除舊資料庫\n")
            except OSError:
                print("⚠️  無法刪除舊資料庫，請先關閉 Excel。"); return

    macro_df     = get_macro_data(incr_start)
    benchmark_df = get_benchmark_data(incr_start)

    # ── 讀入產業訊號（industry_scraper.py 的輸出）────────────────
    industry_signals = {}   # code -> DataFrame(月份, 各訊號欄)
    if os.path.exists(INDUSTRY_SIGNALS_FILE):
        try:
            df_sig = pd.read_excel(INDUSTRY_SIGNALS_FILE, sheet_name="個股月頻對照")
            df_sig["month"] = df_sig["month"].astype(str).str[:7]
            for code, grp in df_sig.groupby("code"):
                industry_signals[str(code)] = grp.set_index("month")[[
                    "sentiment_score","sentiment_3m_avg","sentiment_trend",
                    "pos_count","neg_count","article_count"
                ]]
            print(f"📡 產業訊號已載入：{len(industry_signals)} 支股票")
        except Exception as e:
            print(f"⚠️  產業訊號讀取失敗（跳過）: {e}")
    else:
        print(f"ℹ️  未找到 {INDUSTRY_SIGNALS_FILE}，產業訊號補零（先執行 industry_scraper.py）")

    # ★優化B：並行抓取各股
    print(f"\n🚀 並行抓取（{MAX_WORKERS} threads）...")
    results = {}

    def fetch_one(args):
        code, name = args
        try:
            return code, build_stock_df(code, name, incr_start, benchmark_df=benchmark_df)
        except Exception as e:
            print(f"   ❌ {code} 例外: {e}")
            return code, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, item): item[0]
                   for item in INDIVIDUAL_STOCKS.items()}
        for fut in as_completed(futures):
            code, df_new = fut.result()
            results[code] = df_new

    # 合併到舊資料（增量模式）或直接寫（全量模式）
    with pd.ExcelWriter(FILE_NAME, engine="openpyxl") as writer:

        # 0050 大盤分頁
        df_bench = build_benchmark_df(incr_start, macro_df)
        if df_bench is not None:
            if is_incremental and BENCHMARK_CODE in existing_dates:
                try:
                    old = pd.read_excel(FILE_NAME, sheet_name=BENCHMARK_CODE)
                    old["日期"] = old["日期"].astype(str).str[:10]
                    df_bench = pd.concat([
                        old[old["日期"] < df_bench["日期"].min()],
                        df_bench
                    ]).drop_duplicates("日期").sort_values("日期").reset_index(drop=True)
                except Exception:
                    pass
            df_bench.to_excel(writer, sheet_name=BENCHMARK_CODE, index=False)
            print(f"   💾 {BENCHMARK_CODE} 已寫入（{len(df_bench)} 筆）")

        # 個股
        for code, name in INDIVIDUAL_STOCKS.items():
            df_new = results.get(code)
            if df_new is None:
                continue

            # macro join + market_stress
            if not macro_df.empty:
                df_new = df_new.set_index("日期").join(macro_df, how="left").reset_index()
                for c in ["台幣匯率", "VIX指數", "SP500_ret", "DXY_ret"]:
                    if c in df_new.columns:
                        df_new[c] = df_new[c].ffill()

            # ── 產業訊號 merge（月頻 ffill 到日線）──────────────
            sig_cols = ["sentiment_score","sentiment_3m_avg","sentiment_trend",
                        "pos_count","neg_count","article_count"]
            if code in industry_signals:
                sig_df = industry_signals[code].reset_index()
                sig_df.columns = ["month"] + sig_cols
                df_new["month"] = df_new["日期"].astype(str).str[:7]
                df_new = df_new.merge(sig_df, on="month", how="left")
                df_new = df_new.drop(columns=["month"])
                for c in sig_cols:
                    df_new[c] = df_new[c].ffill().fillna(0)
                print(f"   ✅ 產業訊號: {(df_new['sentiment_score']!=0).sum()} 筆非零")
            else:
                for c in sig_cols:
                    df_new[c] = 0

            if "VIX指數" in df_new.columns and "台幣匯率" in df_new.columns:
                vix_ma = df_new["VIX指數"].rolling(20).mean()
                fx_ma  = df_new["台幣匯率"].rolling(20).mean()
                df_new["market_stress"] = (
                    ((df_new["VIX指數"] - vix_ma) / vix_ma.replace(0, 1e-9)) * 0.5 +
                    ((df_new["台幣匯率"] - fx_ma)  / fx_ma.replace(0, 1e-9)) * 0.5
                ).round(4).fillna(0)
            else:
                df_new["market_stress"] = 0

            # ── Beta_SP500：靜態全期平均值（每支股票固定，不隨時間變動）
            # 靜態版比動態版（60日滾動）更適合 12M 預測，因為長期特性比短期波動更有信號
            BETA_SP500_STATIC = {'3481': -0.0882, '2323': -0.068, '2409': -0.0529, '4904': -0.0467, '1216': -0.0196, '1210': -0.0015, '9933': -0.0008, '2353': 0.0117, '1513': 0.0337, '9914': 0.038, '2374': 0.0635, '9921': 0.076, '6477': 0.1007, '3374': 0.1259, '2376': 0.2046, '2330': 0.18, '2303': 0.09, '2454': 0.15, '2882': 0.03, '2603': 0.06, '2317': 0.11, '2412': -0.03, '1301': 0.05}
            df_new["beta_sp500_static"] = BETA_SP500_STATIC.get(code, 0.0)

            all_cols = FINAL_COLS + (["RSP"] if "RSP" in df_new.columns else [])
            for c in all_cols:
                if c not in df_new.columns:
                    df_new[c] = 0
            df_new = df_new[all_cols].fillna(0)

            # ★優化C：增量合併
            if is_incremental and code in existing_dates:
                try:
                    old = pd.read_excel(FILE_NAME, sheet_name=code)
                    old["日期"] = old["日期"].astype(str).str[:10]
                    cutoff = df_new["日期"].min()
                    df_new = pd.concat([
                        old[old["日期"] < cutoff],
                        df_new
                    ]).drop_duplicates("日期").sort_values("日期").reset_index(drop=True)
                    for c in all_cols:
                        if c not in df_new.columns:
                            df_new[c] = 0
                    df_new = df_new[all_cols].fillna(0)
                except Exception as e:
                    print(f"   ⚠️  {code} 增量合併失敗（使用新資料）: {e}")

            df_new.to_excel(writer, sheet_name=code, index=False)
            print(f"   💾 {code} 已寫入（共 {len(df_new)} 筆）")

    mode = "增量更新" if is_incremental else f"{YEARS_BACK}年期全量"
    print(f"\n✨ 【{mode}資料庫】構建完成！檔案：{FILE_NAME}")
    print(f"   個股分頁：{len(INDIVIDUAL_STOCKS)} 支")
    print(f"   大盤參考：{BENCHMARK_CODE}")


if __name__ == "__main__":
    main()
