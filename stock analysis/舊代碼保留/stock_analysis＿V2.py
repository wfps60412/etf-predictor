"""
stock_analysis.py  v3.0
- 資料改為4年（1460天），確保12M回測有足夠樣本
- 新增欄位：外資連續買超天數、融資餘額變化率、月增率(MoM)、PER、PBR
- 0050 改為純大盤參考分頁，不參與個股模型訓練
- 使用 TaiwanStockPriceAdj（還原股價）消除除權假訊號
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

# 個股列表（0050 移出，另外單獨處理為大盤參考）
INDIVIDUAL_STOCKS = {
    "1210":"大成",  "1216":"統一",  "1513":"中興電",
    "2323":"中環",  "2353":"宏碁",  "2374":"佳能",  "2376":"技嘉",
    "2409":"友達",  "3374":"精材",  "3481":"群創",  "4904":"遠傳",
    "6477":"安集",  "9914":"美利達","9921":"巨大",  "9933":"中鼎",
}
BENCHMARK_CODE = "0050"
BENCHMARK_NAME = "元大台灣50"

FILE_NAME  = "個股歷史分析資料庫.xlsx"
# 改為4年，12M回測的test樣本從~42筆提升至~120筆
YEARS_BACK = 4
DAYS_BACK  = 365 * YEARS_BACK


# ==========================================
# FinMind 通用查詢
# ==========================================
def finmind_get(dataset, stock_id, start_date, end_date=None):
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    try:
        r = requests.get(FINMIND_API, params=params, headers=FINMIND_HDR, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == 200 and j.get("data"):
            return pd.DataFrame(j["data"])
        print(f"      ℹ️  [{dataset}] status={j.get('status')}: {j.get('msg','no data')}")
    except requests.HTTPError as e:
        print(f"      ⚠️  [{dataset}] HTTP {e.response.status_code}")
    except Exception as e:
        print(f"      ⚠️  [{dataset}] 錯誤: {e}")
    return pd.DataFrame()


# ==========================================
# 1. 宏觀指標（VIX + 台幣匯率）
# ==========================================
def get_macro_data(start_date):
    print("🌐 正在抓取宏觀指標 (VIX, 匯率, S&P500, DXY)...")
    try:
        # 同時抓取 4 個指標：台幣匯率、VIX、S&P500、美元指數
        raw = yf.download(
            ["TWD=X", "^VIX", "^GSPC", "DX-Y.NYB"],
            start=start_date, progress=False
        )["Close"]
        if not raw.empty:
            raw = raw.rename(columns={
                "TWD=X":    "台幣匯率",
                "^VIX":     "VIX指數",
                "^GSPC":    "SP500",
                "DX-Y.NYB": "DXY",
            })
            raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")

            # 計算日漲跌幅（前一日→當日），作為領先指標
            # 用 shift(1) 確保當天預測只用到前一日的美市收盤
            raw["SP500_ret"] = raw["SP500"].pct_change(1).round(6)
            raw["DXY_ret"]   = raw["DXY"].pct_change(1).round(6)

            # 保留原始收盤供參考，主要特徵用日報酬率
            print(f"   ✅ yfinance 宏觀指標成功（含 SP500_ret / DXY_ret）")
            return raw[["台幣匯率", "VIX指數", "SP500_ret", "DXY_ret"]]
    except Exception as e:
        print(f"   ⚠️  yfinance 失敗: {e}")
    return pd.DataFrame()


# ==========================================
# 2. 股價（優先使用還原股價，消除除權假訊號）
# ==========================================
def _parse_finmind_price(df, label):
    """
    從 FinMind 股價 DataFrame 取出標準欄位，容忍欄位大小寫差異。
    Trading_Volume / trading_volume / TradeVolume 都能對應。
    """
    if df.empty:
        return pd.DataFrame()
    # 日期
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        return pd.DataFrame()
    # 成交量：找第一個含 "volume" 的欄位（大小寫不限）
    vol_col = next((c for c in df.columns if "volume" in c.lower()), None)
    # 其他固定欄位（FinMind 固定小寫）
    for required in ["open", "max", "min", "close", "spread"]:
        if required not in df.columns:
            return pd.DataFrame()
    try:
        cols = [date_col, "open", "max", "min", "close", "spread"]
        out = df[cols + ([vol_col] if vol_col else [])].copy()
        if vol_col:
            out.columns = ["日期", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交量"]
        else:
            out.columns = ["日期", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差"]
            out["成交量"] = 0
        out["日期"] = out["日期"].astype(str).str[:10]
        out["資料來源"] = label
        return out
    except Exception as e:
        print(f"      ⚠️  欄位解析失敗 ({label}): {e} | 欄位={list(df.columns)}")
        return pd.DataFrame()


def get_price_data(code, start_date):
    """
    優先順序：
    1. yfinance（.TW / .TWO）— auto_adjust=True 預設回傳除權還原後的收盤價，免費
       TaiwanStockPriceAdj（還原版）需付費帳號（backer/sponsor），不嘗試
    2. FinMind TaiwanStockPrice — 原始未還原股價，作為 yfinance 失敗時的備援
    """
    # ── 優先：yfinance 還原價（auto_adjust 預設 True）──────────────
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

    # ── 備援：FinMind 原始股價（未還原）────────────────────────────
    raw = finmind_get("TaiwanStockPrice", code, start_date)
    out = _parse_finmind_price(raw, "FinMind_Raw")
    if not out.empty:
        print(f"      ⚠️  {code} 使用未還原股價（yfinance 失敗），除權日附近資料可能失真")
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
    # 本地模擬（備援）
    hi, lo = df_price["最高價"], df_price["最低價"]
    mfm = ((df_price["收盤價"] - lo) - (hi - df_price["收盤價"])) / (hi - lo).replace(0, 0.01)
    return (mfm * df_price["成交量"]).cumsum()


# ==========================================
# 4. 外資持股比 + 連續買超天數
# ==========================================
def get_foreign_data(code, start_date):
    """
    回傳：外資持股比（%）、外資連續買超天數
    """
    # 持股比
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

    # 連續買超天數（用法人買賣超計算）
    df_inst = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", code, start_date)
    foreign_streak = pd.DataFrame(columns=["日期", "外資連買天數"])
    if not df_inst.empty and "name" in df_inst.columns:
        mask = df_inst["name"].apply(lambda n: "外資" in str(n))
        sub = df_inst[mask].copy()
        if not sub.empty:
            sub["日期"] = sub["date"].astype(str).str[:10]
            sub["net"] = (pd.to_numeric(sub["buy"],  errors="coerce").fillna(0) -
                          pd.to_numeric(sub["sell"], errors="coerce").fillna(0))
            sub = sub.groupby("日期", as_index=False)["net"].sum().sort_values("日期")
            # 連續買超：net > 0 記為 +1，net < 0 重置為 0，net = 0 延續
            streaks = []
            cur = 0
            for net in sub["net"]:
                if net > 0:
                    cur = cur + 1 if cur >= 0 else 1
                elif net < 0:
                    cur = cur - 1 if cur <= 0 else -1
                streaks.append(cur)
            sub["外資連買天數"] = streaks
            foreign_streak = sub[["日期", "外資連買天數"]]

    return foreign_hold, foreign_streak


# ==========================================
# 5. 投信買賣超累計
# ==========================================
def get_trust_data(code, start_date):
    df = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", code, start_date)
    if df.empty or "name" not in df.columns:
        return pd.DataFrame(columns=["日期", "投信持股比"])
    KEYWORDS = ["投信", "Investment Trust", "Trust"]
    mask = df["name"].apply(lambda n: any(kw in str(n) for kw in KEYWORDS))
    sub = df[mask].copy()
    if sub.empty:
        actual = df["name"].unique().tolist()
        print(f"      ℹ️  投信關鍵字無匹配，實際 name 值: {actual}")
        return pd.DataFrame(columns=["日期", "投信持股比"])
    sub["日期"] = sub["date"].astype(str).str[:10]
    sub["net"] = (pd.to_numeric(sub["buy"],  errors="coerce").fillna(0) -
                  pd.to_numeric(sub["sell"], errors="coerce").fillna(0))
    sub = sub.groupby("日期", as_index=False)["net"].sum()
    sub["投信持股比"] = sub["net"].cumsum()
    return sub[["日期", "投信持股比"]]


# ==========================================
# 6. 融資餘額變化率（散戶情緒反向指標）
# ==========================================
def get_margin_data(code, start_date):
    df = finmind_get("TaiwanStockMarginPurchaseShortSale", code, start_date)
    if df.empty:
        return pd.DataFrame(columns=["日期", "融資變化率"])
    df["日期"] = df["date"].astype(str).str[:10]
    # MarginPurchaseTodayBalance = 融資今日餘額
    col = next((c for c in df.columns if "MarginPurchase" in c and "Balance" in c), None)
    if col is None:
        return pd.DataFrame(columns=["日期", "融資變化率"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("日期").reset_index(drop=True)
    # 5日變化率，捕捉融資動能
    df["融資變化率"] = df[col].pct_change(5).round(4)
    return df[["日期", "融資變化率"]].dropna()


# ==========================================
# 7. 本益比（PER）/ 淨值比（PBR）
# ==========================================
def get_valuation_data(code, start_date):
    df = finmind_get("TaiwanStockPER", code, start_date)
    if df.empty:
        return pd.DataFrame(columns=["日期", "PER", "PBR"])
    df["日期"] = df["date"].astype(str).str[:10]
    per_col = next((c for c in df.columns if c.lower() in ["per", "p/e"]), None)
    pbr_col = next((c for c in df.columns if c.lower() in ["pbr", "p/b"]), None)
    cols = {"日期": df["日期"]}
    if per_col:
        cols["PER"] = pd.to_numeric(df[per_col], errors="coerce")
    else:
        cols["PER"] = np.nan
    if pbr_col:
        cols["PBR"] = pd.to_numeric(df[pbr_col], errors="coerce")
    else:
        cols["PBR"] = np.nan
    return pd.DataFrame(cols)


# ==========================================
# 8. 月營收（YoY + MoM）
# ==========================================
def get_revenue_data(code, start_date):
    # 多抓1年以計算 YoY
    start_ext = (pd.Timestamp(start_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    df = finmind_get("TaiwanStockMonthRevenue", code, start_ext)
    if df.empty or "revenue" not in df.columns:
        return pd.DataFrame(columns=["日期", "營收年增率", "營收月增率"])

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # YoY：用 revenue_year/month 精確對應去年同月
    if "revenue_year" in df.columns and "revenue_month" in df.columns:
        df["_y"] = pd.to_numeric(df["revenue_year"],  errors="coerce")
        df["_m"] = pd.to_numeric(df["revenue_month"], errors="coerce")
        rev_map = df.set_index(["_y", "_m"])["revenue"].to_dict()
        def yoy(row):
            prev = rev_map.get((row["_y"] - 1, row["_m"]), np.nan)
            return round((row["revenue"] / prev) - 1, 4) if prev and prev != 0 else np.nan
        df["營收年增率"] = df.apply(yoy, axis=1)
    else:
        df["營收年增率"] = (df["revenue"] / df["revenue"].shift(12) - 1).round(4)

    # MoM：本月 vs 上月
    df["營收月增率"] = (df["revenue"] / df["revenue"].shift(1) - 1).round(4)

    out = df[df["date"] >= pd.Timestamp(start_date)][["date", "營收年增率", "營收月增率"]].copy()
    out.columns = ["日期", "營收年增率", "營收月增率"]
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    return out.dropna(subset=["營收年增率"])


# ==========================================
# 9. 0050 大盤參考指標
# ==========================================
def get_benchmark_data(start_date):
    """
    抓取 0050，計算大盤收盤序列供個股 RSP 使用。
    回傳：以「日期字串」為 index 的 DataFrame，含 大盤_收盤、大盤_MA20。
    """
    print(f"\n📊 抓取大盤參考 ({BENCHMARK_CODE} {BENCHMARK_NAME})...")
    df = get_price_data(BENCHMARK_CODE, start_date)
    if df is None or df.empty:
        print(f"   ❌ 大盤資料取得失敗，RSP 將全部補 NaN")
        return pd.DataFrame()

    source = df["資料來源"].iloc[0] if "資料來源" in df.columns else "?"
    # 清除資料來源欄，避免後續 set_index 保留多餘欄位
    if "資料來源" in df.columns:
        df = df.drop(columns=["資料來源"])

    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)

    # 建立以日期字串為 index 的序列
    bench = df.set_index("日期")[["收盤價"]].copy()
    bench["大盤_收盤"] = bench["收盤價"]
    bench["大盤_MA20"] = bench["收盤價"].rolling(20).mean()
    result = bench[["大盤_收盤", "大盤_MA20"]]

    print(f"   ✅ 大盤資料: {len(result)} 筆（{source}）"
          f"  {result.index[0]} ~ {result.index[-1]}")
    return result


# ==========================================
# 10. 整合單一個股
# ==========================================
def build_stock_df(code, name, start_date, benchmark_df=None):
    print(f"\n{'='*52}")
    print(f"  🚀 {code} {name}  (起始: {start_date})")
    print(f"{'='*52}")

    # ── 股價（還原版）─────────────────────────────────────────
    df = get_price_data(code, start_date)
    if df is None or df.empty:
        print(f"   ❌ {code} 無法取得股價，跳過。")
        return None
    source = df.pop("資料來源").iloc[0] if "資料來源" in df.columns else "?"
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    print(f"   ✅ 股價: {len(df)} 筆（{source}）")

    # ── 大戶指標 ──────────────────────────────────────────────
    df["大戶指標"] = calc_adl(df, code, start_date)
    print(f"   ✅ 大戶指標: 完成")
    time.sleep(random.uniform(0.5, 1.0))

    # ── 相對強弱（個股收盤 / 0050收盤）────────────────────────────
    if benchmark_df is not None and not benchmark_df.empty:
        # 用 merge 取代 join，避免 index 型別或格式的隱藏差異
        bench_series = benchmark_df[["大盤_收盤"]].reset_index()  # 變回 日期 欄
        bench_series.columns = ["日期", "大盤_收盤"]
        df_rsp = df[["日期", "收盤價"]].merge(bench_series, on="日期", how="left")
        rsp_vals = (df_rsp["收盤價"] / df_rsp["大盤_收盤"].replace(0, np.nan)).round(4)
        df["RSP"] = rsp_vals.values
        nz = df["RSP"].notna().sum()
        miss = df["RSP"].isna().sum()
        print(f"   ✅ 相對強弱(RSP): {nz} 筆有值"
              + (f"（{miss} 筆大盤無對應日期，保留 NaN）" if miss > 0 else ""))
    else:
        df["RSP"] = np.nan
        print("   ⚠️  RSP: 無大盤資料，補 NaN")

    # ── 外資持股比 + 連續買超天數 ─────────────────────────────
    df_foreign_hold, df_foreign_streak = get_foreign_data(code, start_date)
    if not df_foreign_hold.empty:
        df = df.merge(df_foreign_hold, on="日期", how="left")
        df["外資持股比"] = df["外資持股比"].ffill().fillna(0)
        print(f"   ✅ 外資持股比: {(df['外資持股比']!=0).sum()} 筆非零")
    else:
        df["外資持股比"] = 0
        print("   ⚠️  外資持股比: 全部補0")

    if not df_foreign_streak.empty:
        df = df.merge(df_foreign_streak, on="日期", how="left")
        df["外資連買天數"] = df["外資連買天數"].ffill().fillna(0)
        print(f"   ✅ 外資連買天數: 完成")
    else:
        df["外資連買天數"] = 0
    time.sleep(random.uniform(0.8, 1.2))

    # ── 投信持股比 ────────────────────────────────────────────
    df_trust = get_trust_data(code, start_date)
    if not df_trust.empty:
        df = df.merge(df_trust, on="日期", how="left")
        df["投信持股比"] = df["投信持股比"].ffill().fillna(0)
        print(f"   ✅ 投信持股比: {(df['投信持股比']!=0).sum()} 筆非零")
    else:
        df["投信持股比"] = 0
        print("   ⚠️  投信持股比: 全部補0")
    time.sleep(random.uniform(0.8, 1.2))

    # ── 融資變化率 ────────────────────────────────────────────
    df_margin = get_margin_data(code, start_date)
    if not df_margin.empty:
        df = df.merge(df_margin, on="日期", how="left")
        df["融資變化率"] = df["融資變化率"].ffill().fillna(0)
        print(f"   ✅ 融資變化率: {(df['融資變化率']!=0).sum()} 筆非零")
    else:
        df["融資變化率"] = 0
        print("   ⚠️  融資變化率: 全部補0")
    time.sleep(random.uniform(0.5, 0.8))

    # ── PER / PBR ─────────────────────────────────────────────
    df_val = get_valuation_data(code, start_date)
    if not df_val.empty:
        df = df.merge(df_val, on="日期", how="left")
        df["PER"] = df["PER"].ffill().fillna(0)
        df["PBR"] = df["PBR"].ffill().fillna(0)
        print(f"   ✅ PER/PBR: {(df['PER']!=0).sum()} 筆非零")
    else:
        df["PER"] = 0
        df["PBR"] = 0
        print("   ⚠️  PER/PBR: 全部補0")
    time.sleep(random.uniform(0.5, 0.8))

    # ── 月營收（YoY + MoM）────────────────────────────────────
    df_rev = get_revenue_data(code, start_date)
    if not df_rev.empty:
        df_rev["月份"] = df_rev["日期"].str[:7]
        for col in ["營收年增率", "營收月增率"]:
            monthly = df_rev.drop_duplicates("月份").set_index("月份")[col]
            df[col] = df["日期"].str[:7].map(monthly).ffill().fillna(0)
        print(f"   ✅ 營收 YoY/MoM: {(df['營收年增率']!=0).sum()} 筆非零")
    else:
        df["營收年增率"] = 0
        df["營收月增率"] = 0
        note = "（ETF）" if code == BENCHMARK_CODE else "（API 無資料）"
        print(f"   ⚠️  營收: 全部補0 {note}")

    df = df[df["收盤價"] > 0].sort_values("日期").reset_index(drop=True)

    # ── 衍生特徵：股票類型分類 ──────────────────────────────────
    DEFENSIVE_CODES = {"4904","1216","1210","9933","9914","9921","2353","1513"}
    CYCLICAL_CODES  = {"2376","3374","2374","2409","3481","2323"}
    if code in DEFENSIVE_CODES:
        df["StockType"] = 0
    elif code in CYCLICAL_CODES:
        df["StockType"] = 2
    else:
        df["StockType"] = 1
    type_name = ["防禦型","中性","景氣循環型"][int(df["StockType"].iloc[0])]
    print(f"   ✅ 股票類型: {type_name}")

    # ── 衍生特徵：動態 Beta ──────────────────────────────────────
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
        print("   ✅ 動態Beta: rolling_beta_60d / beta_regime 完成")
    else:
        df["rolling_beta_60d"] = 1.0
        df["beta_regime"]      = 0.5
        print("   ⚠️  動態Beta: 無大盤資料，補預設值")

    # ── 衍生特徵：營收動能（訂單能見度代理）────────────────────
    # 注意：market_stress / per_zscore 等需要 VIX/匯率，
    # 在 main() 的 macro_df.join() 後才計算（見 calc_derived_after_macro）
    df["rev_acc3m"]   = df["營收年增率"].rolling(3).mean().round(4).fillna(0)
    df["rev_accel"]   = (df["營收年增率"] - df["營收年增率"].shift(3)).round(4).fillna(0)
    df["rev_mom_acc"] = df["營收月增率"].rolling(3).mean().round(4).fillna(0)
    non_zero = (df["rev_acc3m"] != 0).sum()
    print(f"   ✅ 營收動能: rev_acc3m / rev_accel / rev_mom_acc ({non_zero} 筆非零)")

    # ── 衍生特徵：估值動能 ──────────────────────────────────────
    per_ma60  = df["PER"].replace(0, np.nan).rolling(60).mean()
    pbr_ma60  = df["PBR"].replace(0, np.nan).rolling(60).mean()
    per_std60 = df["PER"].replace(0, np.nan).rolling(60).std().replace(0, 1e-9)
    pbr_std60 = df["PBR"].replace(0, np.nan).rolling(60).std().replace(0, 1e-9)
    df["per_zscore"] = ((df["PER"] - per_ma60) / per_std60).round(4).fillna(0)
    df["pbr_zscore"] = ((df["PBR"] - pbr_ma60) / pbr_std60).round(4).fillna(0)
    df["per_trend"]  = (df["PER"] - df["PER"].shift(20)).round(4).fillna(0)
    print("   ✅ 估值動能: per_zscore / pbr_zscore / per_trend 完成")

    return df



# ==========================================
# 11. 寫出大盤參考分頁（0050）
# ==========================================
def build_benchmark_df(start_date, macro_df):
    print(f"\n{'='*52}")
    print(f"  📈 {BENCHMARK_CODE} {BENCHMARK_NAME}（大盤參考，不納入個股訓練）")
    print(f"{'='*52}")
    df = get_price_data(BENCHMARK_CODE, start_date)
    if df is None or df.empty:
        return None
    if "資料來源" in df.columns:
        df.pop("資料來源")
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    df["大戶指標"] = calc_adl(df, BENCHMARK_CODE, start_date)

    # 不抓法人/營收，0050 這些意義不大
    for col in ["外資持股比","外資連買天數","投信持股比","融資變化率","PER","PBR",
                "營收年增率","營收月增率","RSP",
                "StockType","rolling_beta_60d","beta_regime","market_stress",
                "rev_acc3m","rev_accel","rev_mom_acc",
                "per_zscore","pbr_zscore","per_trend"]:
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
# 欄位順序（統一）
# ==========================================
FINAL_COLS = [
    "日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量",
    "大戶指標","外資持股比","外資連買天數","投信持股比","融資變化率",
    "PER","PBR","營收年增率","營收月增率","台幣匯率","VIX指數",
    "SP500_ret","DXY_ret",
    "StockType","rolling_beta_60d","beta_regime","market_stress",
    "rev_acc3m","rev_accel","rev_mom_acc",
    "per_zscore","pbr_zscore","per_trend",
]


# ==========================================
# 主流程
# ==========================================
def main():
    if os.path.exists(FILE_NAME):
        try:
            os.remove(FILE_NAME)
            print(f"🗑️  已刪除舊資料庫，準備重新構建 {YEARS_BACK} 年數據...\n")
        except OSError:
            print(f"⚠️  無法刪除 {FILE_NAME}，請先關閉 Excel。")
            return

    today      = datetime.now()
    start_date = (today - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    print(f"📅 資料區間：{start_date} ~ {today.strftime('%Y-%m-%d')}（{YEARS_BACK} 年）\n")

    macro_df      = get_macro_data(start_date)
    benchmark_df  = get_benchmark_data(start_date)   # 用於計算 RSP

    with pd.ExcelWriter(FILE_NAME, engine="openpyxl") as writer:

        # ── 先寫 0050 大盤參考分頁 ──────────────────────────────
        df_bench = build_benchmark_df(start_date, macro_df)
        if df_bench is not None:
            df_bench.to_excel(writer, sheet_name=BENCHMARK_CODE, index=False)
            print(f"   💾 {BENCHMARK_CODE}（大盤參考）已寫入（{len(df_bench)} 筆）")

        # ── 逐一處理個股 ─────────────────────────────────────────
        for code, name in INDIVIDUAL_STOCKS.items():
            df = build_stock_df(code, name, start_date, benchmark_df=benchmark_df)
            if df is None:
                continue

            if not macro_df.empty:
                df = df.set_index("日期").join(macro_df, how="left").reset_index()
                for c in ["台幣匯率", "VIX指數", "SP500_ret", "DXY_ret"]:
                    if c in df.columns:
                        df[c] = df[c].ffill()

            # ── market_stress：VIX + 匯率 join 後才能計算 ──────────
            if "VIX指數" in df.columns and "台幣匯率" in df.columns:
                vix_ma20 = df["VIX指數"].rolling(20).mean()
                fx_ma20  = df["台幣匯率"].rolling(20).mean()
                df["market_stress"] = (
                    ((df["VIX指數"] - vix_ma20) / vix_ma20.replace(0, 1e-9)) * 0.5 +
                    ((df["台幣匯率"] - fx_ma20)  / fx_ma20.replace(0,  1e-9)) * 0.5
                ).round(4).fillna(0)
            else:
                df["market_stress"] = 0

            # RSP 欄位加入 final_cols
            all_cols = FINAL_COLS + (["RSP"] if "RSP" in df.columns else [])
            for c in all_cols:
                if c not in df.columns:
                    df[c] = 0
            df = df[all_cols].fillna(0)

            df.to_excel(writer, sheet_name=code, index=False)
            print(f"   💾 {code} 已寫入（共 {len(df)} 筆）")
            time.sleep(random.uniform(1.5, 2.0))

    print(f"\n✨ 【{YEARS_BACK}年期資料庫】構建完成！檔案：{FILE_NAME}")
    print(f"   個股分頁：{len(INDIVIDUAL_STOCKS)} 支")
    print(f"   大盤參考：{BENCHMARK_CODE}（單獨分頁，不納入模型訓練）")


if __name__ == "__main__":
    main()
