"""
etf_analysis.py  v1.0
ETF 歷史資料庫建構

爬取10支 ETF 的股價、技術指標、宏觀特徵
輸出：ETF歷史分析資料庫.xlsx

ETF 清單（5基本盤 + 5建議）：
  0050   元大台灣50
  00713  元大台灣高息低波
  00762  元大全球AI
  00965  元大航太防衛科技
  009816 凱基台灣TOP50
  00878  國泰永續高股息
  00929  復華台灣科技優息
  0056   元大高股息
  00919  群益台灣精選高息
  00881  國泰台灣科技龍頭
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
import urllib3
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 設定
# ==========================================
ETF_LIST = {
    # 基本盤 5 支
    '0050':   '元大台灣50',
    '00713':  '元大台灣高息低波',
    '00762':  '元大全球AI',
    '00965':  '元大航太防衛科技',
    '009816': '凱基台灣TOP50',
    # 建議新增 5 支
    '00878':  '國泰永續高股息',
    '00929':  '復華台灣科技優息',
    '0056':   '元大高股息',
    '00919':  '群益台灣精選高息',
    '00881':  '國泰台灣科技龍頭',
}

# ETF 類型分類（用於特徵選擇）
ETF_TYPE = {
    '0050':   '市值型',
    '009816': '市值型',
    '00881':  '科技型',
    '00762':  '主題型',   # 全球AI
    '00965':  '主題型',   # 航太防衛
    '00713':  '高息低波',
    '0056':   '高息型',
    '00878':  '高息ESG',
    '00919':  '高息型',
    '00929':  '科技高息',
}

# Beta_SP500 靜態估算值（ETF 對美股的敏感度）
ETF_BETA_SP500 = {
    '0050':   0.35,   # 台灣大盤，中度正相關
    '009816': 0.33,   # 台灣TOP50，類似0050
    '00881':  0.40,   # 科技龍頭，較高正相關
    '00762':  0.55,   # 全球AI，高度正相關（追蹤全球AI）
    '00965':  0.45,   # 航太防衛，中高正相關
    '00713':  0.20,   # 高息低波，較低正相關
    '0056':   0.22,   # 高股息，較低正相關
    '00878':  0.25,   # 高息ESG，較低正相關
    '00919':  0.28,   # 精選高息，較低正相關
    '00929':  0.38,   # 科技高息，中度正相關
}

FILE_NAME   = 'ETF歷史分析資料庫.xlsx'

def auto_width(ws, min_w=12, max_w=60):
    from openpyxl.utils import get_column_letter
    for col in ws.columns:
        best = min_w
        for cell in col:
            if cell.value:
                best = min(max_w, max(best, len(str(cell.value))+4))
        ws.column_dimensions[get_column_letter(col[0].column)].width = best
YEARS_BACK  = 4
DAYS_BACK   = 365 * YEARS_BACK
MAX_WORKERS = 4

FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoid2ZwczYwNDEyIiwiZW1haWwiOiJ3ZnBzNjA0MTJAZ21haWwuY29tIn0.QmlmQSgB5NG721tueiYOoYTIj-6nGfPw9sRTu0hDsjw"
FINMIND_API   = "https://api.finmindtrade.com/api/v4/data"
FINMIND_HDR   = {"Authorization": f"Bearer {FINMIND_TOKEN}", "User-Agent": "Mozilla/5.0"}

# ==========================================
# 宏觀資料
# ==========================================
def get_macro_data(start_date):
    print("🌐 抓取宏觀指標...")
    try:
        raw = yf.download(
            ["TWD=X", "^VIX", "^GSPC", "DX-Y.NYB"],
            start=start_date, progress=False
        )["Close"]
        raw = raw.rename(columns={
            "TWD=X": "台幣匯率", "^VIX": "VIX指數",
            "^GSPC": "SP500",   "DX-Y.NYB": "DXY",
        })
        raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")
        raw["SP500_ret"] = raw["SP500"].pct_change(1).round(6)
        raw["DXY_ret"]   = raw["DXY"].pct_change(1).round(6)
        print("   ✅ 宏觀指標成功")
        return raw[["台幣匯率", "VIX指數", "SP500_ret", "DXY_ret"]]
    except Exception as e:
        print(f"   ❌ 宏觀指標失敗: {e}")
        return pd.DataFrame()

# ==========================================
# ETF 股價（yfinance）
# ==========================================
def get_etf_price(code, start_date):
    for suffix in [".TW", ".TWO"]:
        try:
            df = yf.download(f"{code}{suffix}", start=start_date,
                             progress=False, auto_adjust=True,
                             multi_level_index=False)
            if not df.empty and len(df) > 5:
                df = df.reset_index()
                df["漲跌價差"] = df["Close"].diff().round(2).fillna(0)
                out = df.rename(columns={
                    "Date":"日期","Open":"開盤價","High":"最高價",
                    "Low":"最低價","Close":"收盤價","Volume":"成交量",
                })[["日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量"]]
                out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
                out["來源"] = f"yfinance{suffix}"
                return out
        except Exception:
            continue
    return pd.DataFrame()

# ==========================================
# 大盤 0050 作為 benchmark
# ==========================================
def get_benchmark(start_date):
    df = get_etf_price("0050", start_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = df["日期"].astype(str).str[:10]
    bench = df.set_index("日期")[["收盤價"]].rename(columns={"收盤價":"大盤_收盤"})
    return bench

# ==========================================
# 建構單一 ETF 的特徵 DataFrame
# ==========================================
def build_etf_df(code, name, start_date, benchmark_df=None, macro_df=None):
    print(f"\n{'='*52}")
    print(f"  🚀 {code} {name}")
    print(f"{'='*52}")

    df = get_etf_price(code, start_date)
    if df is None or df.empty:
        print(f"   ❌ 無法取得股價，跳過")
        return None

    source = df.pop("來源").iloc[0]
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    print(f"   ✅ 股價: {len(df)} 筆（{source}）")

    # RSP（相對大盤強弱）
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.reset_index()
        bench.columns = ["日期", "大盤_收盤"]
        merged = df[["日期","收盤價"]].merge(bench, on="日期", how="left")
        df["RSP"] = (merged["收盤價"] / merged["大盤_收盤"].replace(0, np.nan)).round(4)
        nz = df["RSP"].notna().sum()
        print(f"   ✅ RSP: {nz} 筆有值")
    else:
        df["RSP"] = np.nan

    # ETF 類型（靜態）
    etype = ETF_TYPE.get(code, '市值型')
    etype_num = {'市值型':0,'高息低波':1,'高息型':1,'高息ESG':1,
                 '科技型':2,'主題型':2,'科技高息':2}.get(etype, 0)
    df["ETFType"]    = etype_num
    df["ETFTypeName"] = etype
    print(f"   ✅ ETF類型: {etype}")

    # Beta_SP500 靜態值
    df["beta_sp500_static"] = ETF_BETA_SP500.get(code, 0.30)

    df = df[df["收盤價"] > 0].sort_values("日期").reset_index(drop=True)

    # 合併宏觀
    if macro_df is not None and not macro_df.empty:
        df = df.set_index("日期").join(macro_df, how="left").reset_index()
        for c in ["台幣匯率","VIX指數","SP500_ret","DXY_ret"]:
            if c in df.columns:
                df[c] = df[c].ffill()

    # 動態 Beta（對 0050）
    if benchmark_df is not None and not benchmark_df.empty:
        bench2 = benchmark_df.reset_index()
        bench2.columns = ["日期","大盤_收盤"]
        bench2["大盤_ret"] = bench2["大盤_收盤"].pct_change()
        df_b = df[["日期","收盤價"]].copy()
        df_b["stock_ret"] = df_b["收盤價"].pct_change()
        df_b = df_b.merge(bench2[["日期","大盤_ret"]], on="日期", how="left")
        cov60 = df_b["stock_ret"].rolling(60).cov(df_b["大盤_ret"])
        var60 = df_b["大盤_ret"].rolling(60).var().replace(0, np.nan)
        df["rolling_beta_60d"] = (cov60/var60).round(4).ffill().fillna(1.0)
        df["beta_regime"]      = df["rolling_beta_60d"].rank(pct=True).round(4)
        print("   ✅ 動態Beta: 完成")
    else:
        df["rolling_beta_60d"] = 1.0
        df["beta_regime"]      = 0.5

    # market_stress
    if "VIX指數" in df.columns and "台幣匯率" in df.columns:
        vix_ma = df["VIX指數"].rolling(20).mean()
        fx_ma  = df["台幣匯率"].rolling(20).mean()
        df["market_stress"] = (
            ((df["VIX指數"]-vix_ma)/vix_ma.replace(0,1e-9))*0.5 +
            ((df["台幣匯率"]-fx_ma)/fx_ma.replace(0,1e-9))*0.5
        ).round(4).fillna(0)
    else:
        df["market_stress"] = 0.0

    # 估值動能（用收盤價的歷史百分位代替 PER）
    close_ma60  = df["收盤價"].rolling(60).mean()
    close_std60 = df["收盤價"].rolling(60).std().replace(0,1e-9)
    df["price_zscore"] = ((df["收盤價"]-close_ma60)/close_std60).round(4).fillna(0)
    df["price_trend"]  = (df["收盤價"]-df["收盤價"].shift(20)).round(4).fillna(0)

    # 確保所有欄位存在
    for col in ["台幣匯率","VIX指數","SP500_ret","DXY_ret"]:
        if col not in df.columns:
            df[col] = 0.0

    return df

# ==========================================
# 欄位順序
# ==========================================
FINAL_COLS = [
    "日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量",
    "RSP","台幣匯率","VIX指數","SP500_ret","DXY_ret",
    "ETFType","beta_sp500_static","rolling_beta_60d","beta_regime",
    "market_stress","price_zscore","price_trend",
]

# ==========================================
# 增量更新輔助
# ==========================================
def get_existing_last_dates():
    if not os.path.exists(FILE_NAME):
        return {}
    try:
        xl    = pd.ExcelFile(FILE_NAME)
        dates = {}
        for sheet in xl.sheet_names:
            df = pd.read_excel(FILE_NAME, sheet_name=sheet, usecols=["日期"])
            if not df.empty:
                dates[sheet.strip()] = str(df["日期"].max())[:10]
        return dates
    except Exception:
        return {}

# ==========================================
# 主流程
# ==========================================
def main():
    today      = datetime.now()
    full_start = (today - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    existing = get_existing_last_dates()
    is_incr  = bool(existing)

    if is_incr:
        min_last   = min(existing.values())
        incr_start = (pd.Timestamp(min_last) - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
        print(f"📅 增量更新模式（補抓起點: {incr_start}）")
    else:
        incr_start = full_start
        print(f"📅 全量建置模式（{YEARS_BACK} 年，共 {len(ETF_LIST)} 支 ETF）")
        if os.path.exists(FILE_NAME):
            try:
                os.remove(FILE_NAME)
                print(f"🗑️  已刪除舊資料庫")
            except OSError:
                print("⚠️  請先關閉 Excel 再執行"); return

    macro_df     = get_macro_data(incr_start)
    benchmark_df = get_benchmark(incr_start)
    if not benchmark_df.empty:
        print(f"   ✅ 大盤資料: {len(benchmark_df)} 筆")

    # 並行抓取
    print(f"\n🚀 並行抓取（{MAX_WORKERS} threads）...")
    results = {}

    def fetch_one(args):
        code, name = args
        try:
            return code, build_etf_df(code, name, incr_start,
                                       benchmark_df=benchmark_df,
                                       macro_df=macro_df)
        except Exception as e:
            print(f"   ❌ {code} 例外: {e}")
            return code, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, item): item[0]
                   for item in ETF_LIST.items()}
        for fut in as_completed(futures):
            code, df_new = fut.result()
            results[code] = df_new

    # 寫入 Excel
    with pd.ExcelWriter(FILE_NAME, engine="openpyxl") as writer:
        ok = 0
        for code, name in ETF_LIST.items():
            df_new = results.get(code)
            if df_new is None:
                continue

            all_cols = FINAL_COLS + (["RSP"] if "RSP" in df_new.columns
                                     and "RSP" not in FINAL_COLS else [])
            for c in FINAL_COLS:
                if c not in df_new.columns:
                    df_new[c] = 0.0
            df_new = df_new[FINAL_COLS].fillna(0)

            # 增量合併
            sheet_name = f"{code} {name}"
            if is_incr and code in existing:
                try:
                    old = pd.read_excel(FILE_NAME, sheet_name=sheet_name)
                    old["日期"] = old["日期"].astype(str).str[:10]
                    df_new = pd.concat([
                        old[old["日期"] < df_new["日期"].min()],
                        df_new
                    ]).drop_duplicates("日期").sort_values("日期").reset_index(drop=True)
                    for c in FINAL_COLS:
                        if c not in df_new.columns: df_new[c] = 0.0
                    df_new = df_new[FINAL_COLS].fillna(0)
                except Exception:
                    pass

            df_new.to_excel(writer, sheet_name=sheet_name, index=False)
            # 套用欄寬
            ws = writer.sheets[sheet_name]
            auto_width(ws)
            print(f"   💾 {code} {name}: {len(df_new)} 筆")
            ok += 1

    mode = "增量" if is_incr else "全量"
    print(f"\n✨ 【{mode}資料庫】完成！共 {ok} 支 ETF → {FILE_NAME}")

if __name__ == "__main__":
    main()
