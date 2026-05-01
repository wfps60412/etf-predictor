"""
etf_analysis.py v2.2
ETF 歷史資料庫建構

修正：
- 暫存檔副檔名改為 _tmp.xlsx，解決 openpyxl 無法辨識 .tmp 的錯誤
- 資料庫存在時自動增量更新（補缺失日期 + 追加最新），不重跑全量
- 完成後自動上傳覆蓋 Google Drive（需 drive_sync.py + credentials.json）
"""
import pandas as pd
import numpy as np
import yfinance as yf
import urllib3
import os
import sys
import shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 設定
# ==========================================
ETF_LIST = {
    '0050':   '元大台灣50',
    '00713':  '元大台灣高息低波',
    '00762':  '元大全球AI',
    '00965':  '元大航太防衛科技',
    '009816': '凱基台灣TOP50',
    '00878':  '國泰永續高股息',
    '00929':  '復華台灣科技優息',
    '0056':   '元大高股息',
    '00919':  '群益台灣精選高息',
    '00881':  '國泰台灣科技龍頭',
}

ETF_TYPE = {
    '0050': '市值型', '009816': '市值型', '00881': '科技型',
    '00762': '主題型', '00965': '主題型', '00713': '高息低波',
    '0056': '高息型', '00878': '高息ESG', '00919': '高息型', '00929': '科技高息',
}

ETF_BETA_SP500 = {
    '0050': 0.35, '009816': 0.33, '00881': 0.40, '00762': 0.55,
    '00965': 0.45, '00713': 0.20, '0056': 0.22, '00878': 0.25,
    '00919': 0.28, '00929': 0.38,
}

ETF_LAUNCH_DATE = {
    '0050':   '2003-07-23',
    '0056':   '2007-12-26',
    '00713':  '2017-09-27',
    '00762':  '2023-06-08',
    '00965':  '2024-01-31',
    '009816': '2022-06-29',
    '00878':  '2020-10-20',
    '00929':  '2022-06-23',
    '00919':  '2022-10-20',
    '00881':  '2020-12-04',
}

FILE_NAME   = 'ETF歷史分析資料庫.xlsx'
# ↑ 如需固定路徑，改成絕對路徑即可，例如：
# FILE_NAME = os.path.expanduser('~/etf_data/ETF歷史分析資料庫.xlsx')

YEARS_BACK  = 20
DAYS_BACK   = 365 * YEARS_BACK
MAX_WORKERS = 4

MACRO_TICKERS = {
    "TWD=X":    "台幣匯率",
    "^VIX":     "VIX指數",
    "^GSPC":    "SP500",
    "DX-Y.NYB": "DXY",
    "^TNX":     "美債10Y",
    "GC=F":     "黃金",
    "CL=F":     "原油",
}

FINAL_COLS = [
    "日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量","ATR",
    "RSP","台幣匯率","VIX指數","SP500_ret","DXY_ret","美債10Y_ret","黃金_ret","原油_ret",
    "ETFType","beta_sp500_static","rolling_beta_60d","price_zscore","price_trend"
]

# ==========================================
# 工具函數
# ==========================================
def auto_width(ws, min_w=12, max_w=60):
    from openpyxl.utils import get_column_letter
    for col in ws.columns:
        best = min_w
        for cell in col:
            if cell.value:
                best = min(max_w, max(best, len(str(cell.value)) + 4))
        ws.column_dimensions[get_column_letter(col[0].column)].width = best


def safe_write(all_sheets: dict, target_path: str):
    """
    先寫到同目錄下的 _tmp.xlsx，成功後再用 shutil.move 替換原檔。
    這樣即使寫到一半失敗，原檔案也不會損壞。
    """
    dir_name  = os.path.dirname(os.path.abspath(target_path))
    base_name = os.path.basename(target_path).replace('.xlsx', '_tmp.xlsx')
    tmp_path  = os.path.join(dir_name, base_name)   # ← 副檔名仍是 .xlsx

    with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
        for s_name, df_s in all_sheets.items():
            df_s.to_excel(writer, sheet_name=s_name, index=False)
            auto_width(writer.sheets[s_name])
            print(f"   💾 {s_name} ({len(df_s)} 筆)")

    shutil.move(tmp_path, target_path)


def get_existing_sheet_info():
    """讀取現有資料庫每個 sheet 的最後日期與全部日期集合"""
    if not os.path.exists(FILE_NAME):
        return {}
    try:
        xl   = pd.ExcelFile(FILE_NAME)
        info = {}
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(FILE_NAME, sheet_name=sheet, usecols=["日期"])
                df["日期"] = df["日期"].astype(str).str[:10]
                info[sheet] = {
                    "last_date": df["日期"].max(),
                    "all_dates": set(df["日期"].tolist()),
                }
            except Exception:
                pass
        return info
    except Exception:
        return {}


def merge_incremental(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    合併新舊資料：
    - 重複日期以 df_new 為準（修正最新資料）
    - df_old 中缺少的日期保留（補歷史空缺）
    - 按日期排序
    """
    df_old = df_old.copy()
    df_new = df_new.copy()
    df_old["日期"] = df_old["日期"].astype(str).str[:10]
    df_new["日期"] = df_new["日期"].astype(str).str[:10]

    new_dates       = set(df_new["日期"])
    df_old_filtered = df_old[~df_old["日期"].isin(new_dates)]

    merged = pd.concat([df_old_filtered, df_new], ignore_index=True)
    return merged.sort_values("日期").reset_index(drop=True)


# ==========================================
# 宏觀資料
# ==========================================
def get_macro_data(start_date):
    print("🌐 抓取宏觀指標...")
    try:
        raw = yf.download(list(MACRO_TICKERS.keys()), start=start_date, progress=False)["Close"]
        raw = raw.rename(columns=MACRO_TICKERS)
        raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")

        macro_feats = pd.DataFrame(index=raw.index)
        for col in raw.columns:
            macro_feats[col] = raw[col]
            macro_feats[f"{col}_ret"] = raw[col].pct_change(1).round(6)

        print("   ✅ 宏觀指標成功")
        return macro_feats
    except Exception as e:
        print(f"   ❌ 宏觀指標失敗: {e}")
        return pd.DataFrame()


# ==========================================
# ETF 股價
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
                h_l  = df['High'] - df['Low']
                h_pc = abs(df['High'] - df['Close'].shift(1))
                l_pc = abs(df['Low']  - df['Close'].shift(1))
                tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
                df['ATR'] = tr.rolling(window=14).mean().round(4).fillna(0)
                out = df.rename(columns={
                    "Date":"日期","Open":"開盤價","High":"最高價",
                    "Low":"最低價","Close":"收盤價","Volume":"成交量",
                })[["日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量","ATR"]]
                out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
                out["來源"] = f"yfinance{suffix}"
                return out
        except Exception:
            continue
    return pd.DataFrame()


def get_benchmark(start_date):
    df = get_etf_price("0050", start_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = df["日期"].astype(str).str[:10]
    return df.set_index("日期")[["收盤價"]].rename(columns={"收盤價": "大盤_收盤"})


# ==========================================
# 建構單一 ETF 特徵
# ==========================================
def build_etf_df(code, name, start_date, benchmark_df=None, macro_df=None):
    print(f"\n{'='*52}\n  🚀 {code} {name}\n{'='*52}")

    df = get_etf_price(code, start_date)
    if df is None or df.empty:
        return None

    df.pop("來源")
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    print(f"   ✅ 股價與ATR: {len(df)} 筆")

    if benchmark_df is not None and not benchmark_df.empty:
        merged = df[["日期","收盤價"]].merge(benchmark_df.reset_index(), on="日期", how="left")
        df["RSP"] = (merged["收盤價"] / merged["大盤_收盤"].replace(0, np.nan)).round(4)

    type_map = {'市值型':0,'高息低波':1,'高息型':1,'高息ESG':1,'科技型':2,'主題型':2,'科技高息':2}
    df["ETFType"]           = type_map.get(ETF_TYPE.get(code, '市值型'), 0)
    df["beta_sp500_static"] = ETF_BETA_SP500.get(code, 0.30)

    if macro_df is not None and not macro_df.empty:
        df = df.set_index("日期").join(macro_df, how="left").reset_index()
        df = df.ffill()

    if benchmark_df is not None and not benchmark_df.empty:
        bench_ret = benchmark_df["大盤_收盤"].pct_change()
        stock_ret = df["收盤價"].pct_change()
        cov60 = stock_ret.rolling(60).cov(bench_ret)
        var60 = bench_ret.rolling(60).var().replace(0, np.nan)
        df["rolling_beta_60d"] = (cov60 / var60).round(4).fillna(1.0)

    close_ma60  = df["收盤價"].rolling(60).mean()
    close_std60 = df["收盤價"].rolling(60).std().replace(0, 1e-9)
    df["price_zscore"] = ((df["收盤價"] - close_ma60) / close_std60).round(4).fillna(0)
    df["price_trend"]  = (df["收盤價"] - df["收盤價"].shift(20)).round(4).fillna(0)

    return df


# ==========================================
# 主程式
# ==========================================
def main(upload_to_drive: bool = True):
    today      = datetime.now()
    full_start = (today - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    # ── Step 0：若本地無資料庫，嘗試從 Drive 下載 ─────────────────
    if not os.path.exists(FILE_NAME):
        print("📂 本地無資料庫，嘗試從 Google Drive 下載...")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from drive_sync import download_db
            download_db()
        except Exception as e:
            print(f"   ℹ️  跳過 Drive 下載（{e}），將進行全量建立")

    # ── Step 1：讀取現有資料狀態 ───────────────────────────────────
    sheet_info     = get_existing_sheet_info()
    is_incremental = bool(sheet_info)

    if is_incremental:
        last_dates  = [v["last_date"] for v in sheet_info.values() if v["last_date"]]
        fetch_start = (pd.Timestamp(min(last_dates)) - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
        print(f"📊 增量模式：從 {fetch_start} 抓取（現有資料最早至 {min(last_dates)}，最晚至 {max(last_dates)}）")
    else:
        fetch_start = full_start
        print(f"🆕 全量模式：從 {fetch_start} 起全量建立")

    # ── Step 2：抓取宏觀與各 ETF 新資料 ───────────────────────────
    macro_df     = get_macro_data(fetch_start)
    benchmark_df = get_benchmark(fetch_start)

    new_results = {}

    def fetch_one(item):
        code, name = item
        sheet_name = f"{code} {name}"
        if is_incremental and sheet_name in sheet_info:
            etf_last  = sheet_info[sheet_name]["last_date"]
            etf_start = (pd.Timestamp(etf_last) - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
        else:
            launch    = ETF_LAUNCH_DATE.get(code, full_start)
            etf_start = max(launch, full_start)
        return code, name, build_etf_df(code, name, etf_start, benchmark_df, macro_df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, it): it[0] for it in ETF_LIST.items()}
        for fut in as_completed(futures):
            code, name, df = fut.result()
            new_results[code] = (name, df)

    # ── Step 3：讀取現有分頁（增量模式才需要）────────────────────
    all_sheets = {}
    if is_incremental:
        print("\n📖 讀取現有資料庫...")
        xl_existing = pd.ExcelFile(FILE_NAME)
        for s in xl_existing.sheet_names:
            try:
                df_s = pd.read_excel(FILE_NAME, sheet_name=s)
                df_s["日期"] = df_s["日期"].astype(str).str[:10]
                all_sheets[s] = df_s
                print(f"   讀入 {s}: {len(df_s)} 筆")
            except Exception as e:
                print(f"   ⚠️ 讀取 {s} 失敗: {e}")
        xl_existing.close()

    # ── Step 4：合併新舊資料 ───────────────────────────────────────
    for code, name in ETF_LIST.items():
        _, df_new = new_results.get(code, (name, None))
        if df_new is None:
            print(f"   ⚠️ {code} 無新資料，跳過")
            continue

        for c in FINAL_COLS:
            if c not in df_new.columns:
                df_new[c] = 0.0
        df_new = df_new[FINAL_COLS].fillna(0)
        df_new["日期"] = df_new["日期"].astype(str).str[:10]

        sheet_name = f"{code} {name}"

        if is_incremental and sheet_name in all_sheets:
            df_old = all_sheets[sheet_name]
            for c in FINAL_COLS:
                if c not in df_old.columns:
                    df_old[c] = 0.0
            df_old    = df_old[FINAL_COLS].fillna(0)
            df_merged = merge_incremental(df_old, df_new)
            added     = len(df_merged) - len(df_old)
            print(f"   🔄 {sheet_name}: 新增/修正 {added} 筆，合計 {len(df_merged)} 筆")
        else:
            df_merged = df_new
            print(f"   🆕 {sheet_name}: 首次建立 {len(df_merged)} 筆")

        all_sheets[sheet_name] = df_merged[FINAL_COLS].fillna(0)

    # ── Step 5：安全寫回（暫存檔副檔名仍為 .xlsx）────────────────
    print(f"\n💾 寫回本地資料庫（共 {len(all_sheets)} 個分頁）...")
    os.makedirs(os.path.dirname(os.path.abspath(FILE_NAME)), exist_ok=True)
    safe_write(all_sheets, FILE_NAME)
    print(f"✨ 本地資料庫更新完成！→ {FILE_NAME}")

    # ── Step 6：上傳到 Google Drive（可選）───────────────────────
    if upload_to_drive:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from drive_sync import upload_file_path
            upload_file_path(os.path.abspath(FILE_NAME), "ETF歷史分析資料庫.xlsx")
        except ImportError:
            print("   ℹ️  找不到 drive_sync.py，跳過上傳")
        except Exception as e:
            print(f"   ⚠️  Drive 上傳失敗（{e}），本地檔案仍完整")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true", help="跳過上傳到 Google Drive")
    args, _ = parser.parse_known_args()
    main(upload_to_drive=not args.no_upload)
