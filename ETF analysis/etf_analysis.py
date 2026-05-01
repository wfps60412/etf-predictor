"""
etf_analysis.py v2.0
ETF 歷史資料庫建構 - 強化版

修改重點：
1. 時間尺度擴張至 10 年[cite: 1]
2. 新增 ATR (波動率指標)[cite: 1]
3. 增加總經權重：美債10Y、黃金、原油[cite: 1]
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

# ETF 實際上市日期（用於早期歷史補充）
ETF_LAUNCH_DATE = {
    '0050':   '2003-07-23',   # 涵蓋 2008金融海嘯、2015股災、2020疫情
    '0056':   '2007-12-26',   # 涵蓋 2008金融海嘯後半、2015、2020
    '00713':  '2017-09-27',   # 以下皆2017後，不需要補充
    '00762':  '2023-06-08',
    '00965':  '2024-01-31',
    '009816': '2022-06-29',
    '00878':  '2020-10-20',
    '00929':  '2022-06-23',
    '00919':  '2022-10-20',
    '00881':  '2020-12-04',
}

# 需要補充早期資料的ETF（上市 > 10年前）
EARLY_HISTORY_CODES = {'0050', '0056'}

FILE_NAME   = 'ETF歷史分析資料庫.xlsx'
YEARS_BACK  = 20  # 修改1：拉長至 20 年[cite: 1]
DAYS_BACK   = 365 * YEARS_BACK
MAX_WORKERS = 4

# 總經代號清單 (修改3：增加總經權重資料)[cite: 1]
MACRO_TICKERS = {
    "TWD=X": "台幣匯率", 
    "^VIX": "VIX指數", 
    "^GSPC": "SP500", 
    "DX-Y.NYB": "DXY",
    "^TNX": "美債10Y", 
    "GC=F": "黃金", 
    "CL=F": "原油"
}

def auto_width(ws, min_w=12, max_w=60):
    from openpyxl.utils import get_column_letter
    for col in ws.columns:
        best = min_w
        for cell in col:
            if cell.value:
                best = min(max_w, max(best, len(str(cell.value))+4))
        ws.column_dimensions[get_column_letter(col[0].column)].width = best

# ==========================================
# 宏觀資料 (修改3：整合多樣化總經數據)[cite: 1]
# ==========================================
def get_macro_data(start_date):
    print("🌐 抓取強化版宏觀指標...")
    try:
        raw = yf.download(list(MACRO_TICKERS.keys()), start=start_date, progress=False)["Close"]
        raw = raw.rename(columns=MACRO_TICKERS)
        raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")
        
        # 計算各指標的日變動率
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
                # 修改2：計算 ATR (14日)[cite: 1]
                h_l = df['High'] - df['Low']
                h_pc = abs(df['High'] - df['Close'].shift(1))
                l_pc = abs(df['Low'] - df['Close'].shift(1))
                tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
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
    if df is None or df.empty: return pd.DataFrame()
    df["日期"] = df["日期"].astype(str).str[:10]
    return df.set_index("日期")[["收盤價"]].rename(columns={"收盤價":"大盤_收盤"})

# ==========================================
# 建構單一 ETF 的特徵 DataFrame
# ==========================================
def build_etf_df(code, name, start_date, benchmark_df=None, macro_df=None):
    print(f"\n{'='*52}\n  🚀 {code} {name}\n{'='*52}")

    df = get_etf_price(code, start_date)
    if df is None or df.empty: return None

    source = df.pop("來源").iloc[0]
    df["日期"] = df["日期"].astype(str).str[:10]
    df = df.sort_values("日期").reset_index(drop=True)
    print(f"   ✅ 股價與ATR: {len(df)} 筆")

    # RSP
    if benchmark_df is not None and not benchmark_df.empty:
        merged = df[["日期","收盤價"]].merge(benchmark_df.reset_index(), on="日期", how="left")
        df["RSP"] = (merged["收盤價"] / merged["大盤_收盤"].replace(0, np.nan)).round(4)
    
    # 靜態特徵
    df["ETFType"] = {'市值型':0,'高息低波':1,'高息型':1,'高息ESG':1,'科技型':2,'主題型':2,'科技高息':2}.get(ETF_TYPE.get(code, '市值型'), 0)
    df["beta_sp500_static"] = ETF_BETA_SP500.get(code, 0.30)

    # 合併宏觀 (包含新增的美債、黃金、原油)[cite: 1]
    if macro_df is not None and not macro_df.empty:
        df = df.set_index("日期").join(macro_df, how="left").reset_index()
        df = df.ffill()

    # 動態 Beta 與市場壓力指標
    if benchmark_df is not None and not benchmark_df.empty:
        bench_ret = benchmark_df["大盤_收盤"].pct_change()
        stock_ret = df["收盤價"].pct_change()
        cov60 = stock_ret.rolling(60).cov(bench_ret)
        var60 = bench_ret.rolling(60).var().replace(0, np.nan)
        df["rolling_beta_60d"] = (cov60/var60).round(4).fillna(1.0)
    
    # 估值動能
    close_ma60  = df["收盤價"].rolling(60).mean()
    close_std60 = df["收盤價"].rolling(60).std().replace(0,1e-9)
    df["price_zscore"] = ((df["收盤價"]-close_ma60)/close_std60).round(4).fillna(0)
    df["price_trend"]  = (df["收盤價"]-df["收盤價"].shift(20)).round(4).fillna(0)

    return df

# ==========================================
# 欄位順序與執行
# ==========================================
FINAL_COLS = [
    "日期","開盤價","最高價","最低價","收盤價","漲跌價差","成交量","ATR",
    "RSP","台幣匯率","VIX指數","SP500_ret","DXY_ret","美債10Y_ret","黃金_ret","原油_ret",
    "ETFType","beta_sp500_static","rolling_beta_60d","price_zscore","price_trend"
]

def get_existing_last_dates():
    if not os.path.exists(FILE_NAME): return {}
    try:
        xl = pd.ExcelFile(FILE_NAME)
        return {sheet.strip(): str(pd.read_excel(FILE_NAME, sheet_name=sheet, usecols=["日期"])["日期"].max())[:10] for sheet in xl.sheet_names}
    except: return {}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch-early', action='store_true',
                        help='只補充 0050/0056 的早期歷史（不重跑其他ETF）')
    args, _ = parser.parse_known_args()

    today     = datetime.now()
    full_start = (today - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    existing  = get_existing_last_dates()
    is_incr   = bool(existing)

    # ── 模式判斷 ─────────────────────────────────────────────────
    if args.patch_early and existing:
        # 只補早期：0050/0056 從上市日起，宏觀從 2003-07-01 起
        print("📅 早期歷史補充模式（只更新 0050 / 0056）")
        macro_start = '2003-07-01'
        macro_df    = get_macro_data(macro_start)
        benchmark_df = get_benchmark(macro_start)

        # ── Step1：先把所有分頁讀入記憶體（overlay 有 bug，不能邊讀邊寫）
        print("  📖 讀取現有資料庫所有分頁...")
        xl_existing = pd.ExcelFile(FILE_NAME)
        all_sheets = {}
        for s in xl_existing.sheet_names:
            try:
                df_s = pd.read_excel(FILE_NAME, sheet_name=s)
                df_s['日期'] = df_s['日期'].astype(str).str[:10]
                all_sheets[s] = df_s
                print(f"    讀入 {s}: {len(df_s)} 筆")
            except Exception as e:
                print(f"    ⚠️  讀取 {s} 失敗: {e}")
        xl_existing.close()

        # ── Step2：為 0050/0056 下載並合併早期資料 ──────────────────
        for code in EARLY_HISTORY_CODES:
            name        = ETF_LIST[code]
            early_start = ETF_LAUNCH_DATE[code]
            sheet_name  = f"{code} {name}"
            print(f"\n  補充 {code} {name}（從 {early_start}）")

            df_new = build_etf_df(code, name, early_start, benchmark_df, macro_df)
            if df_new is None:
                print(f"  ❌ {code} 無資料，跳過")
                continue

            for c in FINAL_COLS:
                if c not in df_new.columns: df_new[c] = 0.0
            df_new = df_new[FINAL_COLS].fillna(0)
            df_new['日期'] = df_new['日期'].astype(str).str[:10]

            df_old = all_sheets.get(sheet_name, pd.DataFrame())
            if not df_old.empty:
                existing_dates = set(df_old['日期'].tolist())
                df_early = df_new[~df_new['日期'].isin(existing_dates)]
                df_merged = pd.concat([df_early, df_old], ignore_index=True)
                df_merged = df_merged.sort_values('日期').reset_index(drop=True)
                for c in FINAL_COLS:
                    if c not in df_merged.columns: df_merged[c] = 0.0
                df_merged = df_merged[FINAL_COLS].fillna(0)
                print(f"  ✅ 補充早期: {len(df_early)} 筆  合計: {len(df_merged)} 筆")
            else:
                df_merged = df_new
                print(f"  ✅ 新建: {len(df_merged)} 筆")

            all_sheets[sheet_name] = df_merged

        # ── Step3：全部分頁一次性寫回新檔案，再替換原檔 ────────────
        import shutil
        tmp_file = FILE_NAME + '.tmp'
        print(f"\n  💾 寫回所有分頁（共 {len(all_sheets)} 個）...")
        with pd.ExcelWriter(tmp_file, engine='openpyxl') as writer:
            for s_name, df_s in all_sheets.items():
                df_s.to_excel(writer, sheet_name=s_name, index=False)
                auto_width(writer.sheets[s_name])
        shutil.move(tmp_file, FILE_NAME)   # 原子性替換，避免寫到一半失敗

        print(f"✅ 早期歷史補充完成！")
        return   # 提前結束，不執行後續全量邏輯

    # ── 一般模式（全量或日常增量）────────────────────────────────
    start_dt = (pd.Timestamp(min(existing.values())) - pd.DateOffset(days=30)).strftime("%Y-%m-%d") if is_incr else full_start

    if not is_incr and os.path.exists(FILE_NAME): os.remove(FILE_NAME)

    macro_df     = get_macro_data(start_dt)
    benchmark_df = get_benchmark(start_dt)

    results = {}
    def fetch_one(item):
        code, name = item
        return code, build_etf_df(code, name, start_dt, benchmark_df, macro_df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, it): it[0] for it in ETF_LIST.items()}
        for fut in as_completed(futures):
            c, df = fut.result()
            results[c] = df

    with pd.ExcelWriter(FILE_NAME, engine="openpyxl") as writer:
        for code, name in ETF_LIST.items():
            df_new = results.get(code)
            if df_new is None: continue
            
            for c in FINAL_COLS:
                if c not in df_new.columns: df_new[c] = 0.0
            df_save = df_new[FINAL_COLS].fillna(0)

            sheet_name = f"{code} {name}"
            df_save.to_excel(writer, sheet_name=sheet_name, index=False)
            auto_width(writer.sheets[sheet_name])
            print(f"   💾 {sheet_name} 已儲存")

    print(f"\n✨ 資料庫建置完成！共 {len(ETF_LIST)} 支 ETF → {FILE_NAME}")

if __name__ == "__main__":
    main()
