import pandas as pd
import numpy as np
import yfinance as yf
import time
import random
import urllib3
import ssl
import os
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# 1. 解決 macOS SSL 憑證問題
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 設定區
# ==========================================
FINMIND_TOKEN = "您的_TOKEN" 
dl = DataLoader()

STOCKS = {
    "0050": "元大台灣50", "1210": "大成", "1216": "統一", "1513": "中興電",
    "2323": "中環", "2353": "宏碁", "2374": "佳能", "2376": "技嘉",
    "2409": "友達", "3374": "精材", "3481": "群創", "4904": "遠傳",
    "6477": "安集", "9914": "美利達", "9921": "巨大", "9933": "中鼎"
}

FILE_NAME = "個股歷史分析資料庫.xlsx"

def get_incremental_data(code, name, start_date):
    """ 抓取數據並過濾無效的 0 軸數據 """
    try:
        # 確保抓取區間包含今天
        end_fetch = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        
        # --- A. 抓取股價 ---
        df_yf = pd.DataFrame()
        for suffix in [".TW", ".TWO"]:
            ticker_str = f"{code}{suffix}"
            df_yf = yf.Ticker(ticker_str).history(start=start_date, end=end_fetch)
            if not df_yf.empty:
                break
        
        if df_yf.empty:
            return None

        df = df_yf.reset_index()
        df = df.rename(columns={'Date': '日期', 'Open': '開盤', 'High': '最高', 'Low': '最低', 'Close': '收盤價', 'Volume': '成交量'})
        
        # 移除時區並轉格式
        df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

        # 🛑 【關鍵修正】過濾價格為 0 或 NaN 的無效資料 (例如尚未開盤或同步中的 4/27)
        df = df[df['收盤價'] > 0].copy()
        df = df.dropna(subset=['收盤價'])
        
        if df.empty:
            return None

        # --- B. 抓取籌碼 ---
        try:
            inst_df = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
            if inst_df is not None and not inst_df.empty:
                inst_df['net_buy'] = inst_df['buy'] - inst_df['sell']
                chip = inst_df.pivot_table(index='date', columns='name', values='net_buy', aggfunc='sum').reset_index()
                chip = chip.rename(columns={'date': '日期', 'Foreign_Investor': '外資(張)', 'Investment_Trust': '投信(張)'})
                for col in ['外資(張)', '投信(張)']:
                    if col in chip.columns:
                        chip[col] = (chip[col] / 1000).round(2)
                    else:
                        chip[col] = 0
                df = pd.merge(df, chip[['日期', '外資(張)', '投信(張)']], on='日期', how='left').fillna(0)
            else:
                df['外資(張)'], df['投信(張)'] = 0, 0
        except:
            df['外資(張)'], df['投信(張)'] = 0, 0

        # --- C. 計算 KD ---
        df = df.sort_values('日期').reset_index(drop=True)
        low_9 = df['最低'].rolling(9).min()
        high_9 = df['最高'].rolling(9).max()
        rsv = ((df['收盤價'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100).fillna(50)
        
        k_list, d_list = [50.0], [50.0]
        for r in rsv.values[1:]:
            new_k = k_list[-1] * (2/3) + r * (1/3)
            k_list.append(new_k)
            d_list.append(d_list[-1] * (2/3) + new_k * (1/3))
        
        df['K'] = np.round(k_list, 2)
        df['D'] = np.round(d_list, 2)
        df['資料來源'] = "FinMind/Yahoo"
        
        return df[['日期', '收盤價', 'K', 'D', '外資(張)', '投信(張)', '成交量', '資料來源']]

    except Exception as e:
        print(f"   ❌ {code} 錯誤: {e}")
        return None

if __name__ == "__main__":
    print(f"📁 數據更新中 (已加入 0 價過濾機制)...")
    
    existing_data = {}
    if os.path.exists(FILE_NAME):
        try:
            with pd.ExcelFile(FILE_NAME) as xls:
                for sheet in xls.sheet_names:
                    existing_data[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except:
            pass

    with pd.ExcelWriter(FILE_NAME, engine='openpyxl') as writer:
        for code, name in STOCKS.items():
            sheet_name = f"{code}_{name}"
            
            # 從最後日期的前兩天開始抓，確保補齊並覆蓋
            if sheet_name in existing_data:
                old_df = existing_data[sheet_name]
                old_df['日期'] = pd.to_datetime(old_df['日期']).dt.strftime('%Y-%m-%d')
                last_date_obj = datetime.strptime(old_df['日期'].max(), '%Y-%m-%d')
                start_fetch = (last_date_obj - timedelta(days=2)).strftime('%Y-%m-%d')
                
                new_df = get_incremental_data(code, name, start_fetch)
                if new_df is not None:
                    final_df = pd.concat([old_df, new_df], ignore_index=True)
                    final_df = final_df.drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
                    # 再次確保寫入前沒有 0 價
                    final_df = final_df[final_df['收盤價'] > 0]
                    final_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   ✅ {code} 更新完成")
                else:
                    old_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # 初次執行
                start_fetch = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                new_df = get_incremental_data(code, name, start_fetch)
                if new_df is not None:
                    new_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   🚀 {code} 初始化完成")

            time.sleep(random.uniform(0.3, 0.6))

    print(f"\n✨ 更新結束！請檢查 Excel，4/27 若無有效交易數據將不會出現 0。")
