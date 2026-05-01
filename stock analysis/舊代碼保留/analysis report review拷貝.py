import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_smart_review():
    db_file = "個股歷史資料庫.xlsx"
    report_file = "三週期複合回測報告.xlsx"
    
    if not os.path.exists(db_file):
        print(f"❌ 找不到資料庫檔案 {db_file}")
        return

    print("🔍 正在從本地資料庫讀取資料進行回測...")
    try:
        all_sheets = pd.read_excel(db_file, sheet_name=None)
    except Exception as e:
        print(f"❌ 讀取 Excel 失敗: {e}")
        return

    results = []
    periods = {"3M": 63, "6M": 126, "12M": 252}
    run_date = datetime.now().strftime('%Y-%m-%d')

    for sn, df in all_sheets.items():
        if sn == 'README' or df.empty:
            continue
            
        # 取得代號與名稱
        parts = sn.split('_')
        code = parts[0]
        name = parts[1] if len(parts) > 1 else "Unknown"
        
        try:
            # 統一日期格式
            df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
            latest = df.iloc[-1]
            p_now = float(latest['收盤價'])
            
            # 每一列的基礎資料
            stock_res = {
                "代號": code, 
                "名稱": name, 
                "目前RSI": latest.get('RSI', 'N/A'),
                "資料起始日": df['日期'].iloc[0]
            }

            for p_k, days in periods.items():
                idx = -(days + 1)
                p_old = None
                p_date = "數據不足"
                pred = latest.get(f'預測{p_k}', 0)
                
                # 1. 優先使用本地資料庫
                if len(df) >= abs(idx):
                    hist_row = df.iloc[idx]
                    p_old = float(hist_row['收盤價'])
                    p_date = hist_row['日期']
                    pred = hist_row.get(f'預測{p_k}', pred)
                
                # 2. 本地不足則聯網補位
                else:
                    for suf in [".TW", ".TWO"]:
                        try:
                            yf_df = yf.download(f"{code}{suf}", period="2y", progress=False)
                            if not yf_df.empty and len(yf_df) >= abs(idx):
                                # 優先用 Adj Close，抓不到就用 Close
                                if 'Adj Close' in yf_df.columns:
                                    p_old = float(yf_df['Adj Close'].iloc[idx])
                                else:
                                    p_old = float(yf_df['Close'].iloc[idx])
                                p_date = yf_df.index[idx].strftime('%Y-%m-%d')
                                break
                        except:
                            continue
                
                # 3. 計算結果
                if p_old is not None and p_old != 0:
                    real_gain = ((p_now - p_old) / p_old) * 100
                    match = (pred > 0 and real_gain > 0) or (pred <= 0 and real_gain <= 0)
                    is_suc = "成功" if abs(real_gain - pred) < 10 or (match and abs(real_gain) > abs(pred)) else "失敗"
                    
                    stock_res[f"{p_k}對標日"] = p_date
                    stock_res[f"預測{p_k}"] = f"{round(float(pred), 1)}%"
                    stock_res[f"真實{p_k}"] = f"{round(real_gain, 1)}%"
                    stock_res[f"結果{p_k}"] = is_suc
                else:
                    stock_res[f"{p_k}對標日"] = "數據不足"
                    stock_res[f"預測{p_k}"] = "N/A"
                    stock_res[f"真實{p_k}"] = "N/A"
                    stock_res[f"結果{p_k}"] = "數據不足"

            results.append(stock_res)
            print(f"✔️ {code} {name} 處理完成")
        except Exception as e:
            print(f"⚠️ {code} 發生錯誤: {e}")

    # --- 寫入 Excel 部分 ---
    if results:
        final_df = pd.DataFrame(results)
        # 指定欄位順序 (確保美觀)
        col_order = ["代號", "名稱", "目前RSI", "資料起始日", 
                     "3M對標日", "預測3M", "真實3M", "結果3M",
                     "6M對標日", "預測6M", "真實6M", "結果6M",
                     "12M對標日", "預測12M", "真實12M", "結果12M"]
        
        # 僅保留 dataframe 中有的欄位
        final_cols = [c for c in col_order if c in final_df.columns]
        final_df = final_df[final_cols]

        try:
            writer = pd.ExcelWriter(report_file, engine='xlsxwriter')
            final_df.to_excel(writer, sheet_name='高勝率回測報告', index=False)
            
            # 套用顏色格式
            workbook = writer.book
            ws = writer.sheets['高勝率回測報告']
            suc_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            fail_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            
            # 設定自動寬度
            for i, col in enumerate(final_df.columns):
                ws.set_column(i, i, 12)
                
            ws.conditional_format(1, 0, len(final_df), len(final_df.columns)-1, 
                                 {'type': 'cell', 'criteria': 'equal to', 'value': '"成功"', 'format': suc_fmt})
            ws.conditional_format(1, 0, len(final_df), len(final_df.columns)-1, 
                                 {'type': 'cell', 'criteria': 'equal to', 'value': '"失敗"', 'format': fail_fmt})
            
            writer.close()
            print(f"\n✨ 報告已成功產出至: {os.path.abspath(report_file)}")
        except Exception as e:
            print(f"❌ 寫入 Excel 失敗: {e}")
    else:
        print("🛑 最終結果列表為空，請檢查資料庫分頁內容。")

if __name__ == "__main__":
    run_smart_review()
