import pandas as pd
import yfinance as yf
import os

# 設定檔案路徑
db_file = "個股歷史資料庫.xlsx"
report_file = "三週期複合回測報告.xlsx" 

def initialize_database():
    if not os.path.exists(report_file):
        print(f"❌ 找不到報告檔案：{report_file}")
        return
    
    if not os.path.exists(db_file):
        pd.DataFrame([["建立日期", "Initial"]]).to_excel(db_file, sheet_name='README', index=False)

    print(f"📊 正在從 {report_file} 讀取舊紀錄...")
    # 直接讀取 Excel
    try:
        raw_df = pd.read_excel(report_file)
    except Exception as e:
        print(f"❌ 讀取 Excel 失敗: {e}")
        return
    
    with pd.ExcelWriter(db_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        for _, row in raw_df.iterrows():
            # 轉換代號為字串，並處理可能的 NaN
            code = str(row['代號']).strip().split('.')[0] 
            name = str(row['名稱']).strip() if '名稱' in row else "Unknown"
            
            print(f"🔍 正在補全 {code} {name} 的歷史數據...")
            
            # 定義需要回補的關鍵日期欄位名稱 (請確認你的 Excel 標題是否叫這些名字)
            date_cols = {'3M': '3M對標日', '6M': '6M對標日', '12M': '12M對標日'}
            pred_cols = {'3M': '預測3M', '6M': '預測6M', '12M': '預測12M'}
            
            history_rows = []
            for period in ['3M', '6M', '12M']:
                date = row.get(date_cols[period])
                pred = row.get(pred_cols[period])
                
                if pd.isna(date) or date == "數據不足" or pd.isna(pred):
                    continue
                
                # 處理日期格式
                target_date = pd.to_datetime(date).strftime('%Y-%m-%d')
                
                # 從 yfinance 抓取那天的還原股價
                for suf in [".TW", ".TWO"]:
                    try:
                        tk = yf.download(f"{code}{suf}", start=target_date, period="5d", progress=False)
                        if not tk.empty:
                            p_old = tk['Adj Close'].iloc[0]
                            # 處理預測值中的百分比符號
                            pred_val = float(str(pred).replace('%',''))
                            
                            history_rows.append({
                                "日期": target_date,
                                "收盤價": round(float(p_old), 2),
                                f"預測{period}": pred_val
                            })
                            break
                    except: continue
            
            if history_rows:
                new_df = pd.DataFrame(history_rows)
                sn = f"{code}_{name}"[:31].replace('/','-')
                try:
                    old_df = pd.read_excel(db_file, sheet_name=sn)
                    final_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['日期']).sort_values('日期')
                except:
                    final_df = new_df
                
                final_df.to_excel(writer, sheet_name=sn, index=False)
                print(f"✅ {code} 補全成功")

    print("\n✨ 補全完成！現在執行分析 review 腳本吧。")

if __name__ == "__main__":
    initialize_database()
