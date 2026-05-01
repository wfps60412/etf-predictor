"""
app.py - 投資分析雲端控制台 v2
整合 Google Drive 同步
"""
import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import json
from datetime import datetime

# ── 路徑設定 ───────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("ETF_DATA_DIR", os.path.expanduser("~/etf_data"))
os.makedirs(DATA_DIR, exist_ok=True)

# 優先找 app.py 旁邊的檔案，找不到才用 DATA_DIR（與各腳本邏輯一致）
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
def _find(name):
    local = os.path.join(_APP_DIR, name)
    return local if os.path.exists(local) else os.path.join(DATA_DIR, name)

DB_FILE     = _find("ETF歷史分析資料庫.xlsx")
BT_REPORT   = _find("ETF回測報告_v2.xlsx")
PRED_REPORT = _find("ETF預測報告_v2.xlsx")
POWER_JSON  = _find("etf_model_power.json")

# ── 啟動時從 Drive 下載所有報告（雲端部署用）───────────────────
@st.cache_resource
def init_from_drive():
    """app 第一次啟動時執行，把 Drive 上的所有報告下載到 app 旁邊。"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from drive_sync import download_all
        results = download_all(dest_dir=os.path.dirname(os.path.abspath(__file__)))
        ok = sum(results.values())
        if ok > 0:
            st.toast(f"✅ 已從 Drive 下載 {ok} 個檔案")
    except Exception:
        pass  # 本地執行或 Drive 未設定時靜默跳過

init_from_drive()

st.set_page_config(
    page_title="📊 ETF 投資分析",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding: 1rem 1rem 2rem; }
    .stButton > button { width:100%; height:3rem; font-size:1rem;
                         border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("📊 ETF 投資分析控制台")
st.caption(f"最後開啟：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ── 工具函數 ───────────────────────────────────────────────────────
def file_mtime(path):
    if os.path.exists(path):
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
    return None

def show_file_status(label, path):
    mtime = file_mtime(path)
    if mtime:
        st.success(f"✅ {label} — 最後更新：{mtime}")
    else:
        st.warning(f"⚠️ {label} 尚未產生")

def run_script(script_name: str, args: list = None):
    """執行腳本並即時串流輸出"""
    # 永遠用 app.py 所在目錄找腳本，本地和雲端都能正確解析
    app_dir     = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(app_dir, script_name)
    cmd = [sys.executable, script_path] + (args or [])
    output_lines = []
    placeholder  = st.empty()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
        )
        for line in proc.stdout:
            output_lines.append(line.rstrip())
            placeholder.code("\n".join(output_lines[-30:]), language="")
        proc.wait()
        success = proc.returncode == 0
    except FileNotFoundError:
        output_lines.append(f"❌ 找不到腳本：{script_name}")
        placeholder.code("\n".join(output_lines), language="")
        success = False
    return success, "\n".join(output_lines)

def drive_upload(filenames: list):
    """上傳指定檔案到 Drive（直接 import drive_sync，有錯誤也不中斷）"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from drive_sync import upload_files
        upload_files(filenames)
        st.info("☁️ 已同步至 Google Drive / etf_predictor")
    except ImportError:
        st.warning("⚠️ 找不到 drive_sync.py，跳過上傳")
    except Exception as e:
        st.warning(f"⚠️ Drive 同步失敗（本地檔案完整）：{e}")

# ── Tabs ──────────────────────────────────────────────────────────
tab_update, tab_backtest, tab_predict, tab_report = st.tabs(
    ["🔄 更新資料庫", "📈 回測", "🔮 預測", "📊 查看報告"]
)

# ═══ Tab 1：更新資料庫 ════════════════════════════════════════════
with tab_update:
    st.header("🔄 ETF 資料庫更新")
    show_file_status("ETF歷史分析資料庫", DB_FILE)

    st.markdown("""
    **更新流程：**
    1. 本地無資料庫 → 從 Google Drive 下載最新版本
    2. 增量合併（補缺失 + 追加最新）
    3. 完整 xlsx 寫回本地
    4. 自動上傳覆蓋 Google Drive
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 啟動資料庫更新（含上傳）", key="btn_update"):
            with st.status("正在更新資料庫...", expanded=True) as status:
                success, _ = run_script("etf_analysis.py")
                if success:
                    status.update(label="✅ 完成！資料庫已更新並同步至 Drive",
                                  state="complete", expanded=False)
                else:
                    status.update(label="❌ 更新失敗", state="error")

    with col2:
        if st.button("🚀 只更新本地（不上傳）", key="btn_update_local"):
            with st.status("正在更新資料庫（本地）...", expanded=True) as status:
                success, _ = run_script("etf_analysis.py", ["--no-upload"])
                if success:
                    status.update(label="✅ 本地資料庫更新完成",
                                  state="complete", expanded=False)
                else:
                    status.update(label="❌ 更新失敗", state="error")

    st.divider()
    st.subheader("☁️ 手動 Drive 同步")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("⬆️ 手動上傳資料庫到 Drive", key="btn_manual_upload"):
            if os.path.exists(DB_FILE):
                with st.spinner("上傳中..."):
                    drive_upload(["ETF歷史分析資料庫.xlsx"])
            else:
                st.error("本地資料庫不存在，請先執行更新")
    with col4:
        if st.button("⬇️ 從 Drive 下載最新資料庫", key="btn_manual_download"):
            with st.spinner("下載中..."):
                success, _ = run_script("drive_sync.py", ["download"])
            if success:
                st.success("✅ 下載完成")
            else:
                st.error("下載失敗，請確認 Drive 連線設定")

# ═══ Tab 2：回測 ══════════════════════════════════════════════════
with tab_backtest:
    st.header("📈 策略回測（WFV + 跨 ETF 聯合訓練）")
    show_file_status("ETF回測報告", BT_REPORT)

    st.markdown("""
    **回測完成後：** 自動上傳報告到 Google Drive，並更新 etf_model_power.json
    """)

    # 偵測是否在 Streamlit Cloud（雲端記憶體不足以跑回測）
    is_cloud = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit_sharing" \
               or os.environ.get("HOME", "") == "/home/appuser"

    col1, col2 = st.columns(2)
    with col1:
        if is_cloud:
            st.info("☁️ 雲端環境記憶體不足以執行回測（需 4~8GB）\n\n請在**本地電腦**執行：\n```\npython etf_backtester.py\n```\n完成後報告會自動上傳到 Drive，這裡可直接查看。")
        else:
            if st.button("▶️ 執行 ETF 回測", key="btn_backtest"):
                with st.status("正在執行回測（需時 10~30 分鐘）...", expanded=True) as status:
                    success, _ = run_script("etf_backtester.py")
                    if success:
                        drive_upload(["ETF回測報告_v2.xlsx", "etf_model_power.json"])
                        status.update(label="✅ 回測完成！報告已同步至 Drive",
                                      state="complete", expanded=False)
                    else:
                        status.update(label="❌ 回測失敗", state="error")
    with col2:
        if os.path.exists(BT_REPORT):
            with open(BT_REPORT, "rb") as f:
                st.download_button("📥 下載回測報告", f,
                    file_name="ETF回測報告.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══ Tab 3：預測 ══════════════════════════════════════════════════
with tab_predict:
    st.header("🔮 AI 機器學習預測")
    show_file_status("ETF預測報告", PRED_REPORT)

    if os.path.exists(POWER_JSON):
        try:
            with open(POWER_JSON, encoding='utf-8') as f:
                power_data = json.load(f)
            rows = [{'ETF': f"{c} {v.get('name','')}", '類型': v.get('category','-'),
                     '12M勝率': f"{v.get('wr_12m',0):.1%}", '評級': v.get('stars','-')}
                    for c, v in power_data.items()]
            df_power = pd.DataFrame(rows).sort_values('12M勝率', ascending=False)
            st.dataframe(df_power, use_container_width=True, hide_index=True)
        except Exception:
            st.info("讀取預測力資料失敗，請先執行回測")
    else:
        st.info("💡 建議先執行回測以取得最新模型預測力評分")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 執行 ETF 預測", key="btn_predict"):
            with st.status("正在執行預測...", expanded=True) as status:
                success, _ = run_script("etf_predictor.py")
                if success:
                    drive_upload(["ETF預測報告_v2.xlsx"])
                    status.update(label="✅ 預測完成！報告已同步至 Drive",
                                  state="complete", expanded=False)
                else:
                    status.update(label="❌ 預測失敗", state="error")
    with col2:
        if os.path.exists(PRED_REPORT):
            with open(PRED_REPORT, "rb") as f:
                st.download_button("📥 下載預測報告", f,
                    file_name="ETF預測報告.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══ Tab 4：查看報告 ══════════════════════════════════════════════
with tab_report:
    st.header("📊 報告總覽")

    report_choice = st.selectbox("選擇要查看的報告",
        ["ETF 回測報告", "ETF 預測報告"])

    FILE_MAP = {
        "ETF 回測報告": (_find("ETF回測報告_v2.xlsx"),   "回測準確率總覽",  1),
        "ETF 預測報告": (_find("ETF預測報告_v2.xlsx"),  "📊 ETF預測總覽", 1),
    }
    filepath, sheet_name, header_row = FILE_MAP[report_choice]

    if not os.path.exists(filepath):
        st.warning(f"找不到 {os.path.basename(filepath)}，請先執行對應功能")
    else:
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            st.info(f"共 {len(df)} 筆，最後更新：{file_mtime(filepath)}")
            st.dataframe(df, use_container_width=True, height=500)
            with open(filepath, "rb") as f:
                st.download_button(f"📥 下載完整 {report_choice}（Excel）", f,
                    file_name=os.path.basename(filepath),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"讀取報告失敗：{e}")

    st.divider()
    st.subheader("📋 資料庫最新狀態")
    if os.path.exists(DB_FILE):
        try:
            xl = pd.ExcelFile(DB_FILE)
            summary = []
            for sheet in xl.sheet_names:
                df_s = pd.read_excel(DB_FILE, sheet_name=sheet, usecols=["日期","收盤價"])
                df_s["日期"] = df_s["日期"].astype(str).str[:10]
                last_row = df_s.sort_values("日期").iloc[-1]
                summary.append({"ETF": sheet, "最新日期": last_row["日期"],
                                 "最新收盤": last_row["收盤價"], "筆數": len(df_s)})
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"讀取資料庫失敗：{e}")
    else:
        st.info("資料庫尚未建立，請先執行更新")
