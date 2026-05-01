"""
drive_sync.py — Google Drive 同步工具 v2
統一上傳到 etf_predictor 資料夾，支援覆蓋同一個檔案（固定 file ID）
"""
import os, sys, json
from pathlib import Path

DRIVE_FOLDER_ID  = os.environ.get("DRIVE_FOLDER_ID", "1WPPwlcEKbYMd6FUPnWBJ8P2O4WVOKa-2")
DATA_DIR         = os.environ.get("ETF_DATA_DIR", str(Path.home() / "etf_data"))
SYNC_FILES = {
    "ETF歷史分析資料庫.xlsx": "ETF歷史分析資料庫.xlsx",
    "ETF回測報告_v2.xlsx":    "ETF回測報告_v2.xlsx",
    "ETF預測報告_v2.xlsx":    "ETF預測報告_v2.xlsx",
    "etf_model_power.json":   "etf_model_power.json",
}
MIME_MAP = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".json": "application/json",
}
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_IDS_CACHE       = os.path.join(_SCRIPT_DIR, "drive_ids.json")
SCOPES           = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_PATH       = os.path.join(_SCRIPT_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(_SCRIPT_DIR, "credentials.json")

def _load_ids():
    if os.path.exists(_IDS_CACHE):
        with open(_IDS_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_ids(ids):
    with open(_IDS_CACHE, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

def _get_service():
    """
    授權優先順序：
    1. Streamlit Secrets 的 token JSON（雲端部署，用 refresh_token 自動換 token）
    2. 本地 token.json（本地開發）
    3. 本地 credentials.json + 瀏覽器授權（第一次本地設定用）
    """
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = None

    # ── 優先：環境變數 GOOGLE_TOKEN（Streamlit Secrets 頂層設定）
    # Streamlit Cloud 會把頂層 Secrets 自動轉成環境變數，
    # subprocess 呼叫的子 process 也能讀到
    raw_token = os.environ.get("GOOGLE_TOKEN", "")
    if raw_token:
        try:
            import json as _json
            token_info = _json.loads(raw_token)
            creds = Credentials.from_authorized_user_info(token_info, SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            if creds and creds.valid:
                print("  [drive] ✅ 使用環境變數 token 授權成功")
                return build("drive", "v3", credentials=creds)
        except Exception as e:
            print(f"  [drive] 環境變數 token 解析失敗：{e}")

    # ── 備用：本地 token.json ────────────────────────────────────
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # 第一次本地授權（只需要跑一次，之後用 token.json）
            from google_auth_oauthlib.flow import InstalledAppFlow
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    "找不到授權資訊！\n"
                    "雲端：在 Streamlit Secrets [google] 填入 token（見說明）\n"
                    "本地：放 credentials.json 到專案資料夾，執行一次授權"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # 更新本地 token
        with open(TOKEN_PATH, "w") as tf:
            tf.write(creds.to_json())

    return build("drive", "v3", credentials=creds)

def _upload_one(service, local_path, drive_name, known_ids):
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    ext   = Path(local_path).suffix.lower()
    mime  = MIME_MAP.get(ext, "application/octet-stream")
    fid   = known_ids.get(drive_name, "")

    if fid:
        # 先確認檔案是否還存在，若已刪除則清掉舊 ID 改為新建
        try:
            service.files().get(fileId=fid, fields="id").execute()
        except HttpError:
            print(f"  ⚠️  舊 ID 已失效（{fid}），改為新建")
            fid = ""

    media = MediaFileUpload(local_path, mimetype=mime, resumable=True)
    if fid:
        result = service.files().update(fileId=fid, media_body=media).execute()
        print(f"  ✅ 覆蓋：{drive_name}  (ID: {result['id']})")
    else:
        meta   = {"name": drive_name, "parents": [DRIVE_FOLDER_ID]}
        result = service.files().create(body=meta, media_body=media, fields="id,name").execute()
        print(f"  🆕 新建：{drive_name}  (ID: {result['id']})")
    return result["id"]

def upload_files(filenames=None):
    """上傳指定本地檔名（不含路徑）到 Drive etf_predictor 資料夾。None = 全部。"""
    targets   = filenames or list(SYNC_FILES.keys())
    print(f"\n☁️  上傳到 Drive / etf_predictor（{len(targets)} 個）...")
    service   = _get_service()
    known_ids = _load_ids()
    updated   = {}
    for fname in targets:
        drive_name = SYNC_FILES.get(fname, fname)
        local_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(local_path):
            print(f"  ⚠️  找不到：{local_path}，跳過")
            continue
        fid = _upload_one(service, local_path, drive_name, known_ids)
        updated[drive_name] = fid
    known_ids.update(updated)
    _save_ids(known_ids)
    print(f"☁️  完成，drive_ids.json 已更新\n")
    return updated

def _download_one(service, drive_name: str, local_path: str) -> bool:
    """下載單一檔案到指定本地路徑，檔案不存在 Drive 時回傳 False。"""
    import io
    from googleapiclient.http import MediaIoBaseDownload
    known_ids = _load_ids()
    fid = known_ids.get(drive_name, "")
    if not fid:
        print(f"  ℹ️  {drive_name} 無 Drive ID，跳過")
        return False
    try:
        buf = io.BytesIO()
        dl  = MediaIoBaseDownload(buf, service.files().get_media(fileId=fid))
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.seek(0)
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(buf.read())
        print(f"  ✅ 下載：{drive_name} → {local_path}")
        return True
    except Exception as e:
        print(f"  ⚠️  {drive_name} 下載失敗：{e}")
        return False


def download_all(dest_dir: str = None):
    """
    從 Drive 下載所有報告檔案到 dest_dir。
    dest_dir 預設為 drive_sync.py 所在目錄（讓 _find() 找得到）。
    """
    if dest_dir is None:
        dest_dir = _SCRIPT_DIR
    print("☁️  從 Drive 下載所有報告...")
    service = _get_service()
    results = {}
    for drive_name in SYNC_FILES:
        local_path = os.path.join(dest_dir, drive_name)
        results[drive_name] = _download_one(service, drive_name, local_path)
    ok = sum(results.values())
    print(f"☁️  下載完成（{ok}/{len(results)} 個成功）")
    return results


def download_db():
    """向下相容舊呼叫，只下載資料庫。"""
    service   = _get_service()
    local_path = os.path.join(_SCRIPT_DIR, "ETF歷史分析資料庫.xlsx")
    return _download_one(service, "ETF歷史分析資料庫.xlsx", local_path)

def upload_file_path(local_path: str, drive_name: str):
    """
    直接傳本地完整路徑上傳，不依賴 DATA_DIR。
    供 etf_analysis / backtester / predictor 呼叫。
    """
    if not os.path.exists(local_path):
        print(f"  ⚠️  找不到檔案：{local_path}，跳過上傳")
        return None
    print(f"\n☁️  上傳到 Drive / etf_predictor：{drive_name}...")
    service   = _get_service()
    known_ids = _load_ids()
    fid = _upload_one(service, local_path, drive_name, known_ids)
    known_ids[drive_name] = fid
    _save_ids(known_ids)
    print(f"☁️  完成\n")
    return fid


if __name__ == "__main__":
    cmd   = sys.argv[1] if len(sys.argv) > 1 else "upload"
    files = sys.argv[2:] if len(sys.argv) > 2 else None
    if cmd == "upload":
        upload_files(files if files else None)
    elif cmd == "download":
        download_db()
    else:
        print("用法：python drive_sync.py [upload [檔名...] | download]")
