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
    from googleapiclient.discovery import build

    # ── 優先 1：Streamlit Secrets 裡的 Service Account JSON ──────
    try:
        import streamlit as st
        if "google" in st.secrets and "service_account" in st.secrets["google"]:
            import json as _json
            from google.oauth2 import service_account
            sa_info = _json.loads(st.secrets["google"]["service_account"])
            creds   = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            return build("drive", "v3", credentials=creds)
    except Exception:
        pass

    # ── 優先 2：本地 service_account.json ───────────────────────
    sa_path = os.path.join(_SCRIPT_DIR, "service_account.json")
    if os.path.exists(sa_path):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        return build("drive", "v3", credentials=creds)

    # ── 備用：OAuth token（本地開發，需要瀏覽器）────────────────
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    "找不到授權方式！請選擇其中一種：\n"
                    "  雲端：在 Streamlit Secrets 填入 service_account JSON\n"
                    "  本地：放 service_account.json 或 credentials.json 到專案資料夾"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

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

def download_db():
    known_ids = _load_ids()
    fid = known_ids.get("ETF歷史分析資料庫.xlsx", "")
    if not fid:
        print("ℹ️  無資料庫 Drive ID，跳過下載")
        return False
    import io
    from googleapiclient.http import MediaIoBaseDownload
    service = _get_service()
    buf = io.BytesIO()
    dl  = MediaIoBaseDownload(buf, service.files().get_media(fileId=fid))
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    local_path = os.path.join(DATA_DIR, "ETF歷史分析資料庫.xlsx")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(buf.read())
    print(f"  ✅ 下載完成：{local_path}")
    return True

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
