# 13MA 的魔力 — Streamlit 儀表板

把老哥的 13MA 操作規則整理成互動儀表板：走勢圖、開盤劇情、回測（單一週期／30 分K定方向＋5 分K進場）、閉環系統（三週期共振、每日時間記錄、規則統計）、零股隔日沖、費氏座標。資料來源可選 Yahoo Finance、永豐 Shioaji API（只讀行情、不下單）或上傳 CSV。

教學用途，不構成投資建議。

## 檔案

| 檔案 | 用途 |
|---|---|
| `ma13_dashboard.py` | 主程式（Streamlit 入口） |
| `ma13_system.py` | 規則與回測引擎（也能單獨在命令列執行） |
| `ma13_closed_loop.py` | 閉環系統：日線＋30 分＋5 分共振、每日時間記錄、交易記錄與規則統計（也能單獨在命令列執行） |
| `sinopac_data.py` | 永豐 API 登入與 K 線下載 |
| `tw_symbols.py` | 台股代號下拉清單、自動判斷 .TW／.TWO |
| `tw_codes.csv` | 上市櫃代號表（約 2,158 檔，下拉清單的資料來源） |
| `requirements.txt` | 套件版本 |
| `.streamlit/config.toml` | Streamlit 設定 |
| `.streamlit/secrets.toml.example` | 密碼／金鑰範本（真正的值放雲端 Secrets） |

## 本機執行

```
pip install -r requirements.txt
streamlit run ma13_dashboard.py
```

## 部署到 Streamlit Community Cloud（免費、固定網址）

### 1. 把程式放上 GitHub

1. 到 <https://github.com> 註冊或登入。
2. 右上角 **+** → **New repository**：名稱例如 `13ma-dashboard`，選 **Private**（私人，別人看不到程式碼），按 **Create repository**。
3. 在新 repo 頁面點 **uploading an existing file**，把這個資料夾裡的**所有檔案和 `.streamlit` 資料夾**拖進去，按 **Commit changes**。
   - 也可以用 GitHub Desktop：File → Add local repository 選這個資料夾 → Publish repository。
   - 確認**沒有**上傳 `.streamlit/secrets.toml`（真正的密碼檔）。

### 2. 在 Streamlit Cloud 建立 App

1. 到 <https://share.streamlit.io>，用 GitHub 帳號登入，允許它讀取你的 repo（含私人 repo）。
2. 按 **Create app** → **Deploy a public app from GitHub**。
3. 填：
   - **Repository**：`你的帳號/13ma-dashboard`
   - **Branch**：`main`
   - **Main file path**：`ma13_dashboard.py`
   - **App URL**：自訂網址，例如 `ma13-你的名字` → 網址會是 `https://ma13-你的名字.streamlit.app`
4. 展開 **Advanced settings**：
   - **Python version** 選 **3.12**
   - **Secrets** 留空即可（不設密碼，打開網址就能用）。日後想加密碼再貼上：

     ```toml
     APP_PASSWORD = "換成你自己的密碼"
     ```

5. 按 **Deploy**，第一次安裝套件約 3–5 分鐘。

### 3. 之後要改程式

在 GitHub 上修改或重新上傳檔案，Streamlit Cloud 會自動重新部署，網址不變。
密碼、金鑰要改：App 右下角 **Manage app** → **Settings** → **Secrets**。

## 注意事項

- **網址是永久的**，只要 App 沒被你刪除就一直有效。但免費方案在一段時間沒人使用後會進入休眠，打開時會看到「Yes, get this app back up!」按鈕，按下等約 30 秒就恢復。
- **不設 `APP_PASSWORD` 時**：任何拿到網址的人都能打開使用；永豐金鑰不會自動帶入，每次都要在網頁上自己填。
- **永豐金鑰**：只存在當次瀏覽的記憶體中，其他訪客看不到，也不會寫入檔案。若在 Secrets 加上 `SJ_API_KEY`、`SJ_SEC_KEY`，只有在設了 `APP_PASSWORD` 時才會自動帶入。
- **永豐 API 在雲端可能連不上**：Streamlit Cloud 的主機在美國，若永豐限制海外連線就會登入失敗，這時改用 Yahoo 資料來源即可。
- **Yahoo 偶爾會限流**：雲端主機的 IP 是共用的，若出現讀取失敗，稍等幾分鐘再試。程式已快取 10 分鐘內的資料。
- 代號清單來自 twstock 內建代號表，新上市股票不在清單時可直接在下拉選單輸入代號。
