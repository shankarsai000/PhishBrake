# PhishBrake Chrome Extension

Manifest V3 React popup for scanning selected text with the local PhishBrake API.

## Build

From this directory:

```powershell
npm install
npm run build
```

The compiled extension is written to `dist/`.

## Load in Chrome

1. Start the PhishBrake API from `C:\jawbreaker-main\jawbreaker-main`:

   ```powershell
   ..\venv\Scripts\python.exe app.py
   ```

2. Open `chrome://extensions`.
3. Enable **Developer mode**.
4. Choose **Load unpacked**.
5. Select `C:\jawbreaker-main\jawbreaker-main\phishbrake-extension\dist`.
6. Select text on any page, right-click it, and choose **Scan with PhishBrake**.

The extension calls `http://127.0.0.1:7860/api/scan`. Keep the local PhishBrake server running while scanning.
