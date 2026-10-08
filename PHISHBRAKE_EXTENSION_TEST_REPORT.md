# PhishBrake Extension Test Report

**Test date:** 2026-10-05  
**Backend:** `JAWBREAKER_BACKEND=heuristic`  
**API:** `http://127.0.0.1:7860`  
**Extension build:** Vite production build from `phishbrake-extension`

## Executive summary

The backend integration and packaged extension were validated successfully. The automated API/security matrix passed all executed cases after retesting Unicode with an explicit UTF-8 request. The Chrome extension build and generated Manifest V3 package passed static validation.

Interactive Chrome-only cases are listed as **MANUAL** because this environment cannot load or drive `chrome://extensions` or invoke a browser context-menu extension action.

## Automated results

| ID | Test | Result | Evidence |
|---|---|---|---|
| PB-001 | Backend health | PASS | `/health` returned HTTP 200 and `status: ok` |
| PB-003 | Urgent account phishing | PASS | `dangerous / credential_theft` |
| PB-004 | Credential harvesting | PASS | `dangerous / credential_theft` |
| PB-005 | QR/quishing text | PASS | `dangerous / credential_theft` |
| PB-006 | Manager gift-card impersonation | PASS | `dangerous / payment_request` |
| PB-007 | Family emergency scam | PASS | `dangerous / family_impersonation` |
| PB-008 | Job/task scam | PASS | `dangerous / job_scam` |
| PB-009 | Lookalike domain | PASS | `dangerous / credential_theft` |
| PB-010 | Safe verification code | PASS | `safe / none` |
| PB-011 | Safe appointment reminder | PASS | `safe / none` |
| PB-012 | Safe security notification | PASS | `safe / none` |
| PB-016 | 10,000-character input | PASS | HTTP 200 and analysis returned |
| PB-017 | Unicode input | PASS | Explicit UTF-8 request returned `dangerous / credential_theft` |
| PB-020 | Missing `text` field | PASS | HTTP 422 |
| PB-021 | Malformed JSON | PASS | HTTP 422 |
| PB-022 | CORS preflight | PASS | HTTP 200, `Access-Control-Allow-Origin: *` |
| PB-023 | Repeated identical scan | PASS | `dangerous` followed by `dangerous` |
| PB-024 | Different messages isolated | PASS | Phishing message was `dangerous`; appointment message was `safe` |

## Extension package validation

| Check | Result |
|---|---|
| `npm run build` | PASS |
| Generated `dist/manifest.json` | PASS |
| Manifest name | `PhishBrake` |
| Manifest permissions | `contextMenus`, `activeTab`, `scripting`, `storage` |
| API host permission | `http://127.0.0.1:7860/*` |
| Compiled service worker | `service-worker-loader.js` |
| Popup entry | `index.html` |
| Source diagnostics | No errors in `app.py`, `App.jsx`, or `background.js` |
| Python regression suite | 72 passed |

## Chrome-interactive cases requiring manual execution

These cases were not falsely marked as passed because they require a real Chrome extension runtime:

| ID | Test | Status |
|---|---|---|
| PB-001 | Load unpacked extension and confirm service worker | MANUAL |
| PB-002 | Select text and confirm context menu item | MANUAL |
| PB-013 | Confirm `pendingScan` is consumed after a successful popup scan | MANUAL |
| PB-014 | Confirm popup loading state | MANUAL |
| PB-015 | Confirm popup renders risk, reasoning, action, and scanned text | MANUAL |
| PB-018 | Stop backend and verify popup error state | MANUAL |
| PB-019 | Restart backend and scan again from popup | MANUAL |
| PB-025 | Reload extension in `chrome://extensions` and scan again | MANUAL |

## Manual Chrome procedure

1. Start the backend:

   ```powershell
   cd C:\jawbreaker-main\jawbreaker-main
   $env:JAWBREAKER_BACKEND = "heuristic"
   ..\venv\Scripts\python.exe app.py
   ```

2. Build the extension:

   ```powershell
   cd C:\jawbreaker-main\jawbreaker-main\phishbrake-extension
   npm run build
   ```

3. Open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**, and select:

   ```text
   C:\jawbreaker-main\jawbreaker-main\phishbrake-extension\dist
   ```

4. Open a normal webpage, select one of the phishing messages from the test plan, right-click, and choose **Scan with PhishBrake**.

5. Verify the popup shows the loading state and then the expected result.

6. Stop the Python server and repeat a scan to validate PB-018. Restart it and repeat to validate PB-019.

## Notes

- Suspicious URLs were analyzed as text only; no test link was opened.
- The initial Unicode failure was caused by the test shell sending non-UTF-8 JSON. The application passed when the request body was explicitly encoded as UTF-8.
- QR image decoding is covered by the web UI tests; the extension currently scans selected text and does not itself decode image QR codes.
