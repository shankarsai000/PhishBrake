import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:7860/api/scan";

function normalizeResult(payload) {
  const analysis = payload?.analysis ?? payload ?? {};
  return {
    risk: analysis.risk_level ?? payload?.risk_level ?? "needs_check",
    scamType: analysis.scam_type ?? payload?.scam_type ?? "unknown",
    reasoning:
      analysis.summary ??
      payload?.summary ??
      analysis.reasoning ??
      payload?.reasoning ??
      "PhishBrake returned an analysis without a summary.",
    safeAction:
      analysis.safe_action ??
      analysis.safest_action ??
      payload?.safe_action ??
      "Verify through an official app, website, or known contact.",
    tactics: analysis.tactics ?? payload?.tactics ?? [],
    trustedMessage:
      analysis.trusted_person_message ??
      payload?.trusted_person_message ??
      payload?.copy_plan ??
      "Can you check this message for me before I respond or click anything?",
  };
}

export default function App() {
  const [inputText, setInputText] = useState("");
  const [scannedText, setScannedText] = useState("");
  const [copied, setCopied] = useState(false);
  const [state, setState] = useState({ status: "loading", result: null, error: "" });

  async function performScan(textToScan) {
    const text = textToScan.trim();
    if (!text) {
      setState({
        status: "empty",
        result: null,
        error: "Paste or select a message to analyze with PhishBrake.",
      });
      return;
    }

    setState({ status: "loading", result: null, error: "" });
    setScannedText(text);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        throw new Error(`PhishBrake API returned HTTP ${response.status}.`);
      }

      const payload = await response.json();
      if (typeof chrome !== "undefined" && chrome.storage?.local) {
        await chrome.storage.local.remove("pendingScan");
      }
      setState({ status: "complete", result: normalizeResult(payload), error: "" });
    } catch (error) {
      setState({
        status: "error",
        result: null,
        error:
          error instanceof TypeError
            ? "Cannot reach PhishBrake. Start the local app at 127.0.0.1:7860."
            : error.message,
      });
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function scanPendingText() {
      try {
        let text = "";
        if (typeof chrome !== "undefined" && chrome.storage?.local) {
          const stored = await chrome.storage.local.get("pendingScan");
          text = stored.pendingScan?.trim() ?? "";
        }
        if (cancelled) return;

        if (text) {
          performScan(text);
        } else {
          setState({
            status: "empty",
            result: null,
            error: "Paste a text, email, or DM below, or select text on any page and choose Scan with PhishBrake.",
          });
        }
      } catch (error) {
        if (!cancelled) {
          setState({
            status: "empty",
            result: null,
            error: "Paste a text, email, or DM below to check it for scams.",
          });
        }
      }
    }

    scanPendingText();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCopyNote = async () => {
    if (!state.result) return;
    const note = `Can you check this message with me?\n\nReceived: "${scannedText}"\n\nVerdict: PhishBrake marked it as ${state.result.risk.replaceAll("_", " ")}.\nWhy: ${state.result.reasoning}\nSafest action: ${state.result.safeAction}`;
    try {
      await navigator.clipboard.writeText(note);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Clipboard error", err);
    }
  };

  const handleReset = () => {
    setInputText("");
    setScannedText("");
    setState({
      status: "empty",
      result: null,
      error: "Paste a text, email, or DM below to analyze it.",
    });
  };

  return (
    <main className="popup">
      <header className="brand">
        <div className="shield" aria-hidden="true">||</div>
        <div>
          <h1>PhishBrake</h1>
          <p>Private local message check</p>
        </div>
      </header>

      {state.status === "loading" && (
        <section className="panel loading" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <strong>Checking message...</strong>
          <span>Your text stays on this device.</span>
        </section>
      )}

      {state.status === "empty" && (
        <section className="panel notice">
          <p>{state.error}</p>
          <textarea
            className="popup-input"
            placeholder="Paste text, email, or DM here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={4}
          />
          <button
            type="button"
            className="scan-btn"
            disabled={!inputText.trim()}
            onClick={() => performScan(inputText)}
          >
            Check Message
          </button>
        </section>
      )}

      {state.status === "error" && (
        <section className="panel error" role="alert">
          <strong>Scan unavailable</strong>
          <p>{state.error}</p>
          <button type="button" className="retry-btn" onClick={handleReset}>
            Try Again
          </button>
        </section>
      )}

      {state.status === "complete" && state.result && (
        <section className={`result result-${state.result.risk}`}>
          <div className="risk-header">
            <span className="risk-label">RISK LEVEL</span>
            <span className={`risk-badge badge-${state.result.risk}`}>
              {state.result.risk.replaceAll("_", " ").toUpperCase()}
            </span>
          </div>

          <h2>{state.result.risk.replaceAll("_", " ")}</h2>

          {state.result.tactics && state.result.tactics.length > 0 && (
            <div className="tactics-list">
              {state.result.tactics.map((tactic, idx) => (
                <span key={idx} className="tactic-tag">
                  {tactic.replaceAll("_", " ")}
                </span>
              ))}
            </div>
          )}

          <div className="copy-block">
            <span>WHY</span>
            <p>{state.result.reasoning}</p>
          </div>

          <div className="action-block">
            <span>SAFEST NEXT STEP</span>
            <p>{state.result.safeAction}</p>
          </div>

          <div className="button-group">
            <button type="button" className="copy-btn" onClick={handleCopyNote}>
              {copied ? "✓ COPIED NOTE" : "COPY WARNING NOTE"}
            </button>
            <button type="button" className="rescan-btn" onClick={handleReset}>
              SCAN ANOTHER
            </button>
          </div>

          {scannedText && (
            <details>
              <summary>Scanned text</summary>
              <p>{scannedText}</p>
            </details>
          )}
        </section>
      )}
    </main>
  );
}

