from __future__ import annotations

from html import escape

from jawbreaker.schema import ScamAnalysis


DNA_LABELS = {
    "Impersonates": "Who they pretend to be",
    "Pressure": "How they pressure you",
    "Ask": "What they want",
    "Risk": "What could happen",
}

VERDICT_COPY = {
    "dangerous": ("CRITICAL: Scam Detected", "Result"),
    "suspicious": ("WARNING: Suspicious Pattern Found", "Result"),
    "needs_check": ("REVIEW: Verify Before Acting", "Result"),
    "safe": ("CLEAR: No Strong Scam Pattern", "Result"),
}

RISK_WINDOW_CLASS = {
    "dangerous": "risk-dangerous",
    "suspicious": "risk-suspicious",
    "needs_check": "risk-needs_check",
    "safe": "risk-safe",
}

RISK_BADGE = {
    "dangerous": "DANGER",
    "suspicious": "SUSPECT",
    "needs_check": "CHECK",
    "safe": "CLEAR",
}

SCAN_STEPS = [
    "Preparing the safety model...",
    "Reading the message...",
    "Checking scam pressure signals...",
    "Building the safest next step...",
    "Writing the report...",
]


def humanize(text: str) -> str:
    cleaned = str(text or "").replace("_", " ").replace("-", " ")
    words = " ".join(cleaned.split())
    if not words:
        return "Unknown"

    aliases = {
        "credential theft": "credential theft",
        "credential request": "credential request",
        "unknown urgent action": "urgent action",
        "unknown urgent payment": "urgent payment",
        "unknown website link": "suspicious link",
        "fake payment portal": "fake payment portal",
        "legitimate company": "legitimate company",
        "sense of urgency": "urgency",
        "click link and verify": "click a link and verify",
        "safe benign": "safe message",
        "needs check": "needs check",
    }
    lowered = words.lower()
    return aliases.get(lowered, words)


def build_copy_plan(message: str, analysis: ScamAnalysis) -> str:
    cleaned = " ".join(message.strip().split())
    if len(cleaned) > 500:
        cleaned = cleaned[:497].rstrip() + "..."

    risk = analysis.risk_level.replace("_", " ")
    scam_type = humanize(analysis.scam_type)
    risk_line = f"PhishBrake marked it as {risk}"
    if scam_type and scam_type != "none":
        risk_line += f" ({scam_type})"

    return (
        "Can you check this message with me before I do anything?\n\n"
        f"Message I received:\n\"{cleaned}\"\n\n"
        f"{risk_line}.\n"
        f"Safest next step: {analysis.safest_action}\n\n"
        "I have not clicked any links, replied, or sent anything."
    )


def render_window(title: str, body: str, class_name: str = "") -> str:
    classes = f"retro-window {class_name}".strip()
    return f"""
    <section class="{classes}">
      <div class="window-titlebar">
        <span>{escape(title)}</span>
      </div>
      <div class="window-body">
        {body}
      </div>
    </section>
    """


def render_analysis_html(message: str, analysis: ScamAnalysis) -> str:
    if not message.strip():
        return """
        <div class="home-stack">
          <section class="retro-window status-window">
            <div class="window-titlebar"><span>Ready to help</span></div>
            <div class="window-body status-body">
              <p class="standing-by">SYSTEM STANDING BY</p>
              <h2>PhishBrake is ready to shield your loved ones from digital fraud.</h2>
              <p>Paste any text message, email, or DM on the left. The local model will evaluate risk factors, unpack the scam strategy, and deliver a plain-English protection plan.</p>
            </div>
          </section>
          <section class="retro-window guide-window">
            <div class="window-titlebar"><span>How to use it</span></div>
            <div class="window-body guide-body">
              <p>1. Copy a text message from your phone or an email that feels off.</p>
              <p>2. Paste it into the input area on the left of this screen.</p>
              <p>3. Click RUN SCAM DETECTOR to analyze it with private local AI.</p>
            </div>
          </section>
        </div>
        """

    tactic_html = "".join(f"<span class='tactic'>{escape(humanize(tactic))}</span>" for tactic in analysis.tactics)
    dna_html = "".join(
        f"""
        <div class="dna-item">
          <div class="dna-label">{escape(DNA_LABELS.get(label, label))}</div>
          <div class="dna-value">{escape(humanize(value))}</div>
        </div>
        """
        for label, value in analysis.scam_dna.items()
    )
    memory_html = f"<p><strong>Memory:</strong> {escape(analysis.similar_memory)}</p>" if analysis.similar_memory else ""
    verdict_title, verdict_file = VERDICT_COPY[analysis.risk_level]
    verdict_subtitle = humanize(analysis.summary.replace("This looks dangerous: likely ", "Likely ").rstrip("."))
    risk_class = RISK_WINDOW_CLASS[analysis.risk_level]

    verdict = f"""
      <div class="verdict-header">
        <span class="verdict-icon" aria-hidden="true"></span>
        <div>
          <h2 class="verdict-title">{escape(verdict_title)}</h2>
          <p class="verdict-subtitle">{escape(verdict_subtitle)}.</p>
        </div>
      </div>
      {memory_html}
    """

    dna = f"""
      <div class="dna-grid">{dna_html}</div>
      <div class="tactics">{tactic_html or "<span class='tactic'>none found</span>"}</div>
    """

    copy_plan = build_copy_plan(message, analysis)
    copy_preview = escape(copy_plan)
    remedy = f"""
      <div class="remedy-copy">
        <p class="terminal-label">RECOMMENDED ACTION:</p>
        <p>{escape(humanize(analysis.safest_action))}</p>
      </div>
      <div class="copy-plan-inline">
        <div class="copy-plan-source">{copy_preview}</div>
        <button type="button" class="inline-copy-btn" data-copy="{escape(copy_plan, quote=True)}">COPY NOTE</button>
      </div>
    """

    return f"""
    <div class="report-stack">
      {render_window(verdict_file, verdict, f"verdict-window {risk_class}")}
      {render_window("How this scam works", dna, "dna-window")}
      {render_window("What to do next", remedy, "action-window")}
    </div>
    """


def render_scanning_html(active_step: int = 0, progress: int = 12) -> str:
    progress = max(0, min(100, progress))
    filled = max(1, round(progress / 100 * 12))
    bar = "█" * filled + "░" * (12 - filled)
    lines = []
    for index, step in enumerate(SCAN_STEPS):
        if index < active_step:
            class_name = "terminal-done"
            prefix = "✓"
        elif index == active_step:
            class_name = "terminal-active"
            prefix = ">"
        else:
            class_name = "terminal-muted"
            prefix = ">"
        lines.append(f'<p class="{class_name}">{prefix} {escape(step)}</p>')

    return """
    <section class="retro-window scanning-state terminal-window">
      <div class="window-titlebar"><span>Checking the message</span></div>
      <div class="window-body">
        <p class="terminal-progress">[%s] %s%% COMPLETE</p>
        <div class="terminal-log">
          %s
        </div>
      </div>
    </section>
    """ % (bar, progress, "\n".join(lines))


def render_memory_html(analysis: ScamAnalysis, memory: list[dict]) -> str:
    if not memory:
        return render_window(
            "Recent checks",
            "<p class='memory-empty'>No messages checked yet this session.</p>",
            "memory-card muted",
        )

    items = "".join(
        f"""
        <div class="memory-row">
          <span>{escape(item.get('summary', ''))}</span>
          <strong class="memory-badge">{escape(RISK_BADGE.get(item.get('risk_level', ''), item.get('risk_level', '')))}</strong>
        </div>
        """
        for item in memory[-5:]
    )
    return render_window("Recent checks", f"<p class='memory-title'>Session scam memory</p>{items}", "memory-card")
