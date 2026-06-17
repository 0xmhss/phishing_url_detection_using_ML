/**
 * PhishGuard — frontend logic
 *
 * Handles:
 *  - Tab / panel navigation
 *  - URL scan form: calling /api/predict and rendering results
 *  - Session-scoped scan history (in-memory, lost on page refresh)
 *  - Feature grid and warning list rendering
 */

/* ── State ─────────────────────────────────────────────────────────────────── */

const scanHistory = [];   // { url, prediction, score, time }

/* ── Navigation ────────────────────────────────────────────────────────────── */

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    const panelId = "panel-" + btn.dataset.panel;

    document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));

    btn.classList.add("active");
    document.getElementById(panelId).classList.add("active");

    if (btn.dataset.panel === "history") renderHistory();
  });
});

/* ── Sample URLs ────────────────────────────────────────────────────────────── */

function loadSample(url) {
  document.getElementById("urlInput").value = url;
  document.getElementById("urlInput").focus();
}

/* ── Scan ───────────────────────────────────────────────────────────────────── */

document.getElementById("urlInput").addEventListener("keydown", e => {
  if (e.key === "Enter") scanURL();
});

async function scanURL() {
  const input = document.getElementById("urlInput");
  const url   = input.value.trim();
  if (!url) { input.focus(); return; }

  const btn = document.getElementById("scanBtn");
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Scanning';

  // Show result area immediately with loading state
  const resultArea = document.getElementById("resultArea");
  resultArea.style.display = "block";
  setVerdictLoading();

  try {
    const res  = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      setVerdictError(data.error || "Unexpected server error.");
      return;
    }

    renderResult(data);

    // Save to history
    const now = new Date();
    const timeStr = now.getHours().toString().padStart(2, "0") + ":" +
                    now.getMinutes().toString().padStart(2, "0");
    const score = Math.round((data.probabilities.phishing || 0) * 100);
    scanHistory.unshift({ url: data.url, prediction: data.prediction, score, time: timeStr });
    if (scanHistory.length > 50) scanHistory.pop();
    updateHistoryBadge();

  } catch (err) {
    setVerdictError("Could not reach the server. Make sure the Flask app is running.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg> Scan`;
  }
}

/* ── Verdict states ──────────────────────────────────────────────────────────── */

function setVerdictLoading() {
  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict-banner";

  document.getElementById("verdictIcon").className = "verdict-icon";
  setSvgIcon("verdictSvg", "loader");

  document.getElementById("verdictTitle").textContent = "Analyzing URL…";
  document.getElementById("verdictSub").textContent   = "Extracting features and running the model";
  document.getElementById("modeBadge").textContent    = "";

  document.getElementById("featuresGrid").innerHTML = "";
  document.getElementById("warningsList").innerHTML = "";
  animateConfidence(50, 50);
  animateNeedle(50);
  document.getElementById("riskScoreVal").textContent = "—";
}

function setVerdictError(msg) {
  document.getElementById("verdictBanner").className = "verdict-banner";
  document.getElementById("verdictIcon").className   = "verdict-icon";
  setSvgIcon("verdictSvg", "alert-triangle");
  document.getElementById("verdictTitle").textContent = "Scan failed";
  document.getElementById("verdictSub").textContent   = msg;
  document.getElementById("modeBadge").textContent    = "Error";
}

function renderResult(data) {
  const isPhishing  = data.prediction === "phishing";
  const dangerPct   = Math.round((data.probabilities.phishing || 0) * 100);
  const safePct     = 100 - dangerPct;
  const modeLabel   = data.mode === "ml" ? "ML Model" : "Heuristic";

  // Banner
  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict-banner " + (isPhishing ? "danger-state" : "safe-state");

  const icon = document.getElementById("verdictIcon");
  icon.className = "verdict-icon " + (isPhishing ? "danger-icon" : "safe-icon");
  setSvgIcon("verdictSvg", isPhishing ? "alert-triangle" : "shield-check");

  document.getElementById("verdictTitle").textContent =
    isPhishing ? "Warning: Phishing URL Detected!" : "URL Appears Safe";
  document.getElementById("verdictSub").textContent =
    isPhishing
      ? "This URL shows strong phishing indicators — do not click or share it."
      : "No significant phishing indicators were found in this URL.";
  document.getElementById("modeBadge").textContent = modeLabel;

  // Confidence bars + needle
  animateConfidence(safePct, dangerPct);
  animateNeedle(dangerPct);
  document.getElementById("riskScoreVal").textContent = dangerPct;

  // Feature grid
  renderFeatures(data.features);

  // Warnings
  renderWarnings(data.warnings);
}

/* ── Feature grid ────────────────────────────────────────────────────────────── */

// Map feature names → human labels and thresholds
const FEATURE_META = {
  url_length:           { label: "URL Length",        bad: v => v > 100, warn: v => v > 75 },
  domain_length:        { label: "Domain Len",        bad: () => false,  warn: () => false },
  has_ip:               { label: "Has IP",            bad: v => v,       warn: () => false },
  subdomain_count:      { label: "Subdomains",        bad: v => v > 3,   warn: v => v > 1 },
  digit_ratio:          { label: "Digit Ratio",       bad: v => v > 0.2, warn: v => v > 0.1 },
  special_ratio:        { label: "Special Ratio",     bad: v => v > 0.1, warn: v => v > 0.05 },
  entropy:              { label: "Entropy",           bad: v => v > 4.5, warn: v => v > 4.0 },
  suspicious_word_count:{ label: "Susp. Words",       bad: v => v > 1,   warn: v => v > 0 },
  longest_token_length: { label: "Longest Token",     bad: v => v > 25,  warn: v => v > 15 },
  url_token_count:      { label: "Token Count",       bad: () => false,  warn: () => false },
  has_punycode:         { label: "Punycode",          bad: v => v,       warn: () => false },
  has_port:             { label: "Has Port",          bad: v => v,       warn: () => false },
  path_depth:           { label: "Path Depth",        bad: v => v > 5,   warn: v => v > 3 },
  dot_count:            { label: "Dots",              bad: v => v > 6,   warn: v => v > 4 },
  hyphen_count:         { label: "Hyphens",           bad: v => v > 3,   warn: v => v > 2 },
  at_count:             { label: "@ Count",           bad: v => v > 0,   warn: () => false },
  question_count:       { label: "? Count",           bad: v => v > 2,   warn: v => v > 1 },
  equal_count:          { label: "= Count",           bad: v => v > 3,   warn: v => v > 2 },
  slash_count:          { label: "/ Count",           bad: () => false,  warn: () => false },
  tld:                  { label: "TLD",               bad: () => false,  warn: () => false },
  tld_length:           { label: "TLD Length",        bad: () => false,  warn: () => false },
};

function renderFeatures(features) {
  const grid = document.getElementById("featuresGrid");
  grid.innerHTML = "";

  Object.entries(features).forEach(([key, val]) => {
    const meta = FEATURE_META[key] || { label: key, bad: () => false, warn: () => false };
    let cls = "ok";
    if (meta.bad(val))  cls = "bad";
    else if (meta.warn(val)) cls = "warn";

    const displayVal = typeof val === "boolean"
      ? (val ? "YES" : "NO")
      : (typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(3)) : val);

    const item = document.createElement("div");
    item.className = "feat-item";
    item.innerHTML = `
      <div class="feat-name">${meta.label}</div>
      <div class="feat-val ${cls}">${displayVal}</div>
    `;
    grid.appendChild(item);
  });
}

/* ── Warnings ─────────────────────────────────────────────────────────────── */

function renderWarnings(warnings) {
  const list = document.getElementById("warningsList");
  if (!warnings || !warnings.length) {
    list.innerHTML = `
      <div class="warning-item">
        <div class="warning-icon ok-icon">${svgCheck()}</div>
        <span>No suspicious indicators detected in this URL.</span>
      </div>`;
    return;
  }

  list.innerHTML = warnings.map(w => `
    <div class="warning-item">
      <div class="warning-icon danger-icon">${svgWarn()}</div>
      <span>${escapeHtml(w)}</span>
    </div>`).join("");
}

/* ── Animations ─────────────────────────────────────────────────────────────── */

function animateConfidence(safePct, dangerPct) {
  document.getElementById("confSafe").style.width    = safePct + "%";
  document.getElementById("confDanger").style.width  = dangerPct + "%";
  document.getElementById("confSafeVal").textContent   = safePct + "%";
  document.getElementById("confDangerVal").textContent = dangerPct + "%";
}

function animateNeedle(pct) {
  document.getElementById("riskNeedle").style.left = Math.min(Math.max(pct, 2), 98) + "%";
}

/* ── History ─────────────────────────────────────────────────────────────────── */

function updateHistoryBadge() {
  const badge = document.getElementById("historyCount");
  if (scanHistory.length > 0) {
    badge.style.display = "inline-block";
    badge.textContent   = scanHistory.length;
  }
}

function renderHistory() {
  const list = document.getElementById("historyList");
  const meta = document.getElementById("historyMeta");

  if (!scanHistory.length) {
    list.innerHTML = `
      <div class="empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="12 8 12 12 14 14"/>
          <path d="M3.05 11a9 9 0 1 0 .5-4.5"/>
        </svg>
        <p>No scans yet. Go to Scan URL to get started.</p>
      </div>`;
    meta.textContent = "No scans yet";
    return;
  }

  const safe    = scanHistory.filter(h => h.prediction === "safe").length;
  const phishing = scanHistory.length - safe;
  meta.textContent = `${scanHistory.length} scans — ${safe} safe, ${phishing} phishing`;

  list.innerHTML = scanHistory.map(h => `
    <div class="hist-row" onclick="reloadScan('${escapeHtml(h.url)}')">
      <span class="hist-badge ${h.prediction === "safe" ? "safe-badge" : "danger-badge"}">
        ${h.prediction}
      </span>
      <span class="hist-url">${escapeHtml(h.url)}</span>
      <span class="hist-score">Risk: ${h.score}%</span>
      <span class="hist-time">${h.time}</span>
    </div>`).join("");
}

function reloadScan(url) {
  // Switch to scan panel
  document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelector('[data-panel="scan"]').classList.add("active");
  document.getElementById("panel-scan").classList.add("active");

  document.getElementById("urlInput").value = url;
  scanURL();
}

function clearHistory() {
  scanHistory.length = 0;
  const badge = document.getElementById("historyCount");
  badge.style.display = "none";
  renderHistory();
}

/* ── SVG icon helpers ────────────────────────────────────────────────────────── */

const SVG_PATHS = {
  "loader":          '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  "alert-triangle":  '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  "shield-check":    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
};

function setSvgIcon(elemId, name) {
  const el = document.getElementById(elemId);
  if (el) el.innerHTML = SVG_PATHS[name] || "";
}

function svgCheck() {
  return `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
}

function svgWarn() {
  return `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
}

/* ── Utility ─────────────────────────────────────────────────────────────────── */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
