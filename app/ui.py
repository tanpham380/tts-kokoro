"""Modern Google Material 3 UI for the Kokoro-Vietnamese TTS Server playground & developer demo.

Modeled directly on the design system of QUAN-ASR (/app/ui.py).
Served at ``/``. Features:
- Google Material Design 3 design system (Light / Dark themes, Google colors, typography)
- Voice Studio & Text-to-Speech Playground (Single-pass StyleTTS 2 synthesis & streaming)
- 1-Click Vietnamese Demo Samples (including numbers & dates test "01 05 2001")
- 14 Vietnamese Voices Catalogue with gender filters and 1-click preview
- GPUStack Playground Advanced Parameters Accordion (task_type, instructions, reference_audio, ICL)
- Interactive Audio Player with Waveform Visualizer & Observability metrics (Latency ms, RTF, Duration)
- Observability Dashboard (Model status, Device, Realtime Factor, Metrics polling)
- API Integration Hub (Scalar /scalar, Swagger /docs, Python SDK, cURL, Node.js & GPUStack YAML generator)
"""

from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Kokoro-Vietnamese Studio · Tổng hợp giọng nói tiếng Việt chuẩn OpenAI & GPUStack</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%231a73e8'><path d='M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z'/></svg>" />

<!-- Google Fonts & Material Symbols -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Text:wght@400;500;700&family=JetBrains+Mono:wght@400;500&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />

<style>
  :root {
    --google-blue: #1a73e8;
    --google-blue-hover: #1557b0;
    --google-red: #ea4335;
    --google-yellow: #fbbc04;
    --google-green: #34a853;

    /* Light Theme (Google Standard) */
    --md-bg: #f8f9fa;
    --md-surface: #ffffff;
    --md-surface-variant: #f1f3f4;
    --md-surface-container: #ffffff;
    --md-surface-container-high: #e8eaed;
    --md-border: #dadce0;
    --md-border-subtle: #e8eaed;
    --md-text: #202124;
    --md-text-secondary: #5f6368;
    --md-text-muted: #80868b;
    --md-primary: #1a73e8;
    --md-primary-container: #e8f0fe;
    --md-on-primary-container: #174ea6;
    --md-shadow-1: 0 1px 3px 0 rgba(60,64,67,0.12), 0 1px 2px 0 rgba(60,64,67,0.24);
    --md-shadow-2: 0 2px 6px 2px rgba(60,64,67,0.15), 0 1px 2px 0 rgba(60,64,67,0.30);
    --md-code-bg: #f1f3f4;
  }

  [data-theme="dark"] {
    /* Dark Theme (Google AI Studio / Gemini Dark) */
    --md-bg: #131314;
    --md-surface: #1e1f20;
    --md-surface-variant: #282a2c;
    --md-surface-container: #1e1f20;
    --md-surface-container-high: #333537;
    --md-border: #3c4043;
    --md-border-subtle: #2d3135;
    --md-text: #e3e3e3;
    --md-text-secondary: #9aa0a6;
    --md-text-muted: #70757a;
    --md-primary: #8ab4f8;
    --md-primary-container: #174ea6;
    --md-on-primary-container: #d2e3fc;
    --md-shadow-1: 0 1px 3px 0 rgba(0,0,0,0.5), 0 1px 2px 0 rgba(0,0,0,0.4);
    --md-shadow-2: 0 4px 12px 0 rgba(0,0,0,0.6);
    --md-code-bg: #18191b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: "Google Sans Text", "Roboto", system-ui, -apple-system, sans-serif;
    background-color: var(--md-bg);
    color: var(--md-text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    transition: background-color 0.25s ease, color 0.25s ease;
    line-height: 1.5;
  }

  /* Material Symbols utility */
  .icon {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    display: inline-block;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    vertical-align: middle;
  }

  /* Header */
  header.google-app-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--md-surface);
    border-bottom: 1px solid var(--md-border);
    padding: 0 24px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: background 0.25s ease, border-color 0.25s ease;
  }

  .brand-group {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: inherit;
  }

  .logo-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #1a73e8, #4285f4);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    box-shadow: 0 2px 6px rgba(26,115,232,0.35);
  }

  .brand-title {
    font-family: 'Google Sans', sans-serif;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.2px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .brand-tag {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 9999px;
    background: var(--md-surface-variant);
    color: var(--md-text-secondary);
    border: 1px solid var(--md-border);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  /* Pills & Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    background: var(--md-surface-variant);
    color: var(--md-text-secondary);
    border: 1px solid var(--md-border);
    transition: all 0.2s ease;
  }

  .badge.status-ready {
    background: rgba(52, 168, 83, 0.12);
    border-color: rgba(52, 168, 83, 0.4);
    color: var(--google-green);
  }
  .badge.status-loading {
    background: rgba(251, 188, 4, 0.12);
    border-color: rgba(251, 188, 4, 0.4);
    color: var(--google-yellow);
  }
  .badge.status-error {
    background: rgba(234, 67, 53, 0.12);
    border-color: rgba(234, 67, 53, 0.4);
    color: var(--google-red);
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: currentColor;
    display: inline-block;
    animation: pulse 2s infinite ease-in-out;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(0.9); opacity: 0.7; }
    50% { transform: scale(1.3); opacity: 1; }
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .spin {
    animation: spin 1s linear infinite;
    display: inline-block;
  }

  /* Buttons */
  .nav-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    color: var(--md-text-secondary);
    background: transparent;
    border: 1px solid transparent;
    transition: all 0.2s ease;
  }
  .nav-btn:hover {
    background: var(--md-surface-variant);
    color: var(--md-text);
    border-color: var(--md-border);
  }

  .icon-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 1px solid var(--md-border);
    background: var(--md-surface);
    color: var(--md-text);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .icon-btn:hover {
    background: var(--md-surface-variant);
    border-color: var(--md-primary);
  }

  /* Main Container */
  main {
    flex: 1;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* Banners */
  .model-banner-container {
    width: 100%;
    box-sizing: border-box;
  }
  .model-banner-card {
    border-radius: 14px;
    padding: 16px 20px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
    box-shadow: var(--md-shadow-1);
    position: relative;
    overflow: hidden;
  }
  .banner-card-loading {
    background: rgba(26, 115, 232, 0.08);
    border: 1px solid rgba(26, 115, 232, 0.35);
  }
  .banner-card-error {
    background: rgba(234, 67, 53, 0.08);
    border: 1px solid rgba(234, 67, 53, 0.35);
  }
  .banner-spinner {
    width: 24px;
    height: 24px;
    border: 3px solid rgba(26, 115, 232, 0.25);
    border-top-color: var(--google-blue);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  /* Google Tabs */
  .google-tabs {
    display: flex;
    border-bottom: 1px solid var(--md-border);
    gap: 8px;
    overflow-x: auto;
  }
  .tab-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--md-text-secondary);
    font-family: 'Google Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s ease;
  }
  .tab-item:hover {
    color: var(--md-text);
    background: var(--md-surface-variant);
    border-radius: 8px 8px 0 0;
  }
  .tab-item.active {
    color: var(--md-primary);
    border-bottom-color: var(--md-primary);
    font-weight: 700;
  }

  .tab-content {
    display: none;
    flex-direction: column;
    gap: 20px;
  }
  .tab-content.active {
    display: flex;
  }

  /* Cards */
  .card {
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--md-shadow-1);
    transition: background 0.25s ease, border-color 0.25s ease;
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
  }
  .card-title {
    font-family: 'Google Sans', sans-serif;
    font-size: 17px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-subtitle {
    font-size: 13px;
    color: var(--md-text-secondary);
    margin-top: 2px;
  }

  /* Sample chips */
  .samples-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    padding: 12px 16px;
    background: var(--md-surface-variant);
    border-radius: 12px;
    border: 1px solid var(--md-border-subtle);
  }
  .samples-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--md-text-secondary);
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .sample-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 9999px;
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    color: var(--md-text);
    font-size: 12.5px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .sample-chip:hover {
    border-color: var(--md-primary);
    color: var(--md-primary);
    transform: translateY(-1px);
    box-shadow: var(--md-shadow-1);
  }
  .sample-chip.highlight {
    background: var(--md-primary-container);
    color: var(--md-on-primary-container);
    border-color: var(--md-primary);
    font-weight: 600;
  }

  /* Textarea */
  .textarea-box {
    width: 100%;
    min-height: 120px;
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid var(--md-border);
    background: var(--md-surface-variant);
    color: var(--md-text);
    font-family: inherit;
    font-size: 14.5px;
    line-height: 1.6;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .textarea-box:focus {
    border-color: var(--md-primary);
    box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.2);
  }

  /* Form Grid */
  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 16px;
  }
  .field-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .field-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--md-text-secondary);
    display: flex;
    justify-content: space-between;
  }
  .google-input {
    height: 42px;
    padding: 0 12px;
    border-radius: 8px;
    border: 1px solid var(--md-border);
    background: var(--md-surface);
    color: var(--md-text);
    font-family: inherit;
    font-size: 13.5px;
    outline: none;
    transition: all 0.2s ease;
  }
  .google-input:focus {
    border-color: var(--md-primary);
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
  }

  /* Accordion */
  .accordion-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    background: none;
    border: 1px solid var(--md-border);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--md-primary);
    font-family: 'Google Sans', sans-serif;
    font-size: 13.5px;
    font-weight: 600;
    cursor: pointer;
    margin-top: 16px;
    width: 100%;
    text-align: left;
    transition: all 0.2s ease;
  }
  .accordion-toggle:hover {
    background: var(--md-surface-variant);
  }
  .accordion-content {
    display: none;
    padding: 16px;
    border: 1px solid var(--md-border);
    border-top: none;
    border-radius: 0 0 10px 10px;
    background: var(--md-surface-variant);
    margin-top: -6px;
  }
  .accordion-content.open {
    display: block;
  }

  /* Action Buttons */
  .btn-row {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    margin-top: 20px;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 9999px;
    font-family: 'Google Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.2s ease;
  }
  .btn-primary {
    background: var(--md-primary);
    color: #ffffff;
    box-shadow: 0 1px 3px rgba(26,115,232,0.3);
  }
  [data-theme="dark"] .btn-primary {
    background: var(--md-primary);
    color: #131314;
  }
  .btn-primary:hover:not(:disabled) {
    background: var(--google-blue-hover);
    color: #ffffff;
    box-shadow: var(--md-shadow-2);
    transform: translateY(-1px);
  }
  .btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .btn-outlined {
    background: transparent;
    color: var(--md-text);
    border-color: var(--md-border);
  }
  .btn-outlined:hover {
    background: var(--md-surface-variant);
  }

  /* Audio Result Box */
  .result-card {
    display: none;
    flex-direction: column;
    gap: 16px;
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: var(--md-shadow-1);
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .player-wrapper {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  audio {
    flex: 1;
    min-width: 280px;
    height: 44px;
    border-radius: 22px;
    outline: none;
  }
  .metrics-pill-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 12.5px;
    color: var(--md-text-secondary);
  }
  .metric-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    background: var(--md-surface-variant);
    border-radius: 6px;
    border: 1px solid var(--md-border-subtle);
  }

  /* Tab 2: Voice Grid */
  .voice-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
  }
  .voice-card {
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 14px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    box-shadow: var(--md-shadow-1);
    transition: all 0.2s ease;
  }
  .voice-card:hover {
    border-color: var(--md-primary);
    transform: translateY(-2px);
    box-shadow: var(--md-shadow-2);
  }
  .voice-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .voice-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    color: #fff;
    font-weight: 700;
  }
  .avatar-female {
    background: linear-gradient(135deg, #e91e63, #ff4081);
  }
  .avatar-male {
    background: linear-gradient(135deg, #1a73e8, #00bcd4);
  }
  .avatar-neutral {
    background: linear-gradient(135deg, #9c27b0, #673ab7);
  }
  .voice-info h4 {
    font-family: 'Google Sans', sans-serif;
    font-size: 15px;
    font-weight: 600;
  }
  .voice-info span {
    font-size: 12px;
    color: var(--md-text-secondary);
  }
  .voice-tags {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }

  /* Code Hub */
  .code-hub-tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1px solid var(--md-border);
    padding-bottom: 8px;
    margin-bottom: 12px;
  }
  .code-hub-btn {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    border: none;
    background: transparent;
    color: var(--md-text-secondary);
    cursor: pointer;
  }
  .code-hub-btn.active {
    background: var(--md-surface-variant);
    color: var(--md-primary);
    font-weight: 700;
  }
  .code-box {
    position: relative;
    background: var(--md-code-bg);
    border: 1px solid var(--md-border);
    border-radius: 12px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    overflow-x: auto;
    color: var(--md-text);
  }
  .copy-code-btn {
    position: absolute;
    top: 12px;
    right: 12px;
    padding: 6px 10px;
    border-radius: 6px;
    border: 1px solid var(--md-border);
    background: var(--md-surface);
    color: var(--md-text-secondary);
    font-size: 12px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: all 0.2s ease;
  }
  .copy-code-btn:hover {
    border-color: var(--md-primary);
    color: var(--md-primary);
  }

  /* Footer */
  footer {
    border-top: 1px solid var(--md-border);
    padding: 20px 24px;
    text-align: center;
    font-size: 13px;
    color: var(--md-text-muted);
    background: var(--md-surface);
    margin-top: auto;
  }
</style>
</head>
<body>

<!-- Top App Bar (Google Standard) -->
<header class="google-app-bar">
  <a href="/" class="brand-group">
    <div class="logo-box">
      <span class="icon" style="font-size:24px;">record_voice_over</span>
    </div>
    <div>
      <div class="brand-title">
        Kokoro-Vietnamese TTS <span class="brand-tag">StyleTTS 2 · 82M</span>
      </div>
    </div>
  </a>

  <div class="header-actions">
    <!-- Server Status Chip -->
    <div class="badge status-loading" id="healthPill" title="Trạng thái mô hình & accelerator">
      <span class="pulse-dot"></span>
      <span id="healthText">Đang kết nối...</span>
    </div>

    <!-- API Docs Links -->
    <a href="/docs" target="_blank" class="nav-btn" title="Mở Swagger UI (OpenAPI tiêu chuẩn)">
      <span class="icon">api</span> Swagger
    </a>
    <a href="/metrics" target="_blank" class="nav-btn" title="Xem Prometheus / Health metrics">
      <span class="icon">monitoring</span> Metrics
    </a>

    <!-- Theme Toggle -->
    <button class="icon-btn" id="themeToggle" title="Chuyển chế độ sáng / tối">
      <span class="icon" id="themeIcon">light_mode</span>
    </button>
  </div>
</header>

<main>
  <!-- Model Loading Banner -->
  <div class="model-banner-container" id="modelLoadingBanner" style="display:none;">
    <div class="model-banner-card banner-card-loading">
      <div class="banner-spinner"></div>
      <div>
        <div style="font-weight:700; font-size:15px; margin-bottom:4px;">Mô hình Kokoro-Vietnamese đang khởi tạo...</div>
        <div style="font-size:13px; color:var(--md-text-secondary);">Hệ thống đang nạp trọng số và giọng đọc. Quá trình này chỉ mất 1-3 giây khi nạp từ cache cục bộ GPUStack.</div>
      </div>
    </div>
  </div>

  <!-- Google Standard Navigation Tabs -->
  <nav class="google-tabs" role="tablist">
    <button class="tab-item active" data-tab="studioTab">
      <span class="icon">mic</span> Phòng thu Tổng hợp (Studio)
    </button>
    <button class="tab-item" data-tab="voicesTab">
      <span class="icon">group</span> 14 Giọng đọc Tiếng Việt
    </button>
    <button class="tab-item" data-tab="metricsTab">
      <span class="icon">monitoring</span> Chỉ số & Hiệu năng
    </button>
    <button class="tab-item" data-tab="apiTab">
      <span class="icon">code</span> Hướng dẫn Tích hợp & Code Hub
    </button>
  </nav>

  <!-- ================= TAB 1: STUDIO PLAYGROUND ================= -->
  <section id="studioTab" class="tab-content active">
    <!-- Sample Chips Picker -->
    <div class="samples-banner">
      <span class="samples-label"><span class="icon" style="font-size:16px;">library_music</span> Thử nhanh mẫu câu:</span>
      <button class="sample-chip" onclick="setSample('Xin chào quý khách! Cảm ơn quý khách đã tin tưởng và sử dụng dịch vụ của chúng tôi.')">
        <span class="icon" style="font-size:16px;">chat</span> Chào khách hàng
      </button>
      <button class="sample-chip" onclick="setSample('Dự báo thời tiết hôm nay tại Hà Nội và Thành phố Hồ Chí Minh có mưa rào vài nơi, chiều tối trời mát mẻ.')">
        <span class="icon" style="font-size:16px;">wb_sunny</span> Thời tiết TP.HCM
      </button>
      <button class="sample-chip highlight" onclick="setSample('Tôi sinh ngày 01 05 2001, mã hồ sơ số 01 05 2001 tại TP.HCM.')">
        <span class="icon" style="font-size:16px;">calendar_month</span> Thử đọc số: "01 05 2001"
      </button>
      <button class="sample-chip" onclick="setSample('Năm 2024, kinh tế Việt Nam đạt mức tăng trưởng 6.5%, khoảng cách di chuyển từ trung tâm là 15km với giá 100k.')">
        <span class="icon" style="font-size:16px;">trending_up</span> Số đo & Viết tắt
      </button>
    </div>

    <!-- Main Generation Card -->
    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title"><span class="icon">edit_note</span> Văn bản tiếng Việt cần tổng hợp</h2>
          <p class="card-subtitle">Hỗ trợ đầy đủ số ngày tháng, chữ cái đơn lẻ (Issue #4), từ viết tắt và nối câu mượt mà (Issue #3)</p>
        </div>
        <div class="badge" id="charWordCountBadge">0 ký tự · 0 từ</div>
      </div>

      <textarea id="textInput" class="textarea-box" placeholder="Nhập văn bản tiếng Việt để đọc...">Tôi sinh ngày 01 05 2001, mã hồ sơ số 01 05 2001 tại TP.HCM.</textarea>

      <!-- Control Settings Grid -->
      <div class="form-grid">
        <div class="field-group">
          <label class="field-label" for="voiceSelect">Giọng đọc (Voice)</label>
          <select id="voiceSelect" class="google-input"></select>
        </div>

        <div class="field-group">
          <label class="field-label" for="formatSelect">Định dạng (Format)</label>
          <select id="formatSelect" class="google-input">
            <option value="mp3" selected>mp3 (Khuyên dùng, kích thước nhỏ)</option>
            <option value="wav">wav (Lossless, chất lượng cao nhất)</option>
            <option value="opus">opus (Chuẩn nén WebRTC / VoIP)</option>
            <option value="aac">aac (Chuẩn Apple / M4A)</option>
            <option value="flac">flac (Lossless nén)</option>
            <option value="pcm">pcm (Raw 16-bit 24kHz LE)</option>
          </select>
        </div>

        <div class="field-group">
          <label class="field-label" for="streamSelect">Chế độ truyền phát (Stream)</label>
          <select id="streamSelect" class="google-input">
            <option value="false" selected>Tải toàn bộ (Standard Batch)</option>
            <option value="audio">Audio chunked stream</option>
            <option value="sse">OpenAI SSE (speech.audio.delta)</option>
          </select>
        </div>

        <div class="field-group">
          <label class="field-label" for="speedRange">
            <span>Tốc độ đọc (Speed)</span>
            <span id="speedVal" style="color:var(--md-primary); font-weight:700;">1.00x</span>
          </label>
          <input type="range" id="speedRange" min="0.5" max="2.0" step="0.05" value="1.0" style="margin-top:10px; accent-color:var(--md-primary); cursor:pointer;">
        </div>
      </div>

      <!-- Advanced GPUStack Accordion -->
      <button class="accordion-toggle" id="advancedToggle">
        <span class="icon" id="advancedIcon">tune</span> Tham số Nâng cao (GPUStack Playground Compatible)
      </button>

      <div class="accordion-content" id="advancedBox">
        <div class="form-grid" style="margin-top:0;">
          <div class="field-group">
            <label class="field-label" for="taskTypeInput">Task Type</label>
            <input type="text" id="taskTypeInput" class="google-input" value="CustomVoice" placeholder="e.g. CustomVoice, TTS">
          </div>
          <div class="field-group">
            <label class="field-label" for="langSelect">Ngôn ngữ (Language)</label>
            <select id="langSelect" class="google-input">
              <option value="vi" selected>vi (Tiếng Việt)</option>
              <option value="en">en (English)</option>
              <option value="auto">auto (Tự động nhận diện)</option>
            </select>
          </div>
          <div class="field-group">
            <label class="field-label" for="maxTokensInput">Max New Tokens</label>
            <input type="number" id="maxTokensInput" class="google-input" value="2048" min="64" max="8192">
          </div>
          <div class="field-group" style="grid-column: 1 / -1;">
            <label class="field-label" for="instructionsInput">Chỉ dẫn phong cách (Instructions)</label>
            <input type="text" id="instructionsInput" class="google-input" placeholder="e.g. Giọng ấm áp, truyền cảm, nhịp nhàng...">
          </div>
          <div class="field-group" style="grid-column: 1 / -1;">
            <label class="field-label" for="refAudioInput">Reference Audio (File path / URL / Base64)</label>
            <input type="text" id="refAudioInput" class="google-input" placeholder="Đường dẫn file .wav / .pt hoặc Base64 audio">
          </div>
          <div class="field-group" style="grid-column: 1 / -1;">
            <label class="field-label" for="refTextInput">Transcript of Reference Audio (ICL mode)</label>
            <input type="text" id="refTextInput" class="google-input" placeholder="Lời thoại văn bản tương ứng với audio mẫu (chế độ ICL)">
          </div>
          <div class="field-group" style="grid-column: 1 / -1; display:flex; flex-direction:row; align-items:center; gap:8px;">
            <input type="checkbox" id="speakerEmbedOnlyCheck" style="width:16px; height:16px; accent-color:var(--md-primary);">
            <label for="speakerEmbedOnlyCheck" style="font-size:13.5px; cursor:pointer;">Use Speaker Embedding Only (no ICL)</label>
          </div>
        </div>
      </div>

      <!-- Action Row -->
      <div class="btn-row">
        <button class="btn btn-outlined" onclick="clearText()">
          <span class="icon">clear_all</span> Xóa văn bản
        </button>
        <button class="btn btn-primary" id="generateBtn" onclick="synthesize()">
          <span class="icon" id="btnIcon">play_arrow</span>
          <span id="btnText">Tổng Hợp Giọng Nói</span>
        </button>
      </div>
    </div>

    <!-- Synthesis Output Card -->
    <div class="result-card" id="resultCard">
      <div class="card-header" style="margin-bottom:8px;">
        <div class="card-title">
          <span class="icon" style="color:var(--google-green);">check_circle</span> Kết quả âm thanh đã tạo
        </div>
        <div class="metrics-pill-row">
          <div class="metric-tag" id="metricFormat">MP3 · 24kHz</div>
          <div class="metric-tag" id="metricLatency">Thời gian: -</div>
          <div class="metric-tag" id="metricRTF">RTF: -</div>
        </div>
      </div>

      <!-- Live Streaming Indicator -->
      <div id="streamingIndicator" style="display:none; align-items:center; justify-content:space-between; background:var(--md-surface-variant); padding:10px 14px; border-radius:10px; border:1px solid var(--md-border); margin-bottom:12px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="pulse-dot" style="background-color:var(--google-green);"></span>
          <span id="streamingStatusText" style="font-size:13px; font-weight:600; color:var(--google-green);">Đang phát luồng âm thanh trực tiếp...</span>
        </div>
        <div style="font-size:12px; color:var(--md-text-secondary); font-family:'JetBrains Mono', monospace;" id="streamingChunkStats">0 câu · 0ms TTFA</div>
      </div>

      <div class="player-wrapper">
        <audio id="audioPlayer" controls></audio>
        <a id="downloadBtn" class="btn btn-outlined" download="speech.mp3" style="text-decoration:none;">
          <span class="icon">download</span> Tải tệp âm thanh
        </a>
      </div>
    </div>
  </section>

  <!-- ================= TAB 2: 14 VOICES CATALOGUE ================= -->
  <section id="voicesTab" class="tab-content">
    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title"><span class="icon">record_voice_over</span> Bộ sưu tập 14 Giọng đọc Tiếng Việt</h2>
          <p class="card-subtitle">Mỗi giọng đọc được trích xuất từ Style Vector (.pt ~200KB), chia sẻ 1 backbone StyleTTS2 duy nhất</p>
        </div>
        <div class="samples-banner" style="padding:4px 10px; margin:0;">
          <button class="sample-chip active" onclick="filterVoices('all')">Tất cả (14)</button>
          <button class="sample-chip" onclick="filterVoices('female')">♀ Giọng Nữ (6)</button>
          <button class="sample-chip" onclick="filterVoices('male')">♂ Giọng Nam (7)</button>
          <button class="sample-chip" onclick="filterVoices('other')">Sách nói (1)</button>
        </div>
      </div>

      <div class="voice-grid" id="voiceGridContainer">
        <!-- Populated via JS -->
      </div>
    </div>
  </section>

  <!-- ================= TAB 3: METRICS & OBSERVABILITY ================= -->
  <section id="metricsTab" class="tab-content">
    <div class="form-grid" style="margin-top:0;">
      <div class="card">
        <div class="card-title"><span class="icon" style="color:var(--google-blue);">psychology</span> Mô hình AI</div>
        <div style="font-size:24px; font-weight:700; margin:12px 0 4px 0;" id="metricModelName">Kokoro-Vietnamese</div>
        <div style="font-size:13px; color:var(--md-text-secondary);" id="metricDevice">Thiết bị: CPU</div>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon" style="color:var(--google-green);">speed</span> Real-time Factor (RTF)</div>
        <div style="font-size:24px; font-weight:700; margin:12px 0 4px 0;" id="metricRTFAvg">< 0.10x</div>
        <div style="font-size:13px; color:var(--md-text-secondary);">Nhanh hơn thời gian thực > 10 lần</div>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon" style="color:var(--google-yellow);">graphic_eq</span> Tổng thời lượng âm thanh</div>
        <div style="font-size:24px; font-weight:700; margin:12px 0 4px 0;" id="metricTotalAudio">0.0 giây</div>
        <div style="font-size:13px; color:var(--md-text-secondary);" id="metricTotalReqs">0 yêu cầu đã xử lý</div>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon" style="color:#ab47bc;">queue</span> Hàng đợi & Trạng thái</div>
        <div style="font-size:24px; font-weight:700; margin:12px 0 4px 0;" id="metricQueueStatus">0 đang chờ</div>
        <div style="font-size:13px; color:var(--md-text-secondary);">Tối đa 256 slot hàng đợi</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h3 class="card-title"><span class="icon">dns</span> Thông số chi tiết máy chủ</h3>
        <button class="btn btn-outlined" onclick="loadMetrics()" style="padding:6px 14px;">
          <span class="icon">refresh</span> Làm mới
        </button>
      </div>
      <pre class="code-box" id="metricsRawJson" style="max-height:300px;">Đang tải...</pre>
    </div>
  </section>

  <!-- ================= TAB 4: API & CODE HUB ================= -->
  <section id="apiTab" class="tab-content">
    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title"><span class="icon">integration_instructions</span> Tích hợp chuẩn OpenAI & GPUStack</h2>
          <p class="card-subtitle">Sao chép nhanh mã nguồn sẵn sàng chạy với OpenAI Python SDK, cURL, hoặc cấu hình GPUStack</p>
        </div>
      </div>

      <div class="code-hub-tabs">
        <button class="code-hub-btn active" onclick="switchCodeLang('python')">Python (OpenAI SDK)</button>
        <button class="code-hub-btn" onclick="switchCodeLang('curl')">cURL (REST API)</button>
        <button class="code-hub-btn" onclick="switchCodeLang('js')">Node.js / JS</button>
        <button class="code-hub-btn" onclick="switchCodeLang('gpustack')">GPUStack YAML</button>
      </div>

      <div style="position:relative;">
        <pre class="code-box" id="codeSnippetBox">Đang chuẩn bị code...</pre>
        <button class="copy-code-btn" onclick="copyCode()"><span class="icon" style="font-size:16px;">content_copy</span> Sao chép</button>
      </div>
    </div>
  </section>
</main>

<footer>
  <div>Kokoro-Vietnamese TTS Server · OpenAI Compatible API · GPUStack Custom Backend Ready</div>
  <div style="margin-top:4px;">Phát triển trên nền tảng StyleTTS 2 & Kokoro-82M · Hỗ trợ tiếng Việt tự nhiên với vig2p</div>
</footer>

<script>
// State
let allVoices = [];
let currentVoiceFilter = 'all';
let currentCodeLang = 'python';

// Elements
const textInput = document.getElementById('textInput');
const voiceSelect = document.getElementById('voiceSelect');
const formatSelect = document.getElementById('formatSelect');
const streamSelect = document.getElementById('streamSelect');
const speedRange = document.getElementById('speedRange');
const speedVal = document.getElementById('speedVal');
const generateBtn = document.getElementById('generateBtn');
const btnIcon = document.getElementById('btnIcon');
const btnText = document.getElementById('btnText');
const resultCard = document.getElementById('resultCard');
const audioPlayer = document.getElementById('audioPlayer');
const downloadBtn = document.getElementById('downloadBtn');
const metricFormat = document.getElementById('metricFormat');
const metricLatency = document.getElementById('metricLatency');
const metricRTF = document.getElementById('metricRTF');
const charWordCountBadge = document.getElementById('charWordCountBadge');
const healthPill = document.getElementById('healthPill');
const healthText = document.getElementById('healthText');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const advancedToggle = document.getElementById('advancedToggle');
const advancedBox = document.getElementById('advancedBox');

// Theme Switcher
function initTheme() {
  const saved = localStorage.getItem('kokoro_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  themeIcon.textContent = saved === 'dark' ? 'light_mode' : 'dark_mode';
}
themeToggle.addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('kokoro_theme', next);
  themeIcon.textContent = next === 'dark' ? 'light_mode' : 'dark_mode';
});
initTheme();

// Tabs Navigation
document.querySelectorAll('.tab-item').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const tabId = btn.getAttribute('data-tab');
    document.getElementById(tabId).classList.add('active');
    if (tabId === 'metricsTab') loadMetrics();
    if (tabId === 'apiTab') updateCodeSnippet();
  });
});

// Accordion Toggle
advancedToggle.addEventListener('click', () => {
  advancedBox.classList.toggle('open');
});

// Speed Range Slider
speedRange.addEventListener('input', () => {
  speedVal.textContent = parseFloat(speedRange.value).toFixed(2) + 'x';
  updateCodeSnippet();
});

// Text Area Counts
function updateCounts() {
  const text = textInput.value;
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  charWordCountBadge.textContent = `${chars} ký tự · ${words} từ`;
  updateCodeSnippet();
}
textInput.addEventListener('input', updateCounts);
updateCounts();

function setSample(text) {
  textInput.value = text;
  updateCounts();
  textInput.focus();
}

function clearText() {
  textInput.value = '';
  updateCounts();
  textInput.focus();
}

// Fetch Health and Voices
async function initData() {
  try {
    const [hRes, vRes] = await Promise.all([
      fetch('/health'),
      fetch('/v1/voices')
    ]);
    if (hRes.ok) {
      const hData = await hRes.json();
      if (hData.loaded) {
        healthPill.className = 'badge status-ready';
        healthText.textContent = `Model Sẵn sàng (${(hData.device || 'cpu').toUpperCase()})`;
        document.getElementById('modelLoadingBanner').style.display = 'none';
      } else if (hData.loading) {
        healthPill.className = 'badge status-loading';
        healthText.textContent = 'Đang tải model...';
        document.getElementById('modelLoadingBanner').style.display = 'block';
      } else {
        healthPill.className = 'badge status-error';
        healthText.textContent = 'Lỗi nạp model';
      }
      document.getElementById('metricModelName').textContent = hData.model || 'Kokoro-Vietnamese';
      document.getElementById('metricDevice').textContent = `Thiết bị: ${(hData.device || 'cpu').toUpperCase()}`;
    }
    if (vRes.ok) {
      const { data } = await vRes.json();
      allVoices = data;
      populateVoiceSelect();
      renderVoiceGrid();
    }
  } catch (err) {
    healthPill.className = 'badge status-error';
    healthText.textContent = 'Server Offline';
  }
}

function populateVoiceSelect() {
  voiceSelect.innerHTML = allVoices.map(v => {
    const genderMark = v.gender === 'female' ? '♀ Nữ' : (v.gender === 'male' ? '♂ Nam' : 'Trực tiếp');
    return `<option value="${v.id}" ${v.id === 'diem_trinh' ? 'selected' : ''}>${v.name} (${genderMark}) - ${v.style}</option>`;
  }).join('');
}

function renderVoiceGrid() {
  const container = document.getElementById('voiceGridContainer');
  let list = allVoices;
  if (currentVoiceFilter !== 'all') {
    if (currentVoiceFilter === 'other') list = allVoices.filter(v => v.gender !== 'female' && v.gender !== 'male');
    else list = allVoices.filter(v => v.gender === currentVoiceFilter);
  }
  container.innerHTML = list.map(v => {
    const avatarClass = v.gender === 'female' ? 'avatar-female' : (v.gender === 'male' ? 'avatar-male' : 'avatar-neutral');
    const genderIcon = v.gender === 'female' ? 'female' : (v.gender === 'male' ? 'male' : 'record_voice_over');
    return `
      <div class="voice-card">
        <div class="voice-card-header">
          <div class="voice-avatar ${avatarClass}">
            <span class="icon">${genderIcon}</span>
          </div>
          <div class="voice-info">
            <h4>${v.name}</h4>
            <span>${v.voice_id}</span>
          </div>
        </div>
        <div style="font-size:13px; color:var(--md-text-secondary);">${v.style}</div>
        <div class="voice-tags">
          <span class="badge">Alias: ${v.id}</span>
        </div>
        <button class="btn btn-outlined" style="margin-top:auto; padding:6px 12px; font-size:12.5px;" onclick="chooseVoiceAndTest('${v.id}')">
          <span class="icon" style="font-size:16px;">play_circle</span> Chọn giọng này
        </button>
      </div>
    `;
  }).join('');
}

function filterVoices(filter) {
  currentVoiceFilter = filter;
  renderVoiceGrid();
}

function chooseVoiceAndTest(voiceId) {
  voiceSelect.value = voiceId;
  document.querySelectorAll('.tab-item')[0].click();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

let currentAudioCtx = null;

// Synthesis
async function synthesize() {
  const text = textInput.value.trim();
  if (!text) {
    alert('Vui lòng nhập văn bản cần tổng hợp.');
    return;
  }

  // Stop any previous playing audio context
  if (currentAudioCtx) {
    try { currentAudioCtx.close(); } catch(e) {}
    currentAudioCtx = null;
  }
  audioPlayer.pause();

  generateBtn.disabled = true;
  btnIcon.textContent = 'hourglass_empty';
  btnIcon.classList.add('spin');
  btnText.textContent = 'Đang xử lý âm thanh...';

  const format = formatSelect.value;
  const streamMode = streamSelect.value;
  const speed = parseFloat(speedRange.value);

  const payload = {
    model: 'Kokoro-Vietnamese',
    input: text,
    voice: voiceSelect.value,
    response_format: format,
    speed: speed
  };

  if (streamMode !== 'false') payload.stream_format = streamMode;

  // GPUStack Advanced
  const taskType = document.getElementById('taskTypeInput').value.trim();
  const lang = document.getElementById('langSelect').value;
  const maxTokens = document.getElementById('maxTokensInput').value;
  const instructions = document.getElementById('instructionsInput').value.trim();
  const refAudio = document.getElementById('refAudioInput').value.trim();
  const refText = document.getElementById('refTextInput').value.trim();
  const speakerEmbedOnly = document.getElementById('speakerEmbedOnlyCheck').checked;

  if (taskType) payload.task_type = taskType;
  if (lang) payload.language = lang;
  if (maxTokens) payload.max_new_tokens = parseInt(maxTokens, 10);
  if (instructions) payload.instructions = instructions;
  if (refAudio) payload.reference_audio = refAudio;
  if (refText) payload.transcript_of_reference_audio = refText;
  if (speakerEmbedOnly) payload.use_speaker_embedding_only = true;

  const streamingIndicator = document.getElementById('streamingIndicator');
  const streamingStatusText = document.getElementById('streamingStatusText');
  const streamingChunkStats = document.getElementById('streamingChunkStats');

  const t0 = performance.now();
  try {
    const res = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
      throw new Error(err.error?.message || 'Lỗi khi tổng hợp giọng nói');
    }

    resultCard.style.display = 'flex';

    if (streamMode === 'sse') {
      // ---------------------------------------------------- Real-time SSE Stream
      streamingIndicator.style.display = 'flex';
      streamingStatusText.textContent = 'Đang nhận luồng sự kiện SSE...';
      streamingChunkStats.textContent = '0 câu · 0ms TTFA';

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContextClass();
      currentAudioCtx = audioCtx;
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      let nextStartTime = audioCtx.currentTime;
      let chunkCount = 0;
      let ttfa = 0;
      let totalAudioDuration = 0;
      const audioBlobs = [];

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep last incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') continue;

          try {
            const evt = JSON.parse(dataStr);
            if (evt.type === 'speech.audio.delta' && evt.audio) {
              chunkCount++;
              if (chunkCount === 1) {
                ttfa = Math.round(performance.now() - t0);
              }
              streamingStatusText.textContent = `🟢 Đang phát câu ${chunkCount} trực tiếp...`;
              streamingChunkStats.textContent = `${chunkCount} câu · TTFA: ${ttfa}ms`;

              // Decode base64 to ArrayBuffer
              const binaryStr = atob(evt.audio);
              const bytes = new Uint8Array(binaryStr.length);
              for (let i = 0; i < binaryStr.length; i++) {
                bytes[i] = binaryStr.charCodeAt(i);
              }
              audioBlobs.push(bytes.buffer);

              // Schedule audio chunk in Web Audio API
              const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer.slice(0));
              totalAudioDuration += audioBuffer.duration;

              const source = audioCtx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(audioCtx.destination);
              const startAt = Math.max(audioCtx.currentTime, nextStartTime);
              source.start(startAt);
              nextStartTime = startAt + audioBuffer.duration;
            } else if (evt.type === 'speech.audio.done') {
              streamingStatusText.textContent = `✅ Hoàn thành luồng (${chunkCount} câu)`;
            }
          } catch (e) {
            console.error('SSE parse error:', e);
          }
        }
      }

      const totalElapsed = Math.round(performance.now() - t0);
      metricFormat.textContent = `SSE STREAM · 24kHz`;
      metricLatency.textContent = `TTFA: ${ttfa}ms · Tổng: ${totalElapsed}ms`;
      const rtf = totalAudioDuration > 0 ? (totalElapsed / 1000 / totalAudioDuration).toFixed(2) : '0.00';
      metricRTF.textContent = `RTF: ${rtf}x (${totalAudioDuration.toFixed(1)}s audio)`;

      if (audioBlobs.length > 0) {
        const fullBlob = new Blob(audioBlobs, { type: 'audio/wav' });
        const url = URL.createObjectURL(fullBlob);
        audioPlayer.src = url;
        downloadBtn.href = url;
        downloadBtn.download = `kokoro_${voiceSelect.value}_${Date.now()}.wav`;
      }
    } else if (streamMode === 'audio') {
      // ---------------------------------------------------- Chunked Raw Audio
      streamingIndicator.style.display = 'flex';
      streamingStatusText.textContent = 'Đang nhận luồng byte âm thanh chunked...';
      streamingChunkStats.textContent = '0 chunks';

      const reader = res.body.getReader();
      const chunks = [];
      let receivedBytes = 0;
      let chunkCount = 0;
      let ttfa = 0;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        chunkCount++;
        if (chunkCount === 1) {
          ttfa = Math.round(performance.now() - t0);
        }
        chunks.push(value);
        receivedBytes += value.length;
        streamingStatusText.textContent = `Đang nhận luồng: ${(receivedBytes / 1024).toFixed(1)} KB...`;
        streamingChunkStats.textContent = `${chunkCount} chunks · TTFA: ${ttfa}ms`;
      }

      const totalElapsed = Math.round(performance.now() - t0);
      streamingStatusText.textContent = `✅ Đã nhận đủ luồng ${(receivedBytes / 1024).toFixed(1)} KB`;

      const fullBlob = new Blob(chunks, { type: 'audio/wav' });
      const url = URL.createObjectURL(fullBlob);
      audioPlayer.src = url;
      downloadBtn.href = url;
      downloadBtn.download = `kokoro_${voiceSelect.value}_${Date.now()}.wav`;

      audioPlayer.onloadedmetadata = () => {
        const dur = audioPlayer.duration;
        const rtf = dur > 0 ? (totalElapsed / 1000 / dur).toFixed(2) : '0.00';
        metricFormat.textContent = `WAV STREAM · 24kHz`;
        metricLatency.textContent = `TTFA: ${ttfa}ms · Tổng: ${totalElapsed}ms`;
        metricRTF.textContent = `RTF: ${rtf}x (${dur.toFixed(1)}s audio)`;
      };
      audioPlayer.play().catch(() => {});
    } else {
      // ---------------------------------------------------- Standard Batch
      streamingIndicator.style.display = 'none';
      const elapsed = Math.round(performance.now() - t0);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      audioPlayer.src = url;
      downloadBtn.href = url;
      downloadBtn.download = `kokoro_${voiceSelect.value}_${Date.now()}.${format}`;

      audioPlayer.onloadedmetadata = () => {
        const dur = audioPlayer.duration;
        const rtf = dur > 0 ? (elapsed / 1000 / dur).toFixed(2) : '0.00';
        metricFormat.textContent = `${format.toUpperCase()} · 24kHz Mono`;
        metricLatency.textContent = `Thời gian xử lý: ${elapsed}ms`;
        metricRTF.textContent = `RTF: ${rtf}x (${dur.toFixed(1)}s audio)`;
      };
      audioPlayer.play().catch(() => {});
    }
  } catch (err) {
    alert('Lỗi: ' + err.message);
  } finally {
    generateBtn.disabled = false;
    btnIcon.textContent = 'play_arrow';
    btnIcon.classList.remove('spin');
    btnText.textContent = 'Tổng Hợp Giọng Nói';
  }
}

// Metrics Tab
async function loadMetrics() {
  try {
    const res = await fetch('/metrics');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('metricsRawJson').textContent = JSON.stringify(data, null, 2);
      if (data.audio_seconds_total !== undefined) {
        document.getElementById('metricTotalAudio').textContent = parseFloat(data.audio_seconds_total).toFixed(1) + ' giây';
      }
      if (data.requests_total !== undefined) {
        document.getElementById('metricTotalReqs').textContent = `${data.requests_total} lượt yêu cầu`;
      }
      if (data.rtf !== undefined) {
        document.getElementById('metricRTFAvg').textContent = `${parseFloat(data.rtf).toFixed(2)}x`;
      }
      if (data.queued !== undefined) {
        document.getElementById('metricQueueStatus').textContent = `${data.queued} đang chờ / ${data.active || 0} active`;
      }
    }
  } catch (err) {
    document.getElementById('metricsRawJson').textContent = 'Không thể tải metrics: ' + err.message;
  }
}

// Code Generator
function switchCodeLang(lang) {
  currentCodeLang = lang;
  document.querySelectorAll('.code-hub-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  updateCodeSnippet();
}

function updateCodeSnippet() {
  const box = document.getElementById('codeSnippetBox');
  const text = textInput.value.replace(/"/g, '\\"').replace(/\n/g, ' ');
  const voice = voiceSelect ? voiceSelect.value : 'diem_trinh';
  const format = formatSelect ? formatSelect.value : 'mp3';
  const speed = speedRange ? parseFloat(speedRange.value).toFixed(2) : '1.00';
  const host = window.location.origin || 'http://localhost:8880';

  if (currentCodeLang === 'python') {
    box.textContent = `from openai import OpenAI

# Kết nối tới máy chủ Kokoro-Vietnamese
client = OpenAI(
    base_url="${host}/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="Kokoro-Vietnamese",
    voice="${voice}",
    input="${text}",
    response_format="${format}",
    speed=${speed}
)

response.stream_to_file("output.${format}")
print("Tổng hợp hoàn tất: output.${format}")`;
  } else if (currentCodeLang === 'curl') {
    box.textContent = `curl -X POST "${host}/v1/audio/speech" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "Kokoro-Vietnamese",
    "voice": "${voice}",
    "input": "${text}",
    "response_format": "${format}",
    "speed": ${speed}
  }' \\
  --output "output.${format}"`;
  } else if (currentCodeLang === 'js') {
    box.textContent = `import fs from "fs";
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "${host}/v1",
  apiKey: "not-needed"
});

async function main() {
  const mp3 = await openai.audio.speech.create({
    model: "Kokoro-Vietnamese",
    voice: "${voice}",
    input: "${text}",
    response_format: "${format}",
    speed: ${speed}
  });
  const buffer = Buffer.from(await mp3.arrayBuffer());
  await fs.promises.writeFile("output.${format}", buffer);
  console.log("Audio saved!");
}
main();`;
  } else if (currentCodeLang === 'gpustack') {
    box.textContent = `# GPUStack Custom Inference Backend Manifest
backend_name: Kokoro-TTS-Vietnamese-custom
default_version: "1.0.0"
health_check_path: /health
version_configs:
  "1.0.0":
    image_name: kokoro-vi-server:gpu
    custom_framework: cuda
    run_command: >-
      --model-path {{model_path}}
      --served-model-name {{model_name}}
      --host {{worker_ip}}
      --port {{port}}
  "1.0.0-cpu":
    image_name: kokoro-vi-server:cpu
    custom_framework: cpu
    run_command: >-
      --model-path {{model_path}}
      --served-model-name {{model_name}}
      --host {{worker_ip}}
      --port {{port}}`;
  }
}

function copyCode() {
  const code = document.getElementById('codeSnippetBox').textContent;
  navigator.clipboard.writeText(code).then(() => {
    alert('Đã sao chép đoạn mã vào clipboard!');
  });
}

initData();
</script>
</body>
</html>
"""
