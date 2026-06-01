"""Renders the landing page for the inference service."""


def render() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Job Posting Forecast</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f5f7fa; color: #1a1a2e; }

    header { background: linear-gradient(135deg, #1976d2, #42a5f5); color: white; padding: 36px 40px 28px; }
    header h1 { font-size: 1.8rem; font-weight: 700; }
    header p { margin-top: 6px; opacity: 0.9; font-size: 0.95rem; }

    nav { background: white; border-bottom: 1px solid #e0e0e0; display: flex; gap: 0; padding: 0 40px; }
    nav button { background: none; border: none; padding: 16px 20px; font-size: 0.9rem; cursor: pointer;
      color: #555; border-bottom: 3px solid transparent; font-weight: 500; transition: all 0.15s; }
    nav button:hover { color: #1976d2; }
    nav button.active { color: #1976d2; border-bottom-color: #1976d2; }

    main { padding: 32px 40px; max-width: 1100px; margin: 0 auto; }

    .section { display: none; }
    .section.active { display: block; }

    /* Health */
    .health-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 8px; }
    .health-card { background: white; border-radius: 10px; padding: 20px 24px; border: 1px solid #e0e0e0; }
    .health-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin-bottom: 6px; }
    .health-card .value { font-size: 1.1rem; font-weight: 600; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge.ok { background: #e8f5e9; color: #2e7d32; }
    .badge.error { background: #ffebee; color: #c62828; }

    /* Forecasts */
    .toolbar { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    .toolbar input { padding: 9px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 0.9rem; width: 220px; }
    .toolbar input:focus { outline: none; border-color: #1976d2; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden;
      border: 1px solid #e0e0e0; font-size: 0.9rem; }
    thead { background: #f0f4f8; }
    th { padding: 12px 16px; text-align: left; font-weight: 600; color: #444; font-size: 0.8rem;
      text-transform: uppercase; letter-spacing: 0.04em; }
    td { padding: 11px 16px; border-top: 1px solid #f0f0f0; }
    tr:hover td { background: #fafbff; }
    .bar-cell { display: flex; align-items: center; gap: 10px; }
    .bar { height: 8px; background: #1976d2; border-radius: 4px; min-width: 2px; }
    .count { font-weight: 600; }
    .table-footer { padding: 10px 16px; font-size: 0.8rem; color: #888; background: white;
      border-top: 1px solid #e0e0e0; border-radius: 0 0 10px 10px; }

    /* Drift */
    .drift-header { background: white; border-radius: 10px; padding: 20px 24px; border: 1px solid #e0e0e0;
      margin-bottom: 20px; display: flex; align-items: center; gap: 16px; }
    .drift-header .status-text { font-size: 1rem; font-weight: 600; }
    .drift-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
    .drift-card { background: white; border-radius: 10px; padding: 20px 24px; border: 1px solid #e0e0e0; }
    .drift-card.drifted { border-color: #ef9a9a; background: #fff8f8; }
    .drift-card .feature-name { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
      color: #888; margin-bottom: 12px; }
    .drift-card .zscore { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
    .drift-card.drifted .zscore { color: #c62828; }
    .drift-card:not(.drifted) .zscore { color: #2e7d32; }
    .drift-card .meta { font-size: 0.8rem; color: #888; }

    /* Shared */
    h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 20px; }
    .loading { color: #888; font-size: 0.9rem; padding: 40px 0; text-align: center; }
    .error-msg { color: #c62828; font-size: 0.9rem; padding: 20px 0; }
    .refresh-btn { background: #1976d2; color: white; border: none; padding: 9px 20px; border-radius: 8px;
      font-size: 0.9rem; cursor: pointer; font-weight: 500; transition: background 0.15s; }
    .refresh-btn:hover { background: #1565c0; }
    .refresh-btn:disabled { background: #90caf9; cursor: not-allowed; }
    .meta-row { font-size: 0.8rem; color: #888; margin-bottom: 16px; }
  </style>
</head>
<body>

<header>
  <h1>Job Posting Forecast</h1>
  <p>3-day-ahead demand forecasts for the Swiss job market — LightGBM · Hopsworks · MLflow</p>
</header>

<nav>
  <button class="active" onclick="showTab('health', this)">Health</button>
  <button onclick="showTab('forecasts', this)">Forecasts</button>
  <button onclick="showTab('drift', this)">Drift</button>
  <button onclick="showTab('dashboard', this)">Dashboard</button>
  <button onclick="showTab('docs', this)">API Docs</button>
</nav>

<main>

  <!-- HEALTH -->
  <div id="tab-health" class="section active">
    <h2>Service Health</h2>
    <div id="health-content" class="loading">Loading...</div>
    <br>
    <button class="refresh-btn" id="refresh-btn" onclick="triggerRefresh()">Refresh Forecasts</button>
    <span id="refresh-status" style="margin-left:12px; font-size:0.85rem; color:#555;"></span>
  </div>

  <!-- FORECASTS -->
  <div id="tab-forecasts" class="section">
    <h2>Forecasts</h2>
    <div class="toolbar">
      <input id="role-filter" type="text" placeholder="Filter by role..." oninput="filterTable()"/>
      <input id="loc-filter" type="text" placeholder="Filter by location..." oninput="filterTable()"/>
    </div>
    <div id="forecasts-content" class="loading">Loading...</div>
  </div>

  <!-- DRIFT -->
  <div id="tab-drift" class="section">
    <h2>Feature Drift</h2>
    <div id="drift-content" class="loading">Loading...</div>
  </div>

  <!-- DASHBOARD (iframe) -->
  <div id="tab-dashboard" class="section">
    <iframe src="/dashboard" style="width:100%; height:80vh; border:none; border-radius:10px;"></iframe>
  </div>

  <!-- DOCS (iframe) -->
  <div id="tab-docs" class="section">
    <iframe src="/docs" style="width:100%; height:80vh; border:none; border-radius:10px;"></iframe>
  </div>

</main>

<script>
  let allForecasts = [];
  let maxCount = 1;

  function showTab(name, btn) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
    if (name === 'forecasts' && allForecasts.length === 0) loadForecasts();
    if (name === 'drift') loadDrift();
  }

  // --- HEALTH ---
  async function loadHealth() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      document.getElementById('health-content').innerHTML = `
        <div class="health-grid">
          <div class="health-card">
            <div class="label">Status</div>
            <div class="value"><span class="badge ok">OK</span></div>
          </div>
          <div class="health-card">
            <div class="label">Model Version</div>
            <div class="value">${d.model_version}</div>
          </div>
          <div class="health-card">
            <div class="label">Forecast Pairs</div>
            <div class="value">${d.num_pairs}</div>
          </div>
          <div class="health-card">
            <div class="label">Generated At</div>
            <div class="value" style="font-size:0.9rem">${new Date(d.generated_at).toLocaleString()}</div>
          </div>
        </div>`;
    } catch(e) {
      document.getElementById('health-content').innerHTML = '<span class="error-msg">Failed to load health data.</span>';
    }
  }

  async function triggerRefresh() {
    const btn = document.getElementById('refresh-btn');
    const status = document.getElementById('refresh-status');
    btn.disabled = true;
    btn.textContent = 'Refreshing...';
    status.textContent = '';
    try {
      const r = await fetch('/refresh', { method: 'POST' });
      const d = await r.json();
      status.textContent = 'Refreshed at ' + new Date(d.generated_at).toLocaleString();
      allForecasts = [];
      loadHealth();
    } catch(e) {
      status.textContent = 'Refresh failed.';
    }
    btn.disabled = false;
    btn.textContent = 'Refresh Forecasts';
  }

  // --- FORECASTS ---
  async function loadForecasts() {
    document.getElementById('forecasts-content').innerHTML = '<div class="loading">Loading...</div>';
    try {
      const r = await fetch('/forecasts');
      const d = await r.json();
      allForecasts = d.forecasts;
      maxCount = Math.max(...allForecasts.map(f => f.predicted_count), 1);
      renderTable(allForecasts, d);
    } catch(e) {
      document.getElementById('forecasts-content').innerHTML = '<span class="error-msg">Failed to load forecasts.</span>';
    }
  }

  function filterTable() {
    const role = document.getElementById('role-filter').value.toLowerCase();
    const loc = document.getElementById('loc-filter').value.toLowerCase();
    const filtered = allForecasts.filter(f =>
      f.job_title.toLowerCase().includes(role) &&
      f.location.toLowerCase().includes(loc)
    );
    renderTable(filtered, null);
  }

  function renderTable(rows, meta) {
    const metaHtml = meta ? `<div class="meta-row">
      Forecast window: <strong>${meta.forecast_window_start} – ${meta.forecast_window_end}</strong> &nbsp;·&nbsp;
      Model v${meta.model_version} &nbsp;·&nbsp; ${meta.count} pairs
    </div>` : '';

    const rowsHtml = rows.map(f => {
      const barWidth = Math.round((f.predicted_count / maxCount) * 120);
      return `<tr>
        <td>${f.job_title}</td>
        <td>${f.location}</td>
        <td>${f.last_observed_window}</td>
        <td><div class="bar-cell">
          <div class="bar" style="width:${barWidth}px"></div>
          <span class="count">${f.predicted_count.toFixed(1)}</span>
        </div></td>
      </tr>`;
    }).join('');

    document.getElementById('forecasts-content').innerHTML = `
      ${metaHtml}
      <table>
        <thead><tr><th>Role</th><th>Location</th><th>Last Window</th><th>Predicted Count</th></tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
      <div class="table-footer">Showing ${rows.length} of ${allForecasts.length} pairs</div>`;
  }

  // --- DRIFT ---
  async function loadDrift() {
    document.getElementById('drift-content').innerHTML = '<div class="loading">Loading...</div>';
    try {
      const r = await fetch('/drift');
      const d = await r.json();
      const overallBadge = d.drift_detected
        ? '<span class="badge error">Drift Detected</span>'
        : '<span class="badge ok">No Drift</span>';

      const cards = Object.entries(d.features).map(([name, f]) => `
        <div class="drift-card ${f.drifted ? 'drifted' : ''}">
          <div class="feature-name">${name.replace(/_/g, ' ')}</div>
          <div class="zscore">z = ${f.z_score.toFixed(2)}</div>
          <div class="meta">
            ref mean: ${f.reference_mean} &nbsp;·&nbsp; current: ${f.current_mean}<br>
            threshold: ${d.threshold} &nbsp;·&nbsp; ${f.drifted ? '⚠ drifted' : '✓ ok'}
          </div>
        </div>`).join('');

      document.getElementById('drift-content').innerHTML = `
        <div class="drift-header">
          ${overallBadge}
          <span class="status-text">${d.drift_detected ? 'One or more features have drifted beyond the threshold.' : 'All features within normal range.'}</span>
        </div>
        <div class="drift-cards">${cards}</div>`;
    } catch(e) {
      document.getElementById('drift-content').innerHTML = '<span class="error-msg">Failed to load drift data.</span>';
    }
  }

  loadHealth();
</script>
</body>
</html>"""
