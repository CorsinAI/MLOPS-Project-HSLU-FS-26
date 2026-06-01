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

    /* Drift */
    .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge.ok { background: #e8f5e9; color: #2e7d32; }
    .badge.error { background: #ffebee; color: #c62828; }
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
  </style>
</head>
<body>

<header>
  <h1>Job Posting Forecast</h1>
  <p>7-day-ahead demand forecasts for the Swiss job market — LightGBM · Hopsworks · MLflow</p>
</header>

<nav>
  <button class="active" onclick="showTab('drift', this)">Drift</button>
  <button onclick="showTab('dashboard', this)">Dashboard</button>
</nav>

<main>

  <!-- DRIFT -->
  <div id="tab-drift" class="section active">
    <h2>Feature Drift</h2>
    <div id="drift-content" class="loading">Loading...</div>
  </div>

  <!-- DASHBOARD (iframe) -->
  <div id="tab-dashboard" class="section">
    <iframe src="/dashboard" style="width:100%; height:80vh; border:none; border-radius:10px;"></iframe>
  </div>

</main>

<script>
  function showTab(name, btn) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
    if (name === 'drift') loadDrift();
  }

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

  loadDrift();
</script>
</body>
</html>"""
