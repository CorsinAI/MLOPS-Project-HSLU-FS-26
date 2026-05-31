"""
Renders the HTML dashboard for the inference API.
Called by the /dashboard endpoint in app.py.
"""
import json


def render(state: dict) -> str:
    forecasts = state["forecasts"]
    drift = state["drift_report"]

    # Top 25 pairs for the bar chart
    top = forecasts[:25]
    labels = [f"{f['job_title']} — {f['location']}" for f in top]
    values = [f["predicted_count"] for f in top]

    # All unique roles and locations for the filter dropdowns
    roles = sorted({f["job_title"] for f in forecasts})
    locations = sorted({f["location"] for f in forecasts})

    drift_color = "#e74c3c" if drift["drift_detected"] else "#2ecc71"
    drift_label = "DRIFT DETECTED" if drift["drift_detected"] else "No drift"

    all_forecasts_json = json.dumps(forecasts)
    labels_json = json.dumps(labels)
    values_json = json.dumps(values)
    roles_json = json.dumps(roles)
    locations_json = json.dumps(locations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>JobAnalysis — Forecasts</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f0f2f5; color: #1a1a2e; }}

    header {{ background: #1a1a2e; color: #fff; padding: 20px 32px;
              display: flex; align-items: center; justify-content: space-between; }}
    header h1 {{ font-size: 1.4rem; font-weight: 600; }}
    header h1 span {{ color: #4fc3f7; }}
    .meta {{ display: flex; gap: 24px; font-size: 0.85rem; opacity: 0.85; }}
    .meta b {{ color: #4fc3f7; }}

    .drift-badge {{ padding: 6px 14px; border-radius: 20px; font-size: 0.8rem;
                   font-weight: 700; background: {drift_color}22;
                   color: {drift_color}; border: 1px solid {drift_color}; }}

    main {{ max-width: 1100px; margin: 28px auto; padding: 0 24px;
            display: grid; gap: 24px; }}

    .card {{ background: #fff; border-radius: 12px; padding: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,.07); }}
    .card h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 18px;
                color: #1a1a2e; border-left: 3px solid #4fc3f7; padding-left: 10px; }}

    .chart-wrap {{ position: relative; height: 520px; }}

    .controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }}
    select, input {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 8px;
                     font-size: 0.9rem; background: #fafafa; min-width: 200px; }}
    select:focus, input:focus {{ outline: none; border-color: #4fc3f7; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th {{ background: #f8f9fb; text-align: left; padding: 10px 14px;
          font-weight: 600; color: #555; border-bottom: 2px solid #eee; }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #f0f0f0; }}
    tr:hover td {{ background: #f8fbff; }}
    .bar-cell {{ width: 120px; }}
    .bar-bg {{ background: #e8f4fc; border-radius: 4px; height: 8px; overflow: hidden; }}
    .bar-fill {{ background: #4fc3f7; height: 100%; border-radius: 4px; }}
    .count {{ font-weight: 600; color: #1a73e8; }}

    .drift-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                   gap: 14px; }}
    .drift-item {{ background: #f8f9fb; border-radius: 8px; padding: 14px;
                   border-left: 4px solid #ddd; }}
    .drift-item.drifted {{ border-left-color: #e74c3c; background: #fff5f5; }}
    .drift-item.ok {{ border-left-color: #2ecc71; }}
    .drift-item .feature {{ font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: .05em; color: #888; margin-bottom: 6px; }}
    .drift-item .zscore {{ font-size: 1.3rem; font-weight: 700; }}
    .drift-item.drifted .zscore {{ color: #e74c3c; }}
    .drift-item.ok .zscore {{ color: #2ecc71; }}
    .drift-item .means {{ font-size: 0.78rem; color: #888; margin-top: 4px; }}
  </style>
</head>
<body>

<header>
  <h1>Job<span>Analysis</span> &mdash; Demand Forecasts</h1>
  <div class="meta">
    <div>Window <b>{state["forecast_window_start"]} → {state["forecast_window_end"]}</b></div>
    <div>Model <b>v{state["model_version"]}</b></div>
    <div>{state["num_pairs"]} pairs</div>
    <div>Generated <b>{state["generated_at"][:10]}</b></div>
  </div>
  <div class="drift-badge">{drift_label}</div>
</header>

<main>

  <!-- Top 25 bar chart -->
  <div class="card">
    <h2>Top 25 — Predicted postings next 3-day window</h2>
    <div class="chart-wrap">
      <canvas id="topChart"></canvas>
    </div>
  </div>

  <!-- Filterable table -->
  <div class="card">
    <h2>All forecasts</h2>
    <div class="controls">
      <select id="roleFilter">
        <option value="">All roles</option>
      </select>
      <select id="locationFilter">
        <option value="">All locations</option>
      </select>
    </div>
    <table id="forecastTable">
      <thead>
        <tr>
          <th>Role</th>
          <th>Location</th>
          <th>Last observed window</th>
          <th>Predicted count</th>
          <th class="bar-cell"></th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>

  <!-- Drift -->
  <div class="card">
    <h2>Feature drift report</h2>
    <div class="drift-grid" id="driftGrid"></div>
  </div>

</main>

<script>
const ALL_FORECASTS = {all_forecasts_json};
const TOP_LABELS    = {labels_json};
const TOP_VALUES    = {values_json};
const ROLES         = {roles_json};
const LOCATIONS     = {locations_json};
const DRIFT         = {json.dumps(drift)};
const MAX_COUNT     = Math.max(...ALL_FORECASTS.map(f => f.predicted_count));

// ── Bar chart ──────────────────────────────────────────────────────────────
new Chart(document.getElementById('topChart'), {{
  type: 'bar',
  data: {{
    labels: TOP_LABELS,
    datasets: [{{
      label: 'Predicted postings',
      data: TOP_VALUES,
      backgroundColor: 'rgba(79, 195, 247, 0.75)',
      borderColor: 'rgba(79, 195, 247, 1)',
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ beginAtZero: true, grid: {{ color: '#f0f0f0' }} }},
      y: {{ ticks: {{ font: {{ size: 11 }} }}, grid: {{ display: false }} }}
    }}
  }}
}});

// ── Role dropdown ──────────────────────────────────────────────────────────
const roleSelect = document.getElementById('roleFilter');
ROLES.forEach(r => {{
  const opt = document.createElement('option');
  opt.value = r; opt.textContent = r;
  roleSelect.appendChild(opt);
}});

// ── Location dropdown ───────────────────────────────────────────────────────
const locSelect = document.getElementById('locationFilter');
LOCATIONS.forEach(l => {{
  const opt = document.createElement('option');
  opt.value = l; opt.textContent = l;
  locSelect.appendChild(opt);
}});

// ── Table ──────────────────────────────────────────────────────────────────
function renderTable() {{
  const role = roleSelect.value.toLowerCase();
  const loc  = locSelect.value.toLowerCase();
  const rows = ALL_FORECASTS.filter(f =>
    (!role || f.job_title.toLowerCase() === role) &&
    (!loc  || f.location.toLowerCase() === loc)
  );
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = rows.map(f => {{
    const pct = (f.predicted_count / MAX_COUNT * 100).toFixed(1);
    return `<tr>
      <td>${{f.job_title}}</td>
      <td>${{f.location}}</td>
      <td>${{f.last_observed_window}}</td>
      <td class="count">${{f.predicted_count}}</td>
      <td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${{pct}}%"></div></div></td>
    </tr>`;
  }}).join('');
}}

roleSelect.addEventListener('change', renderTable);
locSelect.addEventListener('change', renderTable);
renderTable();

// ── Drift ──────────────────────────────────────────────────────────────────
const grid = document.getElementById('driftGrid');
Object.entries(DRIFT.features).forEach(([feat, info]) => {{
  const cls = info.drifted ? 'drifted' : 'ok';
  grid.innerHTML += `
    <div class="drift-item ${{cls}}">
      <div class="feature">${{feat}}</div>
      <div class="zscore">z = ${{info.z_score}}</div>
      <div class="means">ref ${{info.reference_mean}} → now ${{info.current_mean}}</div>
    </div>`;
}});
</script>

</body>
</html>"""
