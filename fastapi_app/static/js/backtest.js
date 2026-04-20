/* ── Backtest Dashboard JS ──────────────────────────────────────────────────── */

let allData = {};
let configData = {};
let activeYear = null;

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'Inter, sans-serif', color: '#dde3f0', size: 12 },
  margin: { t: 30, b: 50, l: 55, r: 20 },
  xaxis: { gridcolor: '#252a3a', zerolinecolor: '#252a3a' },
  yaxis: { gridcolor: '#252a3a', zerolinecolor: '#252a3a' },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ── Helpers ────────────────────────────────────────────────────────────────

function valClass(v) {
  if (v > 0) return 'val-positive';
  if (v < 0) return 'val-negative';
  return 'val-neutral';
}

function fmt(v, suffix = '') {
  if (v === null || v === undefined || v === '--') return '<span class="val-neutral">--</span>';
  const cls = valClass(v);
  return `<span class="${cls}">${v}${suffix}</span>`;
}

function fmtDollar(v) {
  if (v === null || v === undefined) return '<span class="val-neutral">--</span>';
  const cls = valClass(v);
  const prefix = v >= 0 ? '+' : '';
  return `<span class="${cls}">$${prefix}${v.toLocaleString()}</span>`;
}

// ── Fetch & Render ─────────────────────────────────────────────────────────

async function init() {
  try {
    const resp = await fetch('/api/bucket-analysis');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const raw = await resp.json();

    configData = raw.config || {};
    delete raw.config;
    allData = raw;

    const years = Object.keys(allData).sort().reverse();
    if (!years.length) throw new Error('No backtest data found');

    buildYearTabs(years);
    activeYear = years[0];
    renderConfig(configData);
    renderYear(activeYear);
    setupCollapsible();

    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('content').style.display = 'block';
  } catch (e) {
    document.getElementById('loadingState').style.display = 'none';
    const el = document.getElementById('errorState');
    el.style.display = 'block';
    el.textContent = `Error loading backtest data: ${e.message}`;
  }
}

function buildYearTabs(years) {
  const container = document.getElementById('yearTabs');
  container.innerHTML = '';
  years.forEach(y => {
    const btn = document.createElement('button');
    btn.className = 'year-tab';
    btn.textContent = y;
    btn.dataset.year = y;
    btn.onclick = () => switchYear(y);
    container.appendChild(btn);
  });
}

function switchYear(year) {
  activeYear = year;
  document.querySelectorAll('.year-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.year === year);
  });
  renderYear(year);
}

function renderYear(year) {
  const d = allData[year];
  if (!d) return;
  renderKPIs(d);
  renderOddsBuckets(d.odds_buckets);
  renderWeightClass(d.weight_class);
  renderFightsTable(d.fights);
}

// ── KPI Strip ──────────────────────────────────────────────────────────────

function renderKPIs(d) {
  const w = d.weighted_roi;
  const o = d.overall;
  const items = [
    { label: 'Total Bets', value: o.n, suffix: '' },
    { label: 'Win Rate', value: o.win_rate, suffix: '%' },
    { label: 'Flat ROI', value: o.roi, suffix: '%' },
    { label: 'Weighted ROI', value: w.weighted.roi, suffix: '%' },
    { label: 'Flat P&L', value: Math.round(o.profit * 100), dollar: true },
    { label: 'Weighted P&L', value: Math.round(w.weighted.profit), dollar: true },
    { label: 'Lift', value: w.lift_pp, suffix: 'pp' },
  ];

  const el = document.getElementById('kpiStrip');
  el.innerHTML = items.map(i => {
    const cls = i.value > 0 ? 'positive' : i.value < 0 ? 'negative' : '';
    let display;
    if (i.dollar) {
      const prefix = i.value >= 0 ? '+' : '';
      display = `$${prefix}${i.value.toLocaleString()}`;
    } else {
      const prefix = (i.suffix === '%' || i.suffix === 'pp') && i.value > 0 ? '+' : '';
      display = `${prefix}${i.value}${i.suffix}`;
    }
    return `<div class="kpi-card"><div class="kpi-label">${i.label}</div><div class="kpi-value ${cls}">${display}</div></div>`;
  }).join('');
}

// ── Odds Bucket Chart + Table ──────────────────────────────────────────────

function renderOddsBuckets(data) {
  const buckets = data.buckets.filter(b => b.ALL.n > 0);
  const labels = buckets.map(b => b.label);
  const rois = buckets.map(b => b.ALL.roi);
  const winRates = buckets.map(b => b.ALL.win_rate);
  const colors = rois.map(r => r >= 0 ? '#00d4aa' : '#ff4b6e');

  Plotly.newPlot('oddsBucketChart', [
    {
      x: labels, y: rois, type: 'bar', name: 'ROI %',
      marker: { color: colors, opacity: 0.85 },
      text: rois.map(r => `${r.toFixed(1)}%`),
      textposition: 'outside',
      textfont: { color: '#dde3f0', size: 11 },
    },
    {
      x: labels, y: winRates, type: 'scatter', mode: 'lines+markers',
      name: 'Win Rate %', yaxis: 'y2',
      line: { color: '#4facfe', width: 2 },
      marker: { size: 6, color: '#4facfe' },
    },
  ], {
    ...PLOTLY_LAYOUT,
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'ROI %' },
    yaxis2: { overlaying: 'y', side: 'right', title: 'Win Rate %', gridcolor: 'transparent', font: { color: '#4facfe' } },
    legend: { x: 0, y: 1.12, orientation: 'h', font: { size: 11 } },
    barmode: 'group',
  }, PLOTLY_CONFIG);

  // Table
  const allBuckets = data.buckets;
  let html = `<table class="bt-table"><thead><tr>
    <th>Bucket</th><th>Gender</th><th>N</th><th>W</th><th>L</th>
    <th>Win Rate</th><th>Profit</th><th>ROI</th><th>Avg Edge</th><th>Avg Conf</th>
  </tr></thead><tbody>`;

  allBuckets.forEach(b => {
    for (const [gender, stats] of Object.entries(b)) {
      if (['label', 'key'].includes(gender)) continue;
      if (stats.n === 0 && gender === 'ALL') {
        html += `<tr class="${gender !== 'ALL' ? 'gender-row' : ''}"><td>${gender === 'ALL' ? b.label : ''}</td><td>${gender}</td>
          <td colspan="8" class="val-neutral">--</td></tr>`;
        continue;
      }
      if (stats.n === 0) continue;
      const rc = gender !== 'ALL' ? 'gender-row' : '';
      html += `<tr class="${rc}">
        <td>${gender === 'ALL' ? b.label : ''}</td><td>${gender}</td>
        <td>${stats.n}</td><td>${stats.w}</td><td>${stats.l}</td>
        <td>${fmt(stats.win_rate, '%')}</td><td>${fmt(stats.profit)}</td>
        <td>${fmt(stats.roi, '%')}</td><td>${stats.avg_edge}%</td><td>${stats.avg_conf}%</td>
      </tr>`;
    }
  });

  // Totals
  const totals = data.totals;
  for (const [gender, stats] of Object.entries(totals)) {
    const rc = gender !== 'ALL' ? 'gender-row' : 'total-row';
    html += `<tr class="${rc}">
      <td>${gender === 'ALL' ? 'TOTAL' : ''}</td><td>${gender}</td>
      <td>${stats.n}</td><td>${stats.w}</td><td>${stats.l}</td>
      <td>${fmt(stats.win_rate, '%')}</td><td>${fmt(stats.profit)}</td>
      <td>${fmt(stats.roi, '%')}</td><td>${stats.avg_edge}%</td><td>${stats.avg_conf}%</td>
    </tr>`;
  }

  html += '</tbody></table>';
  document.getElementById('oddsBucketTable').innerHTML = html;
}

// ── Weight Class Chart + Table ──────────────────────────────────────────────

function renderWeightClass(data) {
  // Filter to classes with bets
  const classes = data.filter(c => c.n > 0);
  const labels = classes.map(c => c.weight_class);
  const rois = classes.map(c => c.roi);
  const winRates = classes.map(c => c.win_rate);
  const ns = classes.map(c => c.n);
  const colors = rois.map(r => r >= 0 ? '#00d4aa' : '#ff4b6e');

  Plotly.newPlot('weightClassChart', [
    {
      x: labels, y: rois, type: 'bar', name: 'ROI %',
      marker: { color: colors, opacity: 0.85 },
      text: rois.map(r => `${r.toFixed(1)}%`),
      textposition: 'outside',
      textfont: { color: '#dde3f0', size: 11 },
    },
    {
      x: labels, y: winRates, type: 'scatter', mode: 'lines+markers',
      name: 'Win Rate %', yaxis: 'y2',
      line: { color: '#4facfe', width: 2 },
      marker: { size: 6, color: '#4facfe' },
    },
  ], {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -30 },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'ROI %' },
    yaxis2: { overlaying: 'y', side: 'right', title: 'Win Rate %', gridcolor: 'transparent' },
    legend: { x: 0, y: 1.12, orientation: 'h', font: { size: 11 } },
  }, PLOTLY_CONFIG);

  // Table
  let html = `<table class="bt-table"><thead><tr>
    <th>Weight Class</th><th>N</th><th>W</th><th>L</th>
    <th>Win Rate</th><th>Profit</th><th>ROI</th><th>Avg Edge</th><th>Avg Conf</th>
  </tr></thead><tbody>`;

  classes.forEach(c => {
    html += `<tr>
      <td>${c.weight_class}</td>
      <td>${c.n}</td><td>${c.w}</td><td>${c.l}</td>
      <td>${fmt(c.win_rate, '%')}</td><td>${fmt(c.profit)}</td>
      <td>${fmt(c.roi, '%')}</td><td>${c.avg_edge}%</td><td>${c.avg_conf}%</td>
    </tr>`;
  });

  html += '</tbody></table>';
  document.getElementById('weightClassTable').innerHTML = html;
}

// ── Config Display ──────────────────────────────────────────────────────────

const SKIP_LEGEND = [
  { code: 'F1', desc: 'Favorite low confidence' },
  { code: 'F2', desc: 'Favorite odds cap exceeded' },
  { code: 'U1', desc: 'Underdog low confidence' },
  { code: 'U2', desc: 'Low edge' },
  { code: 'U3', desc: 'Underdog odds cap exceeded' },
  { code: 'D1', desc: 'Insufficient fight data' },
  { code: 'ERR', desc: 'Prediction failed' },
];

function renderConfig(cfg) {
  const filters = cfg.filters || {};
  const buckets = cfg.edge_buckets || [];
  const wmma = cfg.wmma_rules || {};
  const baseUnit = cfg.betting?.base_unit || 100;

  let html = '';

  // Filters card
  html += `<div class="config-card">
    <div class="config-card-title">Filters</div>
    <div class="config-row"><span class="config-key">Min Fights</span><span class="config-val">${filters.min_fights || '—'}</span></div>
    <div class="config-row"><span class="config-key">Edge Min</span><span class="config-val">≥ ${filters.edge_min ? (filters.edge_min * 100) + '%' : '—'}</span></div>
    <div class="config-row"><span class="config-key">Dog Edge Min</span><span class="config-val">≥ ${filters.underdog_edge_min ? (filters.underdog_edge_min * 100) + '%' : '—'}</span></div>
    <div class="config-row"><span class="config-key">Fav Confidence</span><span class="config-val">≥ ${filters.favorite_confidence_min ? (filters.favorite_confidence_min * 100) + '%' : '—'}</span></div>
    <div class="config-row"><span class="config-key">Dog Confidence</span><span class="config-val">≥ ${filters.underdog_confidence_min ? (filters.underdog_confidence_min * 100) + '%' : '—'}</span></div>
    <div class="config-row"><span class="config-key">Fav Odds Cap</span><span class="config-val">${filters.favorite_odds_cap || '—'}</span></div>
    <div class="config-row"><span class="config-key">Dog Odds Cap</span><span class="config-val">+${filters.underdog_odds_cap || '—'}</span></div>
  </div>`;

  // Edge buckets card
  html += `<div class="config-card">
    <div class="config-card-title">Edge Buckets (base $${baseUnit})</div>`;
  buckets.forEach(b => {
    const lo = (b.min_edge * 100).toFixed(0);
    const hi = (b.max_edge * 100).toFixed(0);
    const action = b.action === 'skip' ? 'Skip' : `${b.multiplier}x ($${baseUnit * b.multiplier})`;
    html += `<div class="config-row"><span class="config-key">${lo}–${hi}%</span><span class="config-val">${action}</span></div>`;
  });
  html += `</div>`;

  // Skip codes + WMMA card
  html += `<div class="config-card">
    <div class="config-card-title">Skip Codes & WMMA</div>`;
  SKIP_LEGEND.forEach(s => {
    html += `<div class="config-row"><span class="config-key"><span class="skip-code">${s.code}</span></span><span class="config-val">${s.desc}</span></div>`;
  });
  if (wmma.enabled) {
    html += `<div class="config-row" style="margin-top:8px;border-top:1px solid var(--border);padding-top:6px"><span class="config-key">WMMA Min Edge</span><span class="config-val">${(wmma.min_edge * 100)}%</span></div>`;
    html += `<div class="config-row"><span class="config-key">WMMA Max Mult</span><span class="config-val">${wmma.max_multiplier}x</span></div>`;
  }
  html += `</div>`;

  document.getElementById('configDisplay').innerHTML = html;
}

// ── Fights Table ───────────────────────────────────────────────────────────

function renderFightsTable(fights) {
  if (!fights || !fights.length) {
    document.getElementById('fightsTable').innerHTML = '<p class="val-neutral" style="padding:16px">No fight data available.</p>';
    return;
  }

  let html = `<table class="bt-table"><thead><tr>
    <th>Date</th><th>Pick</th><th>vs</th><th>Odds</th><th>Model %</th>
    <th>Bet</th><th>Result</th><th>P&L</th><th>Class</th>
  </tr></thead><tbody>`;

  fights.forEach(f => {
    const opponent = f.pick === f.fighter1 ? f.fighter2 : f.fighter1;
    const isBet = f.bet;
    const rowClass = isBet ? (f.correct ? 'result-correct' : 'result-wrong') : 'result-skip';

    const betCell = isBet
      ? '<span class="bet-yes">BET</span>'
      : `<span class="skip-code">${f.skip_code}</span>`;

    const resultCell = isBet
      ? (f.correct ? '<span class="val-positive">✓ W</span>' : '<span class="val-negative">✗ L</span>')
      : '<span class="val-neutral">—</span>';

    const pnlCell = isBet
      ? `<span class="${f.pnl >= 0 ? 'val-positive' : 'val-negative'}">${f.pnl >= 0 ? '+' : ''}${f.pnl.toFixed(2)}</span>`
      : '<span class="val-neutral">—</span>';

    html += `<tr class="${rowClass}">
      <td>${f.date}</td>
      <td>${f.pick}</td>
      <td>${opponent}</td>
      <td>${f.pick_odds >= 0 ? '+' : ''}${f.pick_odds}</td>
      <td>${f.pick_prob}%</td>
      <td>${betCell}</td>
      <td>${resultCell}</td>
      <td>${pnlCell}</td>
      <td>${f.weight_class}</td>
    </tr>`;
  });

  html += '</tbody></table>';
  document.getElementById('fightsTable').innerHTML = html;
}

// ── Collapsible ────────────────────────────────────────────────────────────

function setupCollapsible() {
  const toggle = document.getElementById('fightsToggle');
  const content = document.getElementById('fightsContent');
  const arrow = toggle.querySelector('.toggle-arrow');

  toggle.addEventListener('click', () => {
    const open = content.style.display !== 'none';
    content.style.display = open ? 'none' : 'block';
    arrow.classList.toggle('open', !open);
  });
}

// ── Boot ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
