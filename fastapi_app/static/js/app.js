/* ── State ─────────────────────────────────────────────────────────────────── */
let allFights   = [];
let allSkipped  = [];
let allEvents   = [];
let sortState   = { col: null, dir: 1 };
let currentFocus = 'all';
let currentTab   = 'selected';

/* ── Boot ──────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  await loadMeta();
  wireControls();
});

async function loadMeta() {
  try {
    const res  = await fetch('/api/meta');
    const meta = await res.json();

    // Default to 2025-01-01 → latest available (true out-of-sample window)
    document.getElementById('startDate').value = '2025-01-01';
    document.getElementById('endDate').value   = meta.date_max;

    // Update header
    document.getElementById('headerMeta').innerHTML =
      `<span class="badge">mar_4_v2 + underdog blend</span>
       <span class="badge" style="margin-left:6px;background:rgba(0,212,170,.1);color:var(--green);border-color:rgba(0,212,170,.25)">
         ${meta.total_fights} fights · ${meta.date_min} → ${meta.date_max}
       </span>
       <span class="badge" style="margin-left:6px;background:rgba(247,201,72,.1);color:var(--yellow);border-color:rgba(247,201,72,.25)"
             title="${meta.holdout_note}">
         ⚠ 2025+ out-of-sample
       </span>`;

    // Weight class chips
    const grid = document.getElementById('wcGrid');
    grid.innerHTML = meta.weight_classes.map(wc =>
      `<div class="wc-chip" data-wc="${wc}">${wc}</div>`
    ).join('');
    grid.querySelectorAll('.wc-chip').forEach(chip => {
      chip.addEventListener('click', () => chip.classList.toggle('active'));
    });
  } catch (e) {
    console.error('Failed to load meta:', e);
  }
}

function wireControls() {
  // Focus radio
  document.getElementById('focusGroup').querySelectorAll('.radio-option').forEach(opt => {
    opt.addEventListener('click', () => {
      document.querySelectorAll('#focusGroup .radio-option').forEach(o => o.classList.remove('active'));
      opt.classList.add('active');
      currentFocus = opt.dataset.value;
      updateBlendVisibility();
    });
  });

  // Sliders → hints
  const sliders = [
    ['udThreshold', 'udThresholdHint', v => `< ${v}%`],
    ['minConf',     'minConfHint',     v => `${v}%`],
    ['maxConf',     'maxConfHint',     v => `${v}%`],
    ['minEdge',     'minEdgeHint',     v => `${v}%`],
    ['blendWeight', 'blendHint',       v => `${v}% ud_v1 / ${100 - v}% general`],
  ];
  sliders.forEach(([id, hintId, fmt]) => {
    const el   = document.getElementById(id);
    const hint = document.getElementById(hintId);
    if (el && hint) {
      el.addEventListener('input', () => { hint.textContent = fmt(el.value); });
    }
  });

  // Blend toggle
  document.getElementById('useBlend').addEventListener('change', updateBlendVisibility);

  // Table search
  document.getElementById('tableSearch').addEventListener('input', e => {
    applySearch(e.target.value.toLowerCase());
  });

  // Sortable headers (delegated — works for dynamically shown tables)
  document.addEventListener('click', e => {
    const th = e.target.closest('th.sortable');
    if (th) sortTable(th.dataset.col, th);
  });

  // Tab switcher
  document.getElementById('tableTabGroup').querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Collapse params
  const collapseBtn = document.getElementById('collapseParams');
  const paramsGrid  = document.getElementById('paramsGrid');
  collapseBtn.addEventListener('click', () => {
    const collapsed = paramsGrid.classList.toggle('collapsed');
    collapseBtn.textContent = collapsed ? '▼' : '▲';
  });
}

function updateBlendVisibility() {
  const isUnderdogs = currentFocus === 'underdogs';
  const wrapper = document.getElementById('blendWeightWrapper');
  const blendGroup = document.getElementById('blendGroup');
  if (wrapper) {
    const useBlend = document.getElementById('useBlend').checked;
    wrapper.style.opacity = (isUnderdogs && useBlend) ? '1' : '0.35';
    wrapper.style.pointerEvents = (isUnderdogs && useBlend) ? 'auto' : 'none';
  }
  if (blendGroup) {
    blendGroup.style.opacity = isUnderdogs ? '1' : '0.4';
  }
}

/* ── Backtest Call ──────────────────────────────────────────────────────────── */
async function runBacktest() {
  const btn     = document.getElementById('runBtn');
  const btnText = document.getElementById('runBtnText');
  const spinner = document.getElementById('runSpinner');

  btn.disabled     = true;
  btnText.textContent = 'Running…';
  spinner.classList.remove('hidden');
  hideError();

  // Collect params
  const selectedWC = Array.from(document.querySelectorAll('.wc-chip.active'))
                          .map(c => c.dataset.wc);

  const minOddsRaw = document.getElementById('minOdds').value;
  const maxOddsRaw = document.getElementById('maxOdds').value;

  const body = {
    start_date:        document.getElementById('startDate').value,
    end_date:          document.getElementById('endDate').value,
    focus:             currentFocus,
    ud_threshold:      +document.getElementById('udThreshold').value / 100,
    min_confidence:    +document.getElementById('minConf').value / 100,
    max_confidence:    +document.getElementById('maxConf').value / 100,
    min_edge:          +document.getElementById('minEdge').value / 100,
    min_american_odds: minOddsRaw !== '' ? +minOddsRaw : null,
    max_american_odds: maxOddsRaw !== '' ? +maxOddsRaw : null,
    weight_classes:    selectedWC,
    use_underdog_blend: document.getElementById('useBlend').checked,
    blend_weight:      +document.getElementById('blendWeight').value / 100,
    flat_bet:          +document.getElementById('flatBet').value,
  };

  hideError();
  hideInsampleBanner();

  try {
    const res  = await fetch('/api/backtest', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok || data.detail) {
      showError(data.detail || 'Unexpected error');
      return;
    }

    renderResults(data, body.flat_bet, body.start_date);
  } catch (e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.disabled     = false;
    btnText.textContent = 'Run Backtest';
    spinner.classList.add('hidden');
  }
}

/* ── Render Results ─────────────────────────────────────────────────────────── */
function renderResults(data, flatBet, startDate) {
  const { summary, charts, fights = [], skipped = [], events = [], coverage } = data;

  // In-sample warning banner (from backend)
  if (summary.in_sample_warning) {
    showInsampleBanner(summary.in_sample_warning, summary.n_in_sample, summary.n_out_sample);
  } else {
    hideInsampleBanner();
  }

  // Coverage funnel bar
  if (coverage) renderCoverageBar(coverage);

  // Show results section
  document.getElementById('results').classList.remove('hidden');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Summary cards
  const roi = summary.roi;
  const profit = summary.total_profit;
  let betsLabel = summary.n_bets.toLocaleString();
  if (summary.n_in_sample > 0 && summary.n_out_sample > 0) {
    betsLabel += ` (${summary.n_out_sample} OOS / ${summary.n_in_sample} IS)`;
  }
  setText('statBets',     betsLabel);
  setText('statAccuracy', summary.accuracy + '%');
  setTextWithColor('statRoi',    (roi >= 0 ? '+' : '') + roi + '%', roi);
  setTextWithColor('statProfit', fmtDollar(profit), profit);
  const dd = summary.max_drawdown;
  const ddStr = dd > 0 ? '-$' + (dd >= 1000 ? (dd/1000).toFixed(1)+'k' : dd.toFixed(0)) : '$0';
  const ddEl = document.getElementById('statDrawdown');
  if (ddEl) { ddEl.textContent = ddStr; ddEl.className = 'stat-value ' + (dd > 0 ? 'negative' : ''); }
  setText('statEdge',    summary.avg_edge + '%');
  setText('statConf',    summary.avg_confidence + '%');

  const upsetCard = document.getElementById('cardUpset');
  if (currentFocus === 'underdogs' && summary.upset_detection !== null && summary.upset_detection !== undefined) {
    upsetCard.style.display = '';
    setText('statUpset', summary.upset_detection + '%');
  } else {
    upsetCard.style.display = 'none';
  }

  // Charts
  renderChart('chartPnl',         charts.cumulative_pnl);
  renderChart('chartConfidence',  charts.accuracy_by_confidence);
  renderChart('chartWeightClass', charts.roi_by_weight_class);
  renderChart('chartMonthly',     charts.monthly_roi);
  renderChart('chartYearly',      charts.yearly_roi);

  // Tables
  allFights  = Array.isArray(fights)  ? fights  : [];
  allSkipped = Array.isArray(skipped) ? skipped : [];
  allEvents  = Array.isArray(events)  ? events  : [];
  sortState  = { col: null, dir: 1 };
  switchTab(currentTab);
}

/* ── Coverage Bar ───────────────────────────────────────────────────────────── */
function renderCoverageBar(cov) {
  const el = document.getElementById('coverageBar');
  el.classList.remove('hidden');
  el.innerHTML = `
    <span class="cov-item"><strong>${cov.total_with_odds}</strong> fights with odds</span>
    <span class="cov-sep">→</span>
    <span class="cov-item cov-selected"><strong>${cov.selected}</strong> selected</span>
    <span class="cov-sep">+</span>
    <span class="cov-item cov-skipped"><strong>${cov.skipped_edge}</strong> skipped (edge &lt; threshold)</span>
    <span class="cov-sep">+</span>
    <span class="cov-item cov-nodata"><strong>${cov.no_odds}</strong> no odds data</span>
  `;
}

/* ── Tab Switching ───────────────────────────────────────────────────────────── */
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.getElementById('viewSelected').classList.toggle('hidden', tab !== 'selected');
  document.getElementById('viewEvents').classList.toggle('hidden', tab !== 'events');
  document.getElementById('viewSkipped').classList.toggle('hidden', tab !== 'skipped');

  const q = document.getElementById('tableSearch').value.toLowerCase();
  if (tab === 'selected') renderFightTable(applyFightFilter(allFights, q));
  if (tab === 'events')   renderEventTable(applyEventFilter(allEvents, q));
  if (tab === 'skipped')  renderSkippedTable(applyFightFilter(allSkipped, q));
}

function applySearch(q) {
  if (currentTab === 'selected') renderFightTable(applyFightFilter(allFights, q));
  if (currentTab === 'events')   renderEventTable(applyEventFilter(allEvents, q));
  if (currentTab === 'skipped')  renderSkippedTable(applyFightFilter(allSkipped, q));
}

function applyFightFilter(rows, q) {
  if (!q) return rows;
  return rows.filter(r =>
    (r.bet_on  || '').toLowerCase().includes(q) ||
    (r.against || '').toLowerCase().includes(q) ||
    (r.class   || '').toLowerCase().includes(q) ||
    (r.event   || '').toLowerCase().includes(q)
  );
}

function applyEventFilter(rows, q) {
  if (!q) return rows;
  return rows.filter(r => (r.event || '').toLowerCase().includes(q));
}

function renderChart(containerId, chartJson) {
  const fig = JSON.parse(chartJson);
  Plotly.react(containerId, fig.data, fig.layout, { responsive: true, displayModeBar: false });
}

/* ── Fight Table ─────────────────────────────────────────────────────────────── */
function renderFightTable(rows) {
  const tbody = document.getElementById('fightTableBody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;color:var(--text-dim);padding:2rem">No fights to display</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const isWin    = r.result === 'WIN';
    const inSample = r.sample === 'IN';
    const badge    = inSample
      ? `<span class="badge-insample" title="In-sample">IN</span>`
      : `<span class="badge-outsample">OOS</span>`;
    return `
      <tr class="${inSample ? 'row-insample' : ''}">
        <td>${r.date}</td>
        <td class="cell-event" title="${r.event}">${shortEvent(r.event)}</td>
        <td style="font-weight:500">${r.bet_on}</td>
        <td style="color:var(--text-muted)">${r.against}</td>
        <td style="color:var(--text-muted)">${r.class}</td>
        <td>${r['mkt%']}%</td>
        <td>${r['mdl%']}%</td>
        <td class="${r['edge%'] >= 10 ? 'cell-pos' : ''}">${r['edge%']}%</td>
        <td>${fmtOdds(r.odds)}</td>
        <td class="${isWin ? 'cell-win' : 'cell-loss'}">${r.result}</td>
        <td class="${r.profit >= 0 ? 'cell-pos' : 'cell-neg'}">${fmtDollar(r.profit)}</td>
        <td class="${r.cumulative >= 0 ? 'cell-pos' : 'cell-neg'}">${fmtDollar(r.cumulative)}</td>
        <td>${badge}</td>
      </tr>`;
  }).join('');
}

/* ── Event Table ─────────────────────────────────────────────────────────────── */
function renderEventTable(rows) {
  const tbody = document.getElementById('eventTableBody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-dim);padding:2rem">No events to display</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const roiClass = r.roi >= 0 ? 'cell-pos' : 'cell-neg';
    const pnlClass = r.pnl >= 0 ? 'cell-pos' : 'cell-neg';
    return `
      <tr>
        <td>${r.date}</td>
        <td style="font-weight:500">${r.event}</td>
        <td>${r.bets}</td>
        <td>${r.wins}</td>
        <td>${r.accuracy}%</td>
        <td class="${pnlClass}">${fmtDollar(r.pnl)}</td>
        <td class="${roiClass}">${r.roi >= 0 ? '+' : ''}${r.roi}%</td>
      </tr>`;
  }).join('');
}

/* ── Skipped Table ───────────────────────────────────────────────────────────── */
function renderSkippedTable(rows) {
  const tbody = document.getElementById('skippedTableBody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text-dim);padding:2rem">No skipped fights</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const inSample = r.sample === 'IN';
    const badge    = inSample
      ? `<span class="badge-insample">IN</span>`
      : `<span class="badge-outsample">OOS</span>`;
    return `
      <tr class="${inSample ? 'row-insample' : ''}">
        <td>${r.date}</td>
        <td class="cell-event" title="${r.event}">${shortEvent(r.event)}</td>
        <td style="font-weight:500">${r.bet_on}</td>
        <td style="color:var(--text-muted)">${r.against}</td>
        <td style="color:var(--text-muted)">${r.class}</td>
        <td>${r['mkt%']}%</td>
        <td>${r['mdl%']}%</td>
        <td class="cell-neg">${r['edge%']}%</td>
        <td>${fmtOdds(r.odds)}</td>
        <td>${badge}</td>
      </tr>`;
  }).join('');
}

function sortTable(col, thEl) {
  if (sortState.col === col) sortState.dir *= -1;
  else { sortState.col = col; sortState.dir = 1; }

  document.querySelectorAll('thead th').forEach(th => th.classList.remove('sort-asc', 'sort-desc'));
  if (thEl) thEl.classList.add(sortState.dir === 1 ? 'sort-asc' : 'sort-desc');

  const q = document.getElementById('tableSearch').value.toLowerCase();
  const sortFn = (a, b) => {
    let av = a[col], bv = b[col];
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    return av < bv ? -sortState.dir : av > bv ? sortState.dir : 0;
  };
  if (currentTab === 'selected') renderFightTable(applyFightFilter([...allFights].sort(sortFn), q));
  if (currentTab === 'events')   renderEventTable(applyEventFilter([...allEvents].sort(sortFn), q));
  if (currentTab === 'skipped')  renderSkippedTable(applyFightFilter([...allSkipped].sort(sortFn), q));
}

/* ── Export CSV ─────────────────────────────────────────────────────────────── */
function exportCSV() {
  const data = currentTab === 'events' ? allEvents
             : currentTab === 'skipped' ? allSkipped
             : allFights;
  if (!data.length) return;
  const cols = Object.keys(data[0]);
  const rows = [cols.join(','), ...data.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(','))];
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `ufc_backtest_${currentTab}_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ── Helpers ────────────────────────────────────────────────────────────────── */
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setTextWithColor(id, val, numeric) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  el.className = 'stat-value';
  if (numeric > 0)  el.classList.add('positive');
  if (numeric < 0)  el.classList.add('negative');
}

function fmtDollar(n) {
  const abs = Math.abs(n);
  const formatted = abs >= 1000
    ? (abs / 1000).toFixed(1) + 'k'
    : abs.toFixed(0);
  return (n >= 0 ? '+$' : '-$') + formatted;
}

function fmtOdds(n) {
  return n > 0 ? '+' + n : String(n);
}

function shortEvent(name) {
  if (!name) return '';
  // Strip "UFC Fight Night: " prefix for compactness, keep the rest
  return name.replace(/^UFC Fight Night:\s*/i, 'FN: ')
             .replace(/^UFC\s+/i, 'UFC ');
}

function showError(msg) {
  const el = document.getElementById('errorBanner');
  el.textContent = msg;
  el.classList.remove('hidden');
}

function hideError() {
  const el = document.getElementById('errorBanner');
  el.classList.add('hidden');
  el.style.background = '';
  el.style.borderColor = '';
  el.style.color = '';
}

function showInsampleBanner(msg, nIn, nOut) {
  const el = document.getElementById('insampleBanner');
  el.innerHTML = `
    <strong>⚠ In-Sample Data Detected</strong><br>
    <span>${msg}</span><br>
    <small style="opacity:.8">Out-of-sample bets: <strong>${nOut}</strong> &nbsp;|&nbsp; In-sample bets (inflated): <strong>${nIn}</strong></small>
  `;
  el.classList.remove('hidden');
}

function hideInsampleBanner() {
  document.getElementById('insampleBanner').classList.add('hidden');
}
