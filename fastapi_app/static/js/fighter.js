const $fighterSearch = document.getElementById('fighterSearch');
const $searchDropdown = document.getElementById('searchDropdown');
const $compareSearch = document.getElementById('compareSearch');
const $compareDropdown = document.getElementById('compareDropdown');
const $compareBtn = document.getElementById('compareBtn');
const $loading = document.getElementById('loadingState');
const $error = document.getElementById('errorState');
const $profile = document.getElementById('profileContent');
const $header = document.getElementById('profileHeader');
const $summaryGrid = document.getElementById('summaryGrid');
const $recentFormCard = document.getElementById('recentFormCard');
const $physGrid = document.getElementById('physGrid');
const $statsGrid = document.getElementById('statsGrid');
const $history = document.getElementById('historyTableBody');
const $compareSection = document.getElementById('compareSection');
const $compareStatus = document.getElementById('compareStatus');
const $comparePrediction = document.getElementById('comparePrediction');
const $compareTableWrap = document.getElementById('compareTableWrap');
const $eloChart = document.getElementById('eloChart');
const $eloEmpty = document.getElementById('eloEmpty');
const $eloSubtitle = document.getElementById('eloSubtitle');

let currentFighter = null;
let compareFighter = null;
const searchTimers = new WeakMap();

const COMPARE_METRICS = [
  { key: 'elo_current', label: 'ELO', format: formatInt },
  { key: 'age', label: 'Age', format: formatInt },
  { key: 'height_inches', label: 'Height', format: v => formatNumber(v, '"') },
  { key: 'weight_lbs', label: 'Weight', format: v => formatNumber(v, ' lbs') },
  { key: 'reach_inches', label: 'Reach', format: v => formatNumber(v, '"') },
  { key: 'sig_strikes_landed_per_min', label: 'SLpM', format: v => formatNumber(v) },
  { key: 'sig_strikes_absorbed_per_min', label: 'SApM', format: v => formatNumber(v) },
  { key: 'striking_accuracy_pct', label: 'Str Acc', format: formatPercent },
  { key: 'striking_defense_pct', label: 'Str Def', format: formatPercent },
  { key: 'takedown_avg_per_15min', label: 'TD Avg', format: v => formatNumber(v) },
  { key: 'takedown_accuracy_pct', label: 'TD Acc', format: formatPercent },
  { key: 'takedown_defense_pct', label: 'TD Def', format: formatPercent },
  { key: 'submission_avg_per_15min', label: 'Sub Avg', format: v => formatNumber(v) },
];

wireAutocomplete($fighterSearch, $searchDropdown, async (name) => {
  await loadFighter(name);
});

wireAutocomplete($compareSearch, $compareDropdown, async (name) => {
  compareFighter = name;
  if (currentFighter && currentFighter.name !== name) {
    await loadComparison(name);
  }
});

$compareBtn.addEventListener('click', async () => {
  const name = $compareSearch.value.trim();
  if (!name || !currentFighter) return;
  compareFighter = name;
  await loadComparison(name);
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrapper')) {
    $searchDropdown.classList.add('hidden');
    $compareDropdown.classList.add('hidden');
  }
});

async function wireAutocomplete(input, dropdown, onSelect) {
  input.addEventListener('input', () => {
    clearTimeout(searchTimers.get(input));
    const q = input.value.trim();
    if (q.length < 2) {
      dropdown.classList.add('hidden');
      return;
    }
    const timer = setTimeout(() => doSearch(q, dropdown, onSelect), 200);
    searchTimers.set(input, timer);
  });

  input.addEventListener('keydown', async (e) => {
    if (e.key === 'Escape') {
      dropdown.classList.add('hidden');
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      const name = input.value.trim();
      if (name) await onSelect(name);
      dropdown.classList.add('hidden');
    }
  });
}

async function doSearch(q, dropdown, onSelect) {
  try {
    const res = await fetch(`/api/db/fighters/search?q=${encodeURIComponent(q)}`);
    const items = await res.json();
    renderDropdown(items, dropdown, onSelect);
  } catch {
    dropdown.classList.add('hidden');
  }
}

function renderDropdown(items, dropdown, onSelect) {
  if (!items.length) {
    dropdown.classList.add('hidden');
    return;
  }
  dropdown.innerHTML = items.map(f => `
    <div class="search-item" data-name="${escAttr(f.name)}">
      <span>${esc(f.name)}</span>
      <span class="search-record">${esc(f.record)}</span>
    </div>
  `).join('');
  dropdown.classList.remove('hidden');

  dropdown.querySelectorAll('.search-item').forEach(el => {
    el.addEventListener('click', async () => {
      const name = el.dataset.name;
      const input = dropdown.id === 'searchDropdown' ? $fighterSearch : $compareSearch;
      input.value = name;
      dropdown.classList.add('hidden');
      await onSelect(name);
    });
  });
}

async function loadFighter(name) {
  setLoading(true);
  hideError();
  $profile.classList.add('hidden');

  try {
    const res = await fetch(`/api/db/fighter/${encodeURIComponent(name)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Fighter not found');
    currentFighter = data;
    $fighterSearch.value = data.name;
    renderProfile(data);
    $profile.classList.remove('hidden');
    // Chart was drawn while the profile was hidden (0-width); resize to fill now.
    if (typeof Plotly !== 'undefined' && $eloChart.data) {
      Plotly.Plots.resize($eloChart);
    }
    if (compareFighter && compareFighter !== data.name) {
      await loadComparison(compareFighter);
    } else {
      resetComparison();
    }
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

function renderProfile(fighter) {
  const nicknameHtml = fighter.nickname ? `<div class="nickname">"${esc(fighter.nickname)}"</div>` : '';
  const recentForm = (fighter.recent_form || []).map(result => {
    const cls = result === 'W' ? 'win' : result === 'L' ? 'loss' : 'other';
    return `<span class="form-pill ${cls}">${result}</span>`;
  }).join('');

  $header.innerHTML = `
    <div class="profile-title">
      <div>
        <h1>${esc(fighter.name)}</h1>
        ${nicknameHtml}
      </div>
      <div class="record-stack">
        <div class="record-chip">Record ${esc(fighter.record)}</div>
        <div class="record-subtext">UFC ${fighter.ufc_record.wins}-${fighter.ufc_record.losses}-${fighter.ufc_record.draws}</div>
      </div>
    </div>
  `;

  $summaryGrid.innerHTML = [
    { label: 'ELO Rating', value: formatEloSummary(fighter.elo_current, fighter.elo_peak) },
    { label: 'Overall Win Rate', value: formatPercent(fighter.win_rate_pct) },
    { label: 'UFC Win Rate', value: formatPercent(fighter.ufc_win_rate_pct) },
    { label: 'UFC Bouts', value: formatInt(fighter.ufc_bout_count) },
    { label: 'Age', value: formatInt(fighter.age) },
    { label: 'Stance', value: fighter.stance || '—' },
  ].map(item => `
    <div class="summary-card">
      <div class="summary-label">${item.label}</div>
      <div class="summary-value">${esc(item.value)}</div>
    </div>
  `).join('');

  $recentFormCard.innerHTML = `
    <div class="section-header">
      <h3>Recent form</h3>
      <div class="section-subtitle">Last five UFC results, with overall and UFC records shown once for context.</div>
    </div>
    <div class="glance-grid">
      <div class="glance-item">
        <span class="glance-label">Last 5</span>
        <div class="form-strip">${recentForm || '<span class="muted">—</span>'}</div>
      </div>
      <div class="glance-item">
        <span class="glance-label">Overall</span>
        <strong>${fighter.overall_record.wins}-${fighter.overall_record.losses}-${fighter.overall_record.draws}</strong>
      </div>
      <div class="glance-item">
        <span class="glance-label">UFC</span>
        <strong>${fighter.ufc_record.wins}-${fighter.ufc_record.losses}-${fighter.ufc_record.draws}</strong>
      </div>
      <div class="glance-item">
        <span class="glance-label">Nickname</span>
        <strong>${esc(fighter.nickname || '—')}</strong>
      </div>
    </div>
  `;

  $physGrid.innerHTML = [
    { label: 'Height', value: formatNumber(fighter.height_inches, '"') },
    { label: 'Weight', value: formatNumber(fighter.weight_lbs, ' lbs') },
    { label: 'Reach', value: formatNumber(fighter.reach_inches, '"') },
  ].map(item => `
    <div class="phys-card">
      <div class="phys-label">${item.label}</div>
      <div class="phys-value">${esc(item.value)}</div>
    </div>
  `).join('');

  renderEloChart([{ name: fighter.name, history: fighter.elo_history }]);

  const stats = [
    { label: 'Strikes Landed / Min', value: formatNumber(fighter.sig_strikes_landed_per_min) },
    { label: 'Striking Accuracy', value: formatPercent(fighter.striking_accuracy_pct) },
    { label: 'Strikes Absorbed / Min', value: formatNumber(fighter.sig_strikes_absorbed_per_min) },
    { label: 'Striking Defense', value: formatPercent(fighter.striking_defense_pct) },
    { label: 'Takedowns / 15', value: formatNumber(fighter.takedown_avg_per_15min) },
    { label: 'Takedown Accuracy', value: formatPercent(fighter.takedown_accuracy_pct) },
    { label: 'Takedown Defense', value: formatPercent(fighter.takedown_defense_pct) },
    { label: 'Submissions / 15', value: formatNumber(fighter.submission_avg_per_15min) },
  ];

  $statsGrid.innerHTML = stats.map(item => `
    <div class="stat-card">
      <div class="stat-card-label">${item.label}</div>
      <div class="stat-card-value">${esc(item.value)}</div>
    </div>
  `).join('');

  $history.innerHTML = '';
  for (const fight of fighter.fight_history) {
    const resultClass = fight.result === 'W' ? 'cell-win' :
      fight.result === 'L' ? 'cell-loss' :
      fight.result === 'D' ? 'cell-draw' :
      fight.result === 'NC' ? 'cell-nc' : '';

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${esc(fight.date || '—')}</td>
      <td class="event-cell">${esc(fight.event || '—')}</td>
      <td><a class="opponent-link" data-name="${escAttr(fight.opponent)}">${esc(fight.opponent)}</a></td>
      <td class="${resultClass}">${esc(fight.result)}</td>
      <td>${esc(fight.method || '—')}</td>
      <td>${esc(fight.round || '—')}</td>
      <td>${esc(fight.closing_odds || '—')}</td>
    `;
    $history.appendChild(row);
  }

  $history.querySelectorAll('.opponent-link').forEach(link => {
    link.addEventListener('click', async () => {
      const name = link.dataset.name;
      $fighterSearch.value = name;
      await loadFighter(name);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

async function loadComparison(name) {
  if (!currentFighter) return;
  if (name === currentFighter.name) {
    resetComparison('Choose a different fighter to compare.');
    return;
  }

  $compareSection.classList.remove('hidden');
  $compareStatus.textContent = 'Loading matchup...';
  $comparePrediction.innerHTML = '';
  $compareTableWrap.innerHTML = '';

  try {
    const [profileRes, predictRes] = await Promise.all([
      fetch(`/api/db/fighter/${encodeURIComponent(name)}`),
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fighter1: currentFighter.name,
          fighter2: name,
        }),
      }),
    ]);

    const compareProfile = await profileRes.json();
    const prediction = await predictRes.json();
    if (!profileRes.ok) throw new Error(compareProfile.detail || 'Comparison fighter not found');
    if (!predictRes.ok) throw new Error(prediction.detail || 'Prediction failed');

    compareFighter = compareProfile.name;
    $compareSearch.value = compareProfile.name;
    renderComparison(compareProfile, prediction);
    $compareSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    resetComparison(err.message);
  }
}

function renderComparison(other, prediction) {
  $compareStatus.textContent = 'Neutral-line model comparison';
  $comparePrediction.innerHTML = renderMatchupPanel(currentFighter, other, prediction);
  $compareTableWrap.innerHTML = '';
  renderEloChart([
    { name: currentFighter.name, history: currentFighter.elo_history },
    { name: other.name, history: other.elo_history },
  ]);
}

function renderCompareRow(left, right, metric) {
  const leftValue = left[metric.key];
  const rightValue = right[metric.key];
  const winner = compareMetric(leftValue, rightValue);
  return `<tr>
    <td class="mp-stat-val ${winner === 'left' ? 'mp-stat-edge' : ''}">${esc(metric.format(leftValue))}</td>
    <td class="mp-stat-label">${metric.label}</td>
    <td class="mp-stat-val right ${winner === 'right' ? 'mp-stat-edge' : ''}">${esc(metric.format(rightValue))}</td>
  </tr>`;
}

function compareMetric(left, right) {
  const a = typeof left === 'number' ? left : null;
  const b = typeof right === 'number' ? right : null;
  if (a == null || b == null || a === b) return '';
  return a > b ? 'left' : 'right';
}

function resetComparison(message = 'Select another fighter to compare.') {
  $compareSection.classList.add('hidden');
  $compareStatus.textContent = message;
  $comparePrediction.innerHTML = '';
  $compareTableWrap.innerHTML = '';
  if (currentFighter) {
    renderEloChart([{ name: currentFighter.name, history: currentFighter.elo_history }]);
  }
}

function renderMatchupPanel(left, right, prediction) {
  const thinData = prediction.thin_data_warning ? '<span class="mp-meta-chip warning">Thin data</span>' : '';
  return `
    <div class="matchup-panel-inner fighter-matchup-panel">
      <div class="mp-header">
        <span class="mp-name">${esc(left.name)}</span>
        <span class="mp-vs">VS</span>
        <span class="mp-name right">${esc(right.name)}</span>
      </div>
      <div class="mp-subheader">
        <span>${esc(left.record)} · ${left.age ?? '?'}yo · ${esc(left.stance || '—')} · ${formatNumber(left.reach_inches, '"')} reach</span>
        <span>${esc(right.record)} · ${right.age ?? '?'}yo · ${esc(right.stance || '—')} · ${formatNumber(right.reach_inches, '"')} reach</span>
      </div>

      <div class="probability-row fighter-probability-row">
        <div class="probability-side">
          <span>${esc(left.name)}</span>
          <strong>${prediction.model_prob_f1.toFixed(1)}%</strong>
        </div>
        <div class="probability-bar">
          <div class="probability-fill left" style="width:${prediction.model_prob_f1}%"></div>
          <div class="probability-fill right" style="width:${prediction.model_prob_f2}%"></div>
        </div>
        <div class="probability-side align-right">
          <span>${esc(right.name)}</span>
          <strong>${prediction.model_prob_f2.toFixed(1)}%</strong>
        </div>
      </div>

      <div class="fighter-matchup-meta">
        <span class="mp-meta-chip pick">Pick: ${esc(prediction.model_pick)}</span>
        <span class="mp-meta-chip">Edge ${formatSignedPercent(prediction.edge)}</span>
        ${thinData}
      </div>

      <table class="mp-stats-table">
        <tbody>
          ${COMPARE_METRICS.map(metric => renderCompareRow(left, right, metric)).join('')}
        </tbody>
      </table>

      <div class="mp-recent">
        <div class="mp-recent-col">${renderRecentFights(left.fight_history, left.name)}</div>
        <div class="mp-recent-label">Recent</div>
        <div class="mp-recent-col right">${renderRecentFights(right.fight_history, right.name)}</div>
      </div>
    </div>
  `;
}

function renderRecentFights(fights, fighterName) {
  const recent = (fights || []).slice(0, 4);
  if (!recent.length) return '<div class="mp-no-data">No recent fights</div>';
  return recent.map(fight => {
    const cls = fight.result === 'W' ? 'mp-r-w' : fight.result === 'L' ? 'mp-r-l' : 'mp-r-nc';
    const odds = fight.closing_odds ? `<span class="mp-r-odds">${esc(fight.closing_odds)}</span>` : '';
    return `<div class="mp-recent-row">
      <span class="mp-r-badge ${cls}">${esc(fight.result || '—')}</span>
      <span class="mp-r-opp">${esc(fight.opponent || '—')}</span>
      ${odds}
    </div>`;
  }).join('');
}

function setLoading(active) {
  $loading.classList.toggle('hidden', !active);
}

function showError(message) {
  $error.textContent = message;
  $error.classList.remove('hidden');
}

function hideError() {
  $error.classList.add('hidden');
}

function formatEloSummary(current, peak) {
  if (current === null || current === undefined) return '—';
  const peakPart = (peak !== null && peak !== undefined && peak !== current) ? ` (peak ${peak})` : '';
  return `${current}${peakPart}`;
}

const ELO_COLORS = ['#4f9cf9', '#f0883e'];
const ELO_WIN = '#3fb950';
const ELO_LOSS = '#f85149';

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// Convert an elo_history entry to an {x: Date, y: elo} point.
function eloPoint(entry, lastRealDate) {
  let x;
  if (entry.date === 'current') {
    // Plot the current (post-last-fight) rating shortly after the last bout.
    const base = entry.after_date ? new Date(entry.after_date) : lastRealDate;
    x = base ? new Date(base.getTime() + 45 * 864e5) : new Date();
  } else {
    x = new Date(entry.date);
  }
  return { x, y: entry.elo, entry };
}

function renderEloChart(fighters) {
  if (typeof Plotly === 'undefined') return;

  const traces = [];
  let fightersPlotted = 0;
  fighters.forEach((f, idx) => {
    const history = (f.history || []).filter(h => h && h.elo !== null && h.elo !== undefined);
    if (!history.length) return;
    fightersPlotted += 1;
    const realDates = history.filter(h => h.date !== 'current').map(h => new Date(h.date));
    const lastRealDate = realDates.length ? new Date(Math.max(...realDates.map(d => d.getTime()))) : null;
    const points = history.map(h => eloPoint(h, lastRealDate)).sort((a, b) => a.x - b.x);
    const color = ELO_COLORS[idx % ELO_COLORS.length];
    const markerColors = points.map(p => {
      if (p.entry.result === 'W') return ELO_WIN;
      if (p.entry.result === 'L') return ELO_LOSS;
      return color;
    });

    // Opponent pre-fight ELO at each bout (skip the trailing "current" point).
    const oppPoints = points.filter(p => p.entry.date !== 'current'
      && p.entry.opp_elo !== null && p.entry.opp_elo !== undefined);

    // Dotted connectors from the fighter's ELO to the opponent's ELO per bout,
    // so the rating gap at each fight is easy to read.
    const connX = [];
    const connY = [];
    oppPoints.forEach(p => {
      connX.push(p.x, p.x, null);
      connY.push(p.y, p.entry.opp_elo, null);
    });
    if (connX.length) {
      traces.push({
        type: 'scatter', mode: 'lines', showlegend: false, hoverinfo: 'skip',
        x: connX, y: connY,
        line: { color, width: 1, dash: 'dot' }, opacity: 0.4,
      });
    }

    // Opponent markers (hollow diamonds in the fighter's colour).
    if (oppPoints.length) {
      traces.push({
        type: 'scatter', mode: 'markers', showlegend: false,
        name: `${f.name} — opponents`,
        x: oppPoints.map(p => p.x),
        y: oppPoints.map(p => p.entry.opp_elo),
        marker: { color, size: 9, symbol: 'diamond-open', line: { color, width: 1.5 } },
        hovertemplate: oppPoints.map(p =>
          `Opponent: ${p.entry.opponent || '?'}<br>Their ELO %{y}<extra></extra>`),
      });
    }

    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: f.name,
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      line: { color, width: 2, shape: 'hv' },
      marker: {
        color: markerColors,
        size: 9,
        line: { color: cssVar('--surface', '#181c28'), width: 1.5 },
      },
      hovertemplate: points.map(p => {
        if (p.entry.date === 'current') return `<b>${f.name}</b><br>Current ELO %{y}<extra></extra>`;
        const res = p.entry.result ? ` (${p.entry.result})` : '';
        const oppElo = (p.entry.opp_elo !== null && p.entry.opp_elo !== undefined)
          ? `<br>opp ELO ${p.entry.opp_elo}` : '';
        const opp = p.entry.opponent ? `<br>vs ${p.entry.opponent}${res}${oppElo}` : '';
        return `<b>${f.name}</b><br>%{x|%b %Y} · ELO %{y}${opp}<extra></extra>`;
      }),
    });
  });

  if (!traces.length) {
    $eloChart.classList.add('hidden');
    $eloEmpty.classList.remove('hidden');
    Plotly.purge($eloChart);
    return;
  }
  $eloChart.classList.remove('hidden');
  $eloEmpty.classList.add('hidden');
  if ($eloSubtitle) {
    const legend = ' Filled dots = fighter (green win / red loss); hollow diamonds = opponent ELO.';
    $eloSubtitle.textContent = (fightersPlotted > 1
      ? 'Cross-promotion ELO over career — both fighters overlaid.'
      : 'Cross-promotion ELO over career. Compare a fighter to overlay both curves.') + legend;
  }

  const text = cssVar('--text-secondary', '#9aa4b2');
  const grid = cssVar('--border', 'rgba(255,255,255,0.08)');
  const layout = {
    margin: { l: 52, r: 24, t: 10, b: 44 },
    height: 460,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: text, family: 'Inter, sans-serif', size: 12 },
    xaxis: { gridcolor: grid, zeroline: false, type: 'date' },
    yaxis: { gridcolor: grid, zeroline: false, title: { text: 'ELO' } },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { color: text } },
    showlegend: fightersPlotted > 1,
    hovermode: 'closest',
  };
  Plotly.react($eloChart, traces, layout, { displayModeBar: false, responsive: true });
}

function formatNumber(value, suffix = '') {
  if (value === null || value === undefined) return '—';
  return `${Number(value).toFixed(1).replace(/\.0$/, '')}${suffix}`;
}

function formatInt(value) {
  if (value === null || value === undefined) return '—';
  return `${value}`;
}

function formatPercent(value) {
  if (value === null || value === undefined) return '—';
  return `${Number(value).toFixed(1)}%`;
}

function formatSignedPercent(value) {
  if (value === null || value === undefined) return '—';
  return `${value > 0 ? '+' : ''}${Number(value).toFixed(1)}%`;
}

function esc(s) {
  if (s === null || s === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function escAttr(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
