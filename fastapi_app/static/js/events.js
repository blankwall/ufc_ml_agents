'use strict';

/* ── State ─────────────────────────────────────────────────────────────────── */
let allEvents   = [];
let activeIndex = 0;

/* ── Filter state (defaults match the controls) ────────────────────────────── */
let filters = { minEdge: null, favConf: 55, udEdge: 15 };

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const loadingEl    = document.getElementById('loadingState');
const errorEl      = document.getElementById('errorState');
const contentEl    = document.getElementById('eventsContent');
const tabsEl       = document.getElementById('eventTabs');
const panelEl      = document.getElementById('eventPanel');
const overallEl    = document.getElementById('overallStrip');
const headerStats  = document.getElementById('headerStats');

/* ── Boot ──────────────────────────────────────────────────────────────────── */
async function init() {
  try {
    const res  = await fetch('/api/events');
    if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);
    allEvents = await res.json();

    loadingEl.classList.add('hidden');

    if (!allEvents.length) {
      errorEl.textContent = 'No event data found.';
      errorEl.classList.remove('hidden');
      return;
    }

    contentEl.classList.remove('hidden');
    renderOverall();
    renderTabs();
    selectEvent(0);
    wireFilters();
  } catch (err) {
    loadingEl.classList.add('hidden');
    errorEl.textContent = `Error loading events: ${err.message}`;
    errorEl.classList.remove('hidden');
  }
  // Always return resolved so .then() chains work
}

/* ── Filter wiring ──────────────────────────────────────────────────────────── */
function wireFilters() {
  const minEdgeEl  = document.getElementById('minEdge');
  const favConfEl  = document.getElementById('favConf');
  const udEdgeEl   = document.getElementById('udEdge');
  const applyBtn   = document.getElementById('applyFilters');
  const resetBtn   = document.getElementById('resetFilters');
  const rawBtn     = document.getElementById('rawView');

  function readFilters() {
    const me = minEdgeEl.value.trim();
    filters.minEdge = me !== '' ? parseFloat(me) : null;
    filters.favConf = favConfEl.value.trim() !== '' ? parseFloat(favConfEl.value) : null;
    filters.udEdge  = udEdgeEl.value.trim()  !== '' ? parseFloat(udEdgeEl.value)  : null;
  }

  function refresh() {
    renderTabs();
    selectEvent(activeIndex);
    renderOverall();
  }

  applyBtn.addEventListener('click', () => { readFilters(); refresh(); });

  resetBtn.addEventListener('click', () => {
    minEdgeEl.value = '';
    favConfEl.value = 55;
    udEdgeEl.value  = 15;
    filters = { minEdge: null, favConf: 55, udEdge: 15 };
    refresh();
  });

  rawBtn.addEventListener('click', () => {
    minEdgeEl.value = '';
    favConfEl.value = '';
    udEdgeEl.value  = '';
    filters = { minEdge: null, favConf: null, udEdge: null };
    refresh();
  });
}

/* ── Filter predicate ───────────────────────────────────────────────────────── */
function passesFilter(f) {
  if (!f.model_prob_f1) return true;   // no prediction → always show

  const pickProb   = f.model_prob_f1 >= 50 ? f.model_prob_f1 : 100 - f.model_prob_f1;
  const mktForPick = f.model_prob_f1 >= 50 ? f.market_prob_f1 : 100 - f.market_prob_f1;
  const isFav      = mktForPick >= 50;
  const edge       = pickProb - mktForPick;

  // Global edge filter — applies to all fights when set
  if (filters.minEdge !== null && edge < filters.minEdge) return false;

  // Per-type filters (only apply when set)
  if (filters.favConf !== null && isFav  && pickProb < filters.favConf) return false;
  if (filters.udEdge  !== null && !isFav && edge     < filters.udEdge)  return false;

  return true;
}

/* ── Overall summary strip ──────────────────────────────────────────────────── */
function renderOverall() {
  let totalN = 0, totalWins = 0, totalPnl = 0;
  for (const ev of allEvents) {
    const withResult = ev.fights.filter(f => passesFilter(f) && f.correct !== null);
    totalN    += withResult.length;
    totalWins += withResult.filter(f => f.correct).length;
    totalPnl  += withResult.reduce((s, f) => s + (f.pnl || 0), 0);
  }
  const totalBets = totalN;
  const acc = totalN ? ((totalWins / totalN) * 100).toFixed(1) : '—';
  const roi = totalBets ? ((totalPnl / (totalBets * 100)) * 100).toFixed(1) : '—';

  const roiClass   = parseFloat(roi) > 0 ? 'pos' : parseFloat(roi) < 0 ? 'neg' : 'neut';
  const pnlClass   = totalPnl > 0 ? 'pos' : totalPnl < 0 ? 'neg' : 'neut';
  const pnlFmt     = totalPnl >= 0 ? `+$${totalPnl.toFixed(0)}` : `-$${Math.abs(totalPnl).toFixed(0)}`;

  overallEl.innerHTML = `
    <div class="overall-stat">
      <span class="label">Events</span>
      <span class="value">${allEvents.length}</span>
    </div>
    <div class="overall-divider"></div>
    <div class="overall-stat">
      <span class="label">Fights tracked</span>
      <span class="value">${totalN}</span>
    </div>
    <div class="overall-stat">
      <span class="label">Model accuracy</span>
      <span class="value ${roiClass}">${acc}%</span>
    </div>
    <div class="overall-divider"></div>
    <div class="overall-stat">
      <span class="label">ROI (flat $100)</span>
      <span class="value ${roiClass}">${roi !== '—' ? roi + '%' : '—'}</span>
    </div>
    <div class="overall-stat">
      <span class="label">Total P&amp;L</span>
      <span class="value ${pnlClass}">${pnlFmt}</span>
    </div>
  `;

  if (headerStats) {
    headerStats.textContent = `${totalN} fights · ${acc}% acc`;
  }
}

/* ── Event tabs ──────────────────────────────────────────────────────────────── */
function renderTabs() {
  tabsEl.innerHTML = allEvents.map((ev, i) => {
    const roi = filteredRoi(ev);
    const cls = roi !== null ? (roi > 0 ? 'pos' : roi < 0 ? 'neg' : '') : '';
    const roiStr = roi !== null ? (roi >= 0 ? `+${roi}%` : `${roi}%`) : '?';
    const label = shortEventName(ev.event_name);
    const userCls = ev.source_type === 'user_added' ? ' user-added' : '';
    return `<button class="event-tab${userCls}" onclick="selectEvent(${i})" id="etab-${i}">
      ${label}${ev.source_type === 'user_added' ? '<span class="ua-dot" title="User-added event">●</span>' : ''}
      <span class="tab-roi ${cls}">${roiStr}</span>
    </button>`;
  }).join('');
}

function filteredRoi(ev) {
  const bets = ev.fights.filter(f => passesFilter(f) && f.correct !== null);
  if (!bets.length) return null;
  const pnl = bets.reduce((s, f) => s + (f.pnl || 0), 0);
  return parseFloat((pnl / (bets.length * 100) * 100).toFixed(1));
}

function shortEventName(name) {
  // "UFC 324: Gaethje vs. Pimblett" → "UFC 324"
  if (!name) return 'Event';
  const colon = name.indexOf(':');
  if (colon !== -1) return name.slice(0, colon).trim();
  return name.length > 18 ? name.slice(0, 18) + '…' : name;
}

/* ── Select + render event panel ────────────────────────────────────────────── */
function selectEvent(idx) {
  activeIndex = idx;

  // Highlight active tab
  document.querySelectorAll('.event-tab').forEach((btn, i) => {
    btn.classList.toggle('active', i === idx);
  });

  const ev = allEvents[idx];
  if (!ev) return;

  // Recompute stats for filtered fights only
  const filtered    = ev.fights.filter(passesFilter);
  const withResult  = filtered.filter(f => f.correct !== null);
  const wins        = withResult.filter(f => f.correct).length;
  const totalPnl    = withResult.reduce((s, f) => s + (f.pnl || 0), 0);
  const n           = withResult.length;
  const accuracy    = n ? (wins / n * 100).toFixed(1) : null;
  const roi         = n ? (totalPnl / (n * 100) * 100).toFixed(1) : null;

  const accFmt    = accuracy !== null ? `${accuracy}%`  : '—';
  const roiFmt    = roi      !== null ? (roi >= 0 ? `+${roi}%` : `${roi}%`) : '—';
  const pnlFmt    = totalPnl !== 0    ? (totalPnl >= 0 ? `+$${totalPnl.toFixed(0)}` : `-$${Math.abs(totalPnl).toFixed(0)}`) : '$0';
  const roiClass  = roi  !== null ? (roi  > 0 ? 'pos' : roi  < 0 ? 'neg' : '') : '';
  const pnlClass  = totalPnl > 0 ? 'pos' : totalPnl < 0 ? 'neg' : '';

  const fightCards = ev.fights.map(f => renderFightCard(f, passesFilter(f))).join('');

  panelEl.innerHTML = `
    <div class="event-panel">
      <div class="event-panel-header">
        <div>
          <div class="event-panel-title">${ev.event_name || 'UFC Event'}</div>
          <div class="event-panel-date">${ev.event_date}${ev.event_url ? ` · <a href="${ev.event_url}" target="_blank" style="color:var(--text-secondary);text-decoration:none;">odds source ↗</a>` : ''}</div>
        </div>
        <div class="event-stats-row">
          <div class="ev-stat">
            <span class="ev-stat-val">${filtered.length}<span style="font-size:12px;font-weight:400;color:var(--text-secondary)">/${ev.fights.length}</span></span>
            <span class="ev-stat-lbl">Fights</span>
          </div>
          <div class="ev-stat">
            <span class="ev-stat-val ${roiClass}">${accFmt}</span>
            <span class="ev-stat-lbl">Accuracy</span>
          </div>
          <div class="ev-stat">
            <span class="ev-stat-val ${roiClass}">${roiFmt}</span>
            <span class="ev-stat-lbl">ROI</span>
          </div>
          <div class="ev-stat">
            <span class="ev-stat-val ${pnlClass}">${pnlFmt}</span>
            <span class="ev-stat-lbl">P&amp;L ($100 flat)</span>
          </div>
        </div>
      </div>
      <div class="fights-list">
        ${fightCards}
      </div>
    </div>
  `;
}

/* ── Fight card ──────────────────────────────────────────────────────────────── */
function renderFightCard(f, visible = true) {
  const hasPred   = f.model_prob_f1 !== null;
  const hasResult = f.winner !== null;
  const isWin     = f.correct === true;
  const isLoss    = f.correct === false;
  const isPending = !hasResult;

  const cardClass = [
    isWin ? 'fc-win' : isLoss ? 'fc-loss' : isPending && hasPred ? 'fc-tbd' : 'fc-error',
    visible ? '' : 'fc-no-bet',
    f.source_type === 'user_added' ? 'fc-user-added' : '',
  ].filter(Boolean).join(' ');

  const badgeClass = isWin ? 'win' : isLoss ? 'loss' : isPending ? 'tbd' : 'err';
  const badgeText  = isWin ? '✓' : isLoss ? '✗' : isPending ? '?' : '!';

  // Determine model pick and winner highlight
  const modelPickF1 = hasPred && f.model_prob_f1 >= 50;
  const f1IsWinner  = hasResult && f.winner && f.fighter1.toLowerCase().replace(/['\.-]/g,'') ===
                        f.winner.toLowerCase().replace(/['\.-]/g,'').slice(0, f.fighter1.length);

  // Simplified: actual winner highlighting
  const w = (f.winner || '').toLowerCase();
  const f1IsActualWinner = hasResult && (w.includes(f.fighter1.toLowerCase().split(' ')[1] || f.fighter1.toLowerCase()));
  const f2IsActualWinner = hasResult && !f1IsActualWinner;

  const f1Class = [
    'fighter-name',
    modelPickF1 ? 'model-pick' : '',
  ].filter(Boolean).join(' ');

  const f2Class = [
    'fighter-name',
    !modelPickF1 && hasPred ? 'model-pick' : '',
  ].filter(Boolean).join(' ');

  const f1OddsFmt = f.f1_odds !== null ? fmtOdds(f.f1_odds) : '—';
  const f2OddsFmt = f.f2_odds !== null ? fmtOdds(f.f2_odds) : '—';

  // Probability bar width: model if available, else market
  const barWidth = hasPred ? f.model_prob_f1 : (f.market_prob_f1 || 50);
  const mktLabel = hasPred ? `${f.model_prob_f1}%` : '?';
  const mktLabelR = hasPred ? `${(100 - f.model_prob_f1).toFixed(1)}%` : '?';

  const edgeFmt   = f.edge !== null ? (f.edge > 0 ? `+${f.edge}%` : `${f.edge}%`) : null;
  const edgeClass = f.edge !== null ? (f.edge > 0 ? 'pos' : 'neg') : '';

  // Outcome meta
  let outcomeMeta = '';
  if (hasResult) {
    const oClass = isWin ? 'correct' : 'wrong';
    const mthd   = [f.winner, f.method, f.round ? `R${f.round}` : ''].filter(Boolean).join(' · ');
    outcomeMeta = `<span class="fight-meta-item">Result: <span class="meta-outcome ${oClass} meta-val">${mthd}</span></span>`;
  } else if (hasPred) {
    outcomeMeta = `<span class="fight-meta-item" style="color:var(--text-secondary)">Result: <span class="meta-val">TBD</span></span>`;
  }

  const pnlMeta = f.pnl !== null
    ? `<span class="fight-meta-item">P&amp;L: <span class="meta-val ${f.pnl >= 0 ? 'pos' : 'neg'}" style="color:${f.pnl >= 0 ? 'var(--accent)' : 'var(--danger)'}">${f.pnl >= 0 ? '+' : ''}$${f.pnl}</span></span>`
    : '';

  const srcMeta = f.model_source && f.model_source !== 'not_found' && f.model_source !== 'error'
    ? `<span class="fight-meta-item">Model: <span class="meta-val">${f.model_source}</span></span>`
    : '';

  const edgeMeta = edgeFmt
    ? `<span class="fight-meta-item">Edge: <span class="meta-val meta-edge ${edgeClass}">${edgeFmt}</span></span>`
    : '';

  const errorNote = f.error
    ? `<div class="fight-error">⚠ ${f.error}</div>`
    : '';

  const noBetBadge = visible ? '' : `<span class="no-bet-badge">no bet</span>`;

  return `
    <div class="fight-card ${cardClass}">
      <div class="fight-top">
        <div class="result-badge ${badgeClass}">${badgeText}</div>
        <div class="fighters-row">
          <div class="fighter-block f1">
            <span class="${f1Class}">${f.fighter1}</span>
            <span class="fighter-odds">${f1OddsFmt} · mkt ${f.market_prob_f1}%</span>
          </div>
          <span class="vs-divider">VS</span>
          <div class="fighter-block f2">
            <span class="${f2Class}">${f.fighter2}</span>
            <span class="fighter-odds">${f2OddsFmt} · mkt ${(100 - f.market_prob_f1).toFixed(1)}%</span>
          </div>
        </div>
        ${noBetBadge}
      </div>

      ${hasPred ? `
      <div class="prob-bar-wrap">
        <span class="prob-label">${mktLabel}</span>
        <div class="prob-bar">
          <div class="prob-fill-f1" style="width:${barWidth}%"></div>
        </div>
        <span class="prob-label right">${mktLabelR}</span>
      </div>` : ''}

      <div class="fight-meta">
        ${outcomeMeta}
        ${pnlMeta}
        ${edgeMeta}
        ${srcMeta}
      </div>
      ${errorNote}
    </div>
  `;
}

/* ── Helpers ─────────────────────────────────────────────────────────────────── */
function fmtOdds(n) {
  if (n === null || n === undefined) return '—';
  return n > 0 ? `+${n}` : `${n}`;
}

/* ── Add Event modal ────────────────────────────────────────────────────────── */
function wireAddEvent() {
  const fab      = document.getElementById('addEventFab');
  const overlay  = document.getElementById('addEventOverlay');
  const closeBtn = document.getElementById('addEventClose');
  const cancelBtn= document.getElementById('addEventCancel');
  const submitBtn= document.getElementById('addEventSubmit');
  const bfoInput = document.getElementById('bfoUrl');
  const statsInput = document.getElementById('ufcStatsUrl');
  const errEl    = document.getElementById('addEventError');

  function openModal()  { overlay.classList.remove('hidden'); bfoInput.focus(); }
  function closeModal() { overlay.classList.add('hidden'); errEl.classList.add('hidden'); errEl.textContent = ''; }

  fab.addEventListener('click', openModal);
  closeBtn.addEventListener('click', closeModal);
  cancelBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });

  submitBtn.addEventListener('click', async () => {
    const bfoUrl   = bfoInput.value.trim();
    const statsUrl = statsInput.value.trim();

    if (!bfoUrl) {
      showErr('BestFightOdds URL is required.');
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Scraping…';
    errEl.classList.add('hidden');

    try {
      const res = await fetch('/api/add-event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bfo_url: bfoUrl, ufc_stats_url: statsUrl || null }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || res.statusText);
      }

      const result = await res.json();
      closeModal();
      bfoInput.value   = '';
      statsInput.value = '';

      // Reload events (new event is now in user_events dir)
      await reloadEvents();
      // Jump to the newly added event tab (it will be last or close to last)
      const newIdx = allEvents.findIndex(e => e.source_type === 'user_added' &&
        (e.event_url === bfoUrl || e.event_name === result.event_name));
      if (newIdx >= 0) selectEvent(newIdx);

    } catch (err) {
      showErr(err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Scrape & Add';
    }
  });

  function showErr(msg) {
    errEl.textContent = msg;
    errEl.classList.remove('hidden');
  }
}

async function reloadEvents() {
  try {
    const res = await fetch('/api/events');
    if (!res.ok) throw new Error(`Server ${res.status}`);
    allEvents = await res.json();
    renderOverall();
    renderTabs();
    selectEvent(Math.min(activeIndex, allEvents.length - 1));
  } catch (err) {
    console.error('reloadEvents failed:', err);
  }
}

/* ── Start ─────────────────────────────────────────────────────────────────── */
init().then(() => wireAddEvent());
