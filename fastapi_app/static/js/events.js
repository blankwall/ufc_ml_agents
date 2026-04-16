'use strict';

/* ── State ─────────────────────────────────────────────────────────────────── */
let allEvents   = [];
let activeIndex = 0;

/* ── Betting config (loaded from /api/config) ──────────────────────────────── */
let bettingConfig = null;

/* ── Filter state (defaults overridden from config on boot) ────────────────── */
let filters = { minEdge: 5, favConf: 70, udConf: 53, udEdge: 10,
                favCap: -300, dogCap: 300 };

let THIN_DATA_MIN_FIGHTS = 3;

/* ── Odds cap defaults (overridden from config on boot) ──────────────────── */
let ODDS_CAP_FAV_DEFAULT = -300;
let ODDS_CAP_UD_DEFAULT  =  300;

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
    const [evRes, cfgRes] = await Promise.all([
      fetch('/api/events'),
      fetch('/api/config'),
    ]);
    if (!evRes.ok) throw new Error(`Server ${evRes.status}: ${await evRes.text()}`);
    allEvents = await evRes.json();

    if (cfgRes.ok) {
      bettingConfig = await cfgRes.json();
      _applyConfigDefaults(bettingConfig);
    }

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

/* ── Apply config defaults to filters + HTML inputs ────────────────────────── */
function _applyConfigDefaults(cfg) {
  if (!cfg) return;
  const f = cfg.filters || {};

  if (f.min_fights != null) THIN_DATA_MIN_FIGHTS = f.min_fights;
  if (f.favorite_confidence_min != null) filters.favConf = f.favorite_confidence_min * 100;
  if (f.underdog_confidence_min != null) filters.udConf  = f.underdog_confidence_min * 100;
  if (f.favorite_odds_cap != null) { filters.favCap = f.favorite_odds_cap; ODDS_CAP_FAV_DEFAULT = f.favorite_odds_cap; }
  if (f.underdog_odds_cap != null) { filters.dogCap = f.underdog_odds_cap; ODDS_CAP_UD_DEFAULT  = f.underdog_odds_cap; }

  const _setVal = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.value = v; };
  _setVal('favConf', filters.favConf);
  _setVal('udConf',  filters.udConf);
  _setVal('favCap',  filters.favCap);
  _setVal('dogCap',  filters.dogCap);
}

/* ── Bet decision: unified filter + multiplier ─────────────────────────────── */
function getBetDecision(f) {
  if (!f.model_prob_f1) return { visible: !f.error, multiplier: null };

  // Thin data filter
  if (document.getElementById('excludeThinData')?.checked) {
    const c1 = f.f1_fight_count ?? 999;
    const c2 = f.f2_fight_count ?? 999;
    if (c1 < THIN_DATA_MIN_FIGHTS || c2 < THIN_DATA_MIN_FIGHTS)
      return { visible: false, multiplier: null };
  }

  // Odds cap filter
  if (filters.favCap !== null || filters.dogCap !== null) {
    for (const o of [f.f1_odds, f.f2_odds]) {
      if (o === null || o === undefined) continue;
      if (filters.favCap !== null && o <= filters.favCap) return { visible: false, multiplier: null };
      if (filters.dogCap !== null && o >= filters.dogCap) return { visible: false, multiplier: null };
    }
  }

  const pickProb   = f.model_prob_f1 >= 50 ? f.model_prob_f1 : 100 - f.model_prob_f1;
  const mktForPick = f.model_prob_f1 >= 50 ? f.market_prob_f1 : 100 - f.market_prob_f1;
  const isFav      = mktForPick >= 50;
  const edge       = pickProb - mktForPick;

  if (filters.minEdge !== null && edge < filters.minEdge) return { visible: false, multiplier: null };
  if (filters.favConf !== null && isFav  && pickProb < filters.favConf) return { visible: false, multiplier: null };
  if (filters.udConf  !== null && !isFav && pickProb < filters.udConf)  return { visible: false, multiplier: null };
  if (filters.udEdge  !== null && !isFav && edge     < filters.udEdge)  return { visible: false, multiplier: null };

  // Multiplier from config edge buckets
  if (!bettingConfig) return { visible: true, multiplier: null };

  const edgeFrac = edge / 100;
  const buckets  = bettingConfig.edge_buckets || [];
  let multiplier = null;

  for (const b of buckets) {
    if (edgeFrac >= b.min_edge && edgeFrac < b.max_edge) {
      if (b.action === 'skip') { multiplier = null; break; }
      multiplier = b.multiplier ?? null;
      break;
    }
  }

  // WMMA rules: cap multiplier and require higher edge
  const wmma = bettingConfig.wmma_rules;
  if (wmma && wmma.enabled && f.is_wmma !== false) {
    if (edgeFrac < wmma.min_edge) multiplier = null;
    else if (multiplier !== null) multiplier = Math.min(multiplier, wmma.max_multiplier);
  }

  return { visible: true, multiplier };
}

function passesFilter(f) { return getBetDecision(f).visible; }

/* ── Filter wiring ──────────────────────────────────────────────────────────── */
function wireFilters() {
  const minEdgeEl      = document.getElementById('minEdge');
  const favConfEl      = document.getElementById('favConf');
  const udConfEl       = document.getElementById('udConf');
  const udEdgeEl       = document.getElementById('udEdge');
  const excludeThinEl   = document.getElementById('excludeThinData');
  const favCapEl        = document.getElementById('favCap');
  const dogCapEl        = document.getElementById('dogCap');
  const applyBtn       = document.getElementById('applyFilters');
  const resetBtn       = document.getElementById('resetFilters');
  const rawBtn         = document.getElementById('rawView');

  function readFilters() {
    const me = minEdgeEl.value.trim();
    filters.minEdge = me !== '' ? parseFloat(me) : null;
    filters.favConf = favConfEl.value.trim() !== '' ? parseFloat(favConfEl.value) : null;
    filters.udConf  = udConfEl.value.trim()  !== '' ? parseFloat(udConfEl.value)  : null;
    filters.udEdge  = udEdgeEl.value.trim()  !== '' ? parseFloat(udEdgeEl.value)  : null;
    filters.favCap  = favCapEl.value.trim()  !== '' ? parseInt(favCapEl.value)     : null;
    filters.dogCap  = dogCapEl.value.trim()  !== '' ? parseInt(dogCapEl.value)     : null;
  }

  function refresh() {
    renderTabs();
    selectEvent(activeIndex);
    renderOverall();
  }

  applyBtn.addEventListener('click', () => { readFilters(); refresh(); });

  excludeThinEl.addEventListener('change', () => refresh());
  favCapEl.addEventListener('change', () => { readFilters(); refresh(); });
  dogCapEl.addEventListener('change', () => { readFilters(); refresh(); });

  resetBtn.addEventListener('click', () => {
    minEdgeEl.value        = '';
    favConfEl.value        = filters.favConf ?? 70;
    udConfEl.value         = '';
    udEdgeEl.value         = 10;
    favCapEl.value         = ODDS_CAP_FAV_DEFAULT;
    dogCapEl.value         = ODDS_CAP_UD_DEFAULT;
    excludeThinEl.checked  = true;
    _applyConfigDefaults(bettingConfig);
    readFilters();
    refresh();
  });

  rawBtn.addEventListener('click', () => {
    minEdgeEl.value        = '';
    favConfEl.value        = '';
    udConfEl.value         = '';
    udEdgeEl.value         = '';
    favCapEl.value         = '';
    dogCapEl.value         = '';
    excludeThinEl.checked  = false;
    filters = { minEdge: null, favConf: null, udConf: null, udEdge: null, favCap: null, dogCap: null };
    refresh();
  });

  // Initialise filter state from HTML defaults
  readFilters();
}

/* ── Filter predicate (now delegates to getBetDecision) ────────────────────── */

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

  const fightCards = ev.fights.map(f => {
    const decision = getBetDecision(f);
    return renderFightCard(f, decision.visible, decision.multiplier);
  }).join('');

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
function renderFightCard(f, visible = true, multiplier = null) {
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

  const fightsMeta = (f.f1_fight_count !== null && f.f1_fight_count !== undefined) ||
                     (f.f2_fight_count !== null && f.f2_fight_count !== undefined)
    ? (() => {
        const c1 = f.f1_fight_count ?? '?';
        const c2 = f.f2_fight_count ?? '?';
        const warn1 = (f.f1_fight_count !== null && f.f1_fight_count < 3) ? ' fc-thin' : '';
        const warn2 = (f.f2_fight_count !== null && f.f2_fight_count < 3) ? ' fc-thin' : '';
        return `<span class="fight-meta-item">Fights in DB: <span class="meta-val${warn1}">${c1}</span> · <span class="meta-val${warn2}">${c2}</span></span>`;
      })()
    : '';

  const edgeMeta = edgeFmt
    ? `<span class="fight-meta-item">Edge: <span class="meta-val meta-edge ${edgeClass}">${edgeFmt}</span></span>`
    : '';

  const errorNote = f.error
    ? `<div class="fight-error">⚠ ${f.error}</div>`
    : '';

  const noBetBadge = visible ? '' : `<span class="no-bet-badge">no bet</span>`;

  // Bet-size badge (1x, 1.5x, 2x) — only for visible fights with a multiplier
  let betSizeBadge = '';
  if (visible && multiplier !== null) {
    const label = multiplier % 1 === 0 ? `${multiplier}x` : `${multiplier}x`;
    const tier  = multiplier >= 2 ? 'high' : multiplier >= 1.5 ? 'mid' : 'low';
    betSizeBadge = `<span class="bet-size-badge bet-size-${tier}">${label}</span>`;
  }

  return `
    <div class="fight-card ${cardClass}">
      <div class="fight-top">
        <div class="result-badge ${badgeClass}">${badgeText}</div>
        <div class="fighters-row">
          <div class="fighter-block f1">
            <span class="${f1Class} fighter-clickable" data-fighter="${f.fighter1}">${f.fighter1}</span>
            <span class="fighter-odds">${f1OddsFmt} · mkt ${f.market_prob_f1}%</span>
          </div>
          <span class="vs-divider">VS</span>
          <div class="fighter-block f2">
            <span class="${f2Class} fighter-clickable" data-fighter="${f.fighter2}">${f.fighter2}</span>
            <span class="fighter-odds">${f2OddsFmt} · mkt ${(100 - f.market_prob_f1).toFixed(1)}%</span>
          </div>
        </div>
        ${noBetBadge}
        ${betSizeBadge}
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
        ${fightsMeta}
      </div>
      ${errorNote}
      <button class="matchup-expand-btn"
              data-f1="${f.fighter1}"
              data-f2="${f.fighter2}"
              onclick="toggleMatchup(this, event)">↕ Head-to-Head</button>
      <div class="matchup-panel hidden"></div>
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

/* ── Head-to-head matchup panel ─────────────────────────────────────────────── */

async function toggleMatchup(btn, e) {
  e.stopPropagation();
  const card  = btn.closest('.fight-card');
  const panel = card.querySelector('.matchup-panel');
  const f1    = btn.dataset.f1;
  const f2    = btn.dataset.f2;

  if (!panel.classList.contains('hidden')) {
    panel.classList.add('hidden');
    btn.textContent = '↕ Head-to-Head';
    return;
  }

  btn.textContent = 'Loading…';
  btn.disabled = true;

  try {
    const res  = await fetch(`/api/matchup/${encodeURIComponent(f1)}/${encodeURIComponent(f2)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    panel.innerHTML = renderMatchupPanel(f1, data.fighter1, f2, data.fighter2);
    panel.classList.remove('hidden');
    btn.textContent = '↑ Close';
  } catch (err) {
    panel.innerHTML = `<div class="mp-error">⚠ Could not load matchup: ${err.message}</div>`;
    panel.classList.remove('hidden');
    btn.textContent = '↕ Head-to-Head';
  } finally {
    btn.disabled = false;
  }
}

async function runAiAnalysis(btn, f1, f2) {
  const panel     = btn.closest('.matchup-panel');
  const resultEl  = panel.querySelector('.ai-result');
  btn.disabled    = true;
  btn.textContent = 'Analysing…';
  resultEl.innerHTML = '<div class="ai-thinking"><span class="ai-spinner"></span> AI is thinking…</div>';

  try {
    const res = await fetch('/api/matchup/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ fighter1: f1, fighter2: f2 }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    resultEl.innerHTML = renderAiResult(data);
  } catch (err) {
    resultEl.innerHTML = `<div class="ai-error">⚠ ${err.message}</div>`;
  } finally {
    btn.disabled    = false;
    btn.textContent = '↺ Re-analyse';
  }
}

function fmtStat(v, decimals = 1, suffix = '') {
  return v == null ? '—' : `${Number(v).toFixed(decimals)}${suffix}`;
}
function fmtPct(v) {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`;
}

function renderStatRow(label, v1, v2, higherIsBetter = true) {
  const n1 = parseFloat(v1), n2 = parseFloat(v2);
  const c1 = (!isNaN(n1) && !isNaN(n2)) ? (higherIsBetter ? (n1 > n2 ? 'mp-stat-edge' : '') : (n1 < n2 ? 'mp-stat-edge' : '')) : '';
  const c2 = (!isNaN(n1) && !isNaN(n2)) ? (higherIsBetter ? (n2 > n1 ? 'mp-stat-edge' : '') : (n2 < n1 ? 'mp-stat-edge' : '')) : '';
  return `<tr>
    <td class="mp-stat-val ${c1}">${v1}</td>
    <td class="mp-stat-label">${label}</td>
    <td class="mp-stat-val right ${c2}">${v2}</td>
  </tr>`;
}

function renderRecentFights(fights, fighterName = null) {
  if (!fights || !fights.length) return '<span class="mp-no-data">No recent fights</span>';
  return fights.map(f => {
    const cls = f.result === 'W' ? 'mp-r-w' : f.result === 'L' ? 'mp-r-l' : 'mp-r-nc';
    const odds = f.close_odds ? ` <span class="mp-r-odds">${f.close_odds}</span>` : '';
    const hasFightResult = f.result && f.result !== 'N/A';
    const statsBtn = (hasFightResult && fighterName && f.opponent)
      ? ` <span class="fight-stats-link mp-r-stats"
               data-fighter="${escAttr(fighterName)}"
               data-opponent="${escAttr(f.opponent)}"
               title="View fight stats">↗</span>`
      : '';
    return `<div class="mp-recent-row">
      <span class="mp-r-badge ${cls}">${f.result}</span>
      <span class="mp-r-opp">${f.opponent}</span>${odds}${statsBtn}
    </div>`;
  }).join('');
}

function renderMatchupPanel(f1Name, f1, f2Name, f2) {
  const f1Esc = f1Name.replace(/"/g, '&quot;');
  const f2Esc = f2Name.replace(/"/g, '&quot;');

  return `
  <div class="matchup-panel-inner">
    <div class="mp-header">
      <span class="mp-name">${f1.name}</span>
      <span class="mp-vs">VS</span>
      <span class="mp-name right">${f2.name}</span>
    </div>
    <div class="mp-subheader">
      <span>${f1.record} · ${f1.age ?? '?'}yo · ${f1.stance ?? '—'} · ${fmtStat(f1.reach_inches, 1, '"')} reach</span>
      <span>${f2.record} · ${f2.age ?? '?'}yo · ${f2.stance ?? '—'} · ${fmtStat(f2.reach_inches, 1, '"')} reach</span>
    </div>

    <table class="mp-stats-table">
      <tbody>
        ${renderStatRow('Sig strikes/min', fmtStat(f1.sig_strikes_landed_per_min), fmtStat(f2.sig_strikes_landed_per_min))}
        ${renderStatRow('Striking accuracy', fmtPct(f1.striking_accuracy), fmtPct(f2.striking_accuracy))}
        ${renderStatRow('Strikes absorbed/min', fmtStat(f1.sig_strikes_absorbed_per_min), fmtStat(f2.sig_strikes_absorbed_per_min), false)}
        ${renderStatRow('Striking defense', fmtPct(f1.striking_defense), fmtPct(f2.striking_defense))}
        ${renderStatRow('Takedowns/15min', fmtStat(f1.takedown_avg_per_15min), fmtStat(f2.takedown_avg_per_15min))}
        ${renderStatRow('TD accuracy', fmtPct(f1.takedown_accuracy), fmtPct(f2.takedown_accuracy))}
        ${renderStatRow('TD defense', fmtPct(f1.takedown_defense), fmtPct(f2.takedown_defense))}
        ${renderStatRow('Submissions/15min', fmtStat(f1.submission_avg_per_15min), fmtStat(f2.submission_avg_per_15min))}
      </tbody>
    </table>

    <div class="mp-recent">
      <div class="mp-recent-col">${renderRecentFights(f1.recent_fights, f1.name)}</div>
      <div class="mp-recent-label">Recent</div>
      <div class="mp-recent-col right">${renderRecentFights(f2.recent_fights, f2.name)}</div>
    </div>

    <div class="mp-ai-section">
      <button class="mp-ai-btn"
              onclick="runAiAnalysis(this, '${f1Esc}', '${f2Esc}')">
        ✦ AI Analysis
      </button>
      <div class="ai-result"></div>
    </div>
  </div>`;
}

function renderAiResult(data) {
  if (data.error) return `<div class="ai-error">⚠ ${data.error}</div>`;

  const obs = (data.observations || data.reasons || []).map((r, i) =>
    `<li class="ai-reason"><span class="ai-reason-n">${i + 1}</span>${r}</li>`
  ).join('');

  const lean     = data.lean || data.winner || '—';
  const other    = data.other || data.loser  || '—';
  const strength = data.lean_strength || 'slight';
  const strengthLabel = { slight: 'Slight edge', moderate: 'Moderate edge', clear: 'Clear edge' }[strength] || 'Edge';
  const strengthCls   = { slight: 'ai-lean-slight', moderate: 'ai-lean-moderate', clear: 'ai-lean-clear' }[strength] || '';

  return `
  <div class="ai-card">
    <div class="ai-lean-row">
      <div class="ai-lean-names">
        <span class="ai-lean-fav">${lean}</span>
        <span class="ai-lean-sep">vs</span>
        <span class="ai-lean-other">${other}</span>
      </div>
      <span class="ai-lean-badge ${strengthCls}">${strengthLabel}</span>
    </div>
    <ul class="ai-reasons">${obs}</ul>
    <div class="ai-footer">Statistical observations only · career stats · not a prediction</div>
  </div>`;
}

/* ── Shared helpers ──────────────────────────────────────────────────────────── */
function escAttr(s) {
  return s ? s.replace(/"/g, '&quot;').replace(/'/g, '&#39;') : '';
}

/* ── Fight stats popup ───────────────────────────────────────────────────────── */

let fightStatsPopupEl  = null;
let fightStatsPopupKey = null;

function ensureFightStatsPopup() {
  if (fightStatsPopupEl) return;
  fightStatsPopupEl = document.createElement('div');
  fightStatsPopupEl.id = 'fightStatsPopup';
  fightStatsPopupEl.className = 'fight-stats-popup hidden';
  document.body.appendChild(fightStatsPopupEl);

  document.addEventListener('click', e => {
    if (
      fightStatsPopupEl &&
      !fightStatsPopupEl.classList.contains('hidden') &&
      !fightStatsPopupEl.contains(e.target) &&
      !e.target.classList.contains('fight-stats-link')
    ) {
      closeFightStatsPopup();
    }
  });
}

function closeFightStatsPopup() {
  if (fightStatsPopupEl) {
    fightStatsPopupEl.classList.add('hidden');
    fightStatsPopupKey = null;
  }
}

function positionFightStatsPopup(anchor) {
  if (!fightStatsPopupEl || !anchor) return;
  const rect   = anchor.getBoundingClientRect();
  const scrollY = window.scrollY || 0;
  const scrollX = window.scrollX || 0;
  const popupW  = 480;

  let left = rect.left + scrollX;
  let top  = rect.bottom + scrollY + 8;

  const vw = document.documentElement.clientWidth;
  if (left + popupW > vw + scrollX - 12) left = vw + scrollX - popupW - 12;
  if (left < scrollX + 8) left = scrollX + 8;

  fightStatsPopupEl.style.left = `${left}px`;
  fightStatsPopupEl.style.top  = `${top}px`;
}

async function showFightStatsPopup(fighter, opponent, anchorEl) {
  ensureFightStatsPopup();

  const key = `${fighter}::${opponent}`;
  if (fightStatsPopupKey === key && !fightStatsPopupEl.classList.contains('hidden')) {
    closeFightStatsPopup();
    return;
  }

  fightStatsPopupKey = key;
  fightStatsPopupEl.classList.remove('hidden');
  fightStatsPopupEl.innerHTML = '<div class="fsp-loading">Loading…</div>';
  positionFightStatsPopup(anchorEl);

  try {
    const res = await fetch(`/api/fight-stats/${encodeURIComponent(fighter)}/${encodeURIComponent(opponent)}`);
    if (res.status === 404) {
      fightStatsPopupEl.innerHTML = `
        <div class="fsp-inner">
          <div class="fsp-header">
            <div class="fsp-names">
              <span class="fsp-fname">${fighter}</span>
              <span class="fsp-vs">vs</span>
              <span class="fsp-oname">${opponent}</span>
            </div>
            <button class="fsp-close" onclick="closeFightStatsPopup()">✕</button>
          </div>
          <div class="fsp-no-stats">No fight stats available for this bout.</div>
        </div>`;
      positionFightStatsPopup(anchorEl);
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    fightStatsPopupEl.innerHTML = renderFightStatsCard(data);
    positionFightStatsPopup(anchorEl);
  } catch (err) {
    fightStatsPopupEl.innerHTML = `<div class="fsp-error">⚠ ${err.message}</div>`;
  }
}

function fspTotalsRow(label, fVal, oVal, higherIsBetter = true) {
  const n1 = parseFloat(fVal), n2 = parseFloat(oVal);
  const c1 = (!isNaN(n1) && !isNaN(n2)) ? (higherIsBetter ? (n1 > n2 ? 'fsp-edge' : '') : (n1 < n2 ? 'fsp-edge' : '')) : '';
  const c2 = (!isNaN(n1) && !isNaN(n2)) ? (higherIsBetter ? (n2 > n1 ? 'fsp-edge' : '') : (n2 < n1 ? 'fsp-edge' : '')) : '';
  return `<tr>
    <td class="fsp-val ${c1}">${fVal ?? '—'}</td>
    <td class="fsp-label">${label}</td>
    <td class="fsp-val right ${c2}">${oVal ?? '—'}</td>
  </tr>`;
}

function renderFightStatsCard(d) {
  const resultCls = d.result === 'W' ? 'fsp-w' : d.result === 'L' ? 'fsp-l' : 'fsp-nc';
  const method = [d.method, d.method_detail].filter(Boolean).join(' · ');
  const finish = d.round ? `Rnd ${d.round}${d.time ? ' ' + d.time : ''}` : '';

  const t  = d.fighter?.totals  || {};
  const ot = d.opponent?.totals || {};
  const s  = d.fighter?.sig_strikes  || {};
  const os = d.opponent?.sig_strikes || {};

  const totalsRows = [
    fspTotalsRow('Knockdowns',        t.knockdowns,         ot.knockdowns),
    fspTotalsRow('Sig strikes',       t.sig_strikes,        ot.sig_strikes),
    fspTotalsRow('Sig strike %',      t.sig_strike_pct,     ot.sig_strike_pct),
    fspTotalsRow('Total strikes',     t.total_strikes,      ot.total_strikes),
    fspTotalsRow('Takedowns',         t.takedowns,          ot.takedowns),
    fspTotalsRow('Sub attempts',      t.submission_attempts,ot.submission_attempts),
    fspTotalsRow('Control time',      t.control_time,       ot.control_time, false),
  ].join('');

  const sigRows = [
    fspTotalsRow('Head',    s.head_strikes,     os.head_strikes),
    fspTotalsRow('Body',    s.body_strikes,     os.body_strikes),
    fspTotalsRow('Leg',     s.leg_strikes,      os.leg_strikes),
    fspTotalsRow('Distance',s.distance_strikes, os.distance_strikes),
    fspTotalsRow('Clinch',  s.clinch_strikes,   os.clinch_strikes),
    fspTotalsRow('Ground',  s.ground_strikes,   os.ground_strikes),
  ].join('');

  const hasStats = Object.keys(t).length > 0;

  return `
  <div class="fsp-inner">
    <div class="fsp-header">
      <div class="fsp-names">
        <span class="fsp-fname"><span class="fsp-result-badge ${resultCls}">${d.result}</span> ${d.fighter?.name ?? '—'}</span>
        <span class="fsp-vs">vs</span>
        <span class="fsp-oname">${d.opponent?.name ?? '—'}</span>
      </div>
      <button class="fsp-close" onclick="closeFightStatsPopup()">✕</button>
    </div>
    <div class="fsp-meta">
      ${d.event ? `<span class="fsp-event">${d.event}</span>` : ''}
      ${method   ? `<span class="fsp-method">${method}</span>` : ''}
      ${finish   ? `<span class="fsp-finish">${finish}</span>` : ''}
    </div>
    ${hasStats ? `
    <table class="fsp-table">
      <thead><tr>
        <th class="fsp-th">${d.fighter?.name ?? ''}</th>
        <th class="fsp-th center">Totals</th>
        <th class="fsp-th right">${d.opponent?.name ?? ''}</th>
      </tr></thead>
      <tbody>${totalsRows}</tbody>
    </table>
    <div class="fsp-sig-label">Significant Strikes</div>
    <table class="fsp-table">
      <tbody>${sigRows}</tbody>
    </table>` : '<div class="fsp-no-stats">No detailed stats available for this fight.</div>'}
  </div>`;
}

// Delegated click for fight-stats-link elements
document.addEventListener('click', e => {
  const el = e.target.closest('.fight-stats-link');
  if (!el) return;
  e.stopPropagation();
  const fighter  = el.dataset.fighter;
  const opponent = el.dataset.opponent;
  if (fighter && opponent) showFightStatsPopup(fighter, opponent, el);
});

/* ── Fighter popup ───────────────────────────────────────────────────────────── */

let fighterPopupEl = null;
let fighterPopupName = null;

function ensureFighterPopup() {
  if (fighterPopupEl) return;
  fighterPopupEl = document.createElement('div');
  fighterPopupEl.id = 'fighterPopup';
  fighterPopupEl.className = 'fighter-popup hidden';
  document.body.appendChild(fighterPopupEl);

  // Close on outside click
  document.addEventListener('click', e => {
    if (
      fighterPopupEl &&
      !fighterPopupEl.classList.contains('hidden') &&
      !fighterPopupEl.contains(e.target) &&
      !e.target.classList.contains('fighter-clickable')
    ) {
      closeFighterPopup();
    }
  });
}

function closeFighterPopup() {
  if (fighterPopupEl) {
    fighterPopupEl.classList.add('hidden');
    fighterPopupName = null;
  }
}

async function showFighterPopup(name, anchorEl) {
  ensureFighterPopup();

  // Toggle off if same fighter clicked again
  if (fighterPopupName === name && !fighterPopupEl.classList.contains('hidden')) {
    closeFighterPopup();
    return;
  }

  fighterPopupName = name;
  fighterPopupEl.classList.remove('hidden');
  fighterPopupEl.innerHTML = `<div class="fp-loading">Loading…</div>`;
  positionPopup(anchorEl);

  try {
    const res = await fetch(`/api/fighter/${encodeURIComponent(name)}/recent`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderFighterPopup(data);
    positionPopup(anchorEl);
  } catch (err) {
    fighterPopupEl.innerHTML = `<div class="fp-error">⚠ ${err.message}</div>`;
  }
}

function positionPopup(anchor) {
  if (!fighterPopupEl || !anchor) return;
  const rect = anchor.getBoundingClientRect();
  const scrollY = window.scrollY || 0;
  const scrollX = window.scrollX || 0;
  const popupW  = 340;

  let left = rect.left + scrollX;
  let top  = rect.bottom + scrollY + 8;

  // Keep within viewport horizontally
  const vw = document.documentElement.clientWidth;
  if (left + popupW > vw + scrollX - 12) {
    left = vw + scrollX - popupW - 12;
  }
  if (left < scrollX + 8) left = scrollX + 8;

  fighterPopupEl.style.left = `${left}px`;
  fighterPopupEl.style.top  = `${top}px`;
}

function resultBadge(result) {
  const cls = result === 'W' ? 'fp-w' : result === 'L' ? 'fp-l' : 'fp-nc';
  return `<span class="fp-result ${cls}">${result}</span>`;
}

function renderFighterPopup(data) {
  const rows = data.fights.map(f => {
    const evText = f.event
      ? f.event.replace(/^(UFC \w+:?\s*)/, m => `<strong>${m}</strong>`)
      : '—';
    // Only show stats icon for completed fights
    const hasFightResult = f.result && f.result !== 'N/A';
    const statsCell = (hasFightResult && f.opponent)
      ? `<td class="fp-stats-cell"><span class="fight-stats-link fp-stats-icon"
               data-fighter="${escAttr(data.name)}"
               data-opponent="${escAttr(f.opponent)}"
               title="View fight stats">📊</span></td>`
      : '<td></td>';
    return `
    <tr>
      <td>${resultBadge(f.result)}</td>
      <td class="fp-opponent">${f.opponent}</td>
      <td class="fp-event">${evText}</td>
      <td class="fp-odds">${f.open_odds ?? '—'}</td>
      <td class="fp-odds">${f.close_odds ?? '—'}</td>
      ${statsCell}
    </tr>`;
  }).join('');

  fighterPopupEl.innerHTML = `
    <div class="fp-header">
      <span class="fp-name">${data.name}</span>
      <span class="fp-record">${data.record}</span>
      <button class="fp-close" onclick="closeFighterPopup()">✕</button>
    </div>
    <table class="fp-table">
      <thead>
        <tr>
          <th></th>
          <th>Opponent</th>
          <th>Event</th>
          <th>Open</th>
          <th>Close</th>
          <th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// Delegated click handler for fighter names
document.addEventListener('click', e => {
  const el = e.target.closest('.fighter-clickable');
  if (!el) return;
  e.stopPropagation();
  const name = el.dataset.fighter;
  if (name) showFighterPopup(name, el);
});

/* ── Start ─────────────────────────────────────────────────────────────────── */
init().then(() => wireAddEvent());
