'use strict';

/* ── State ─────────────────────────────────────────────────────────────────── */
let allEvents   = [];
let activeIndex = 0;
const decisionCards = new Map();
const decisionPolls = new Map();

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
    selectEvent(defaultEventIndex());
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
  if (f.edge_min != null) filters.minEdge = f.edge_min * 100;
  if (f.underdog_edge_min != null) filters.udEdge = f.underdog_edge_min * 100;
  filters.favConf = f.favorite_confidence_min != null ? f.favorite_confidence_min * 100 : null;
  if (f.underdog_confidence_min != null) filters.udConf  = f.underdog_confidence_min * 100;
  if (f.favorite_odds_cap != null) { filters.favCap = f.favorite_odds_cap; ODDS_CAP_FAV_DEFAULT = f.favorite_odds_cap; }
  if (f.underdog_odds_cap != null) { filters.dogCap = f.underdog_odds_cap; ODDS_CAP_UD_DEFAULT  = f.underdog_odds_cap; }

  const _setVal = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.value = v; };
  _setVal('minEdge', filters.minEdge);
  const favConfEl = document.getElementById('favConf');
  if (favConfEl) favConfEl.value = filters.favConf ?? '';
  _setVal('udConf',  filters.udConf);
  _setVal('udEdge',  filters.udEdge);
  _setVal('favCap',  filters.favCap);
  _setVal('dogCap',  filters.dogCap);
}

/* ── Bet decision: unified filter + multiplier ─────────────────────────────── */
function getBetDecision(f) {
  if (f.decision_source === 'golden_elo_reopen') return { visible: true, multiplier: null };
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
    favConfEl.value        = '';
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

function normalizeFightName(name) {
  return String(name || '')
    .toLowerCase()
    .replace(/['.`]/g, '')
    .replace(/-/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function didTrackedBetWin(f) {
  const bet = f.bet_placed;
  if (!bet || !f.winner) return null;
  const winner = normalizeFightName(f.winner);
  const fighter = normalizeFightName(bet.fighter);
  if (!winner || !fighter) return null;
  return winner.includes(fighter) || fighter.includes(winner);
}

function getTrackedBetSummary(f) {
  const bet = f.bet_placed;
  if (!bet) return null;
  const stake = Number(bet.stake);
  const odds = Number(bet.bet_odds);
  if (!Number.isFinite(stake) || stake <= 0 || !Number.isFinite(odds) || odds === 0) return null;
  const won = didTrackedBetWin(f);
  const pnl = won === null
    ? null
    : (won
      ? +(stake * (odds > 0 ? odds / 100 : 100 / Math.abs(odds))).toFixed(1)
      : -stake);
  return {
    fighter: bet.fighter,
    stake,
    odds,
    listedOdds: bet.listed_odds,
    opponentListedOdds: bet.opponent_listed_odds,
    placedAt: bet.placed_at,
    settled: won !== null,
    won,
    pnl,
    risk: stake,
  };
}

function summarizeFights(fights) {
  const visible = fights.filter(passesFilter);
  const tracked = visible
    .map(f => ({ fight: f, bet: getTrackedBetSummary(f) }))
    .filter(item => item.bet);
  const useTracked = tracked.length > 0;
  const settled = [];

  if (useTracked) {
    for (const item of tracked) {
      if (item.bet.settled && item.bet.pnl !== null) {
        settled.push({ won: item.bet.won, pnl: item.bet.pnl, risk: item.bet.risk });
      }
    }
  } else {
    for (const f of visible) {
      if (f.correct !== null) {
        settled.push({ won: f.correct, pnl: f.pnl || 0, risk: 100 });
      }
    }
  }

  const totalPnl = settled.reduce((sum, item) => sum + item.pnl, 0);
  const totalRisk = settled.reduce((sum, item) => sum + item.risk, 0);
  const wins = settled.filter(item => item.won).length;
  const settledCount = settled.length;

  return {
    mode: useTracked ? 'tracked' : 'model',
    visibleCount: visible.length,
    settledCount,
    wins,
    totalPnl,
    totalRisk,
    accuracy: settledCount ? +((wins / settledCount) * 100).toFixed(1) : null,
    roi: totalRisk ? +((totalPnl / totalRisk) * 100).toFixed(1) : null,
  };
}

async function reloadEventsData() {
  const evRes = await fetch('/api/events');
  if (!evRes.ok) throw new Error(`Server ${evRes.status}: ${await evRes.text()}`);
  allEvents = await evRes.json();
  if (!allEvents.length) return;
  activeIndex = Math.min(activeIndex, allEvents.length - 1);
  renderOverall();
  renderTabs();
  selectEvent(Math.max(activeIndex, 0));
}

/* ── Overall summary strip ──────────────────────────────────────────────────── */
function renderOverall() {
  const summary = summarizeFights(allEvents.flatMap(ev => ev.fights));
  const acc = summary.accuracy !== null ? summary.accuracy.toFixed(1) : '—';
  const roi = summary.roi !== null ? summary.roi.toFixed(1) : '—';
  const totalPnl = summary.totalPnl;
  const roiLabel = summary.mode === 'tracked' ? 'ROI (tracked bets)' : 'ROI (flat $100)';
  const totalLabel = summary.mode === 'tracked' ? 'Tracked P&amp;L' : 'Total P&amp;L';

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
      <span class="label">${summary.mode === 'tracked' ? 'Bets settled' : 'Fights tracked'}</span>
      <span class="value">${summary.settledCount}</span>
    </div>
    <div class="overall-stat">
      <span class="label">${summary.mode === 'tracked' ? 'Bet accuracy' : 'Model accuracy'}</span>
      <span class="value ${roiClass}">${acc}%</span>
    </div>
    <div class="overall-divider"></div>
    <div class="overall-stat">
      <span class="label">${roiLabel}</span>
      <span class="value ${roiClass}">${roi !== '—' ? roi + '%' : '—'}</span>
    </div>
    <div class="overall-stat">
      <span class="label">${totalLabel}</span>
      <span class="value ${pnlClass}">${pnlFmt}</span>
    </div>
  `;

  if (headerStats) {
    headerStats.textContent = `${summary.settledCount} ${summary.mode === 'tracked' ? 'bets' : 'fights'} · ${acc}% acc`;
  }
}

/* ── Event tabs ──────────────────────────────────────────────────────────────── */
/* ── Event date parsing + recency ───────────────────────────────────────────── */
const RECENT_WINDOW_MONTHS = 1;   // events within ~1 month stay as tabs
const MIN_VISIBLE_TABS      = 3;  // never collapse the newest few into the dropdown

const _MONTHS = { january:0, february:1, march:2, april:3, may:4, june:5,
  july:6, august:7, september:8, october:9, november:10, december:11,
  jan:0, feb:1, mar:2, apr:3, jun:5, jul:6, aug:7, sep:8, sept:8, oct:9, nov:10, dec:11 };

function parseEventDate(raw) {
  if (!raw) return null;
  const s = String(raw).trim();
  let m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);       // ISO: 2026-07-11
  if (m) return new Date(+m[1], +m[2] - 1, +m[3]);
  const cleaned = s.replace(/(\d+)(st|nd|rd|th)/i, '$1');
  m = cleaned.match(/^([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?$/);  // "May 2nd" / "February 15, 2025"
  if (m) {
    const mo = _MONTHS[m[1].toLowerCase()];
    if (mo === undefined) return null;
    const year = m[3] ? +m[3] : _inferYear(mo, +m[2]);
    return new Date(year, mo, +m[2]);
  }
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

// Yearless dates: pick the year that places the date nearest today without being
// more than ~1 month in the future (cards a long way "ahead" are last year's).
function _inferYear(mo, day) {
  const now = new Date();
  const cand = new Date(now.getFullYear(), mo, day);
  const oneMonthAhead = new Date(now.getFullYear(), now.getMonth() + 1, now.getDate());
  return cand > oneMonthAhead ? now.getFullYear() - 1 : now.getFullYear();
}

function eventDate(ev) { return parseEventDate(ev && ev.event_date); }

function isRecentEvent(ev) {
  const d = eventDate(ev);
  if (!d) return true;   // unknown date → keep visible rather than hide
  const cutoff = new Date();
  cutoff.setMonth(cutoff.getMonth() - RECENT_WINDOW_MONTHS);
  cutoff.setHours(0, 0, 0, 0);
  return d >= cutoff;
}

// Index of the newest event (largest date) — used as the default active tab.
function defaultEventIndex() {
  let best = -1, bestT = -Infinity;
  allEvents.forEach((ev, i) => {
    const d = eventDate(ev);
    const t = d ? d.getTime() : -Infinity;
    if (t >= bestT) { bestT = t; best = i; }
  });
  return best >= 0 ? best : Math.max(0, allEvents.length - 1);
}

function _byNewest(a, b) {
  const da = eventDate(allEvents[a]), db = eventDate(allEvents[b]);
  return (db ? db.getTime() : 0) - (da ? da.getTime() : 0);
}

/* ── Tabs (recent) + dropdown (older than 1 month) ──────────────────────────── */
function eventTabHtml(ev, i) {
  const roi = filteredRoi(ev);
  const cls = roi !== null ? (roi > 0 ? 'pos' : roi < 0 ? 'neg' : '') : '';
  const roiStr = roi !== null ? (roi >= 0 ? `+${roi}%` : `${roi}%`) : '?';
  const label = shortEventName(ev.event_name, ev.event_date);
  const userCls = ev.source_type === 'user_added' ? ' user-added' : '';
  return `<button class="event-tab${userCls}" onclick="selectEvent(${i})" id="etab-${i}"
      data-event-name="${escAttr(ev.event_name || '')}"
      data-event-date="${escAttr(ev.event_date || '')}"
      title="${escAttr(ev.event_name || '')}">
      ${label}${ev.source_type === 'user_added' ? '<span class="ua-dot" title="User-added event">●</span>' : ''}
      <span class="tab-roi ${cls}">${roiStr}</span>
    </button>`;
}

function renderTabs() {
  const recent = [], older = [];
  allEvents.forEach((ev, i) => (isRecentEvent(ev) ? recent : older).push(i));

  // Guard: keep at least the newest few as tabs so the bar is never empty
  // (e.g. when every event is already older than a month).
  if (recent.length < MIN_VISIBLE_TABS && older.length) {
    const byNewest = older.slice().sort(_byNewest);
    while (recent.length < MIN_VISIBLE_TABS && byNewest.length) {
      const idx = byNewest.shift();
      recent.push(idx);
      older.splice(older.indexOf(idx), 1);
    }
    recent.sort((a, b) => a - b);
  }

  const tabsHtml = recent.map(i => eventTabHtml(allEvents[i], i)).join('');

  let dropdownHtml = '';
  if (older.length) {
    const opts = older.slice().sort(_byNewest).map(i => {
      const ev = allEvents[i];
      const roi = filteredRoi(ev);
      const roiStr = roi !== null ? (roi >= 0 ? ` (+${roi}%)` : ` (${roi}%)`) : '';
      return `<option value="${i}">${escHtml(shortEventName(ev.event_name, ev.event_date) + roiStr)}</option>`;
    }).join('');
    dropdownHtml = `<select class="event-older-select" id="olderEventSelect"
        onchange="if(this.value!=='')selectEvent(parseInt(this.value,10))"
        title="Events older than one month">
        <option value="">Older events (${older.length}) ▾</option>
        ${opts}
      </select>`;
  }

  tabsEl.innerHTML = tabsHtml + dropdownHtml;
  syncActiveTab(activeIndex);
}

// Highlight the active tab by id and reflect an older selection in the dropdown.
function syncActiveTab(idx) {
  document.querySelectorAll('.event-tab').forEach(btn => {
    btn.classList.toggle('active', btn.id === `etab-${idx}`);
  });
  const sel = document.getElementById('olderEventSelect');
  if (sel) sel.value = document.getElementById(`etab-${idx}`) ? '' : String(idx);
}

function filteredRoi(ev) {
  return summarizeFights(ev.fights).roi;
}

function shortEventName(name, eventDate) {
  // "UFC 324: Gaethje vs. Pimblett" → "UFC 324"
  if (!name) return 'Event';
  const dateLabel = shortEventDate(eventDate);
  if (name.startsWith('UFC Fight Night:')) {
    const matchup = name.split(':', 2)[1]?.trim() || 'Fight Night';
    return `FN: ${truncateLabel(matchup, 22)}${dateLabel ? ` · ${dateLabel}` : ''}`;
  }
  if (name.startsWith('MMA Card') && dateLabel) {
    return `MMA Card · ${dateLabel}`;
  }
  const colon = name.indexOf(':');
  if (colon !== -1) return name.slice(0, colon).trim();
  return truncateLabel(name, 18);
}

function shortEventDate(dateStr) {
  if (!dateStr) return '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
    return dateStr.slice(5);
  }

  const cleaned = String(dateStr).replace(/(\d+)(st|nd|rd|th)\b/i, '$1').trim();
  const match = cleaned.match(/^([A-Za-z]+)\s+(\d{1,2})$/);
  if (match) {
    return `${match[1].slice(0, 3)} ${match[2]}`;
  }
  return truncateLabel(cleaned, 10);
}

function truncateLabel(text, maxLen) {
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen) + '…' : text;
}

/* ── Select + render event panel ────────────────────────────────────────────── */
function selectEvent(idx) {
  activeIndex = idx;

  // Highlight active tab (id-based, since older events live in the dropdown)
  syncActiveTab(idx);

  const ev = allEvents[idx];
  if (!ev) return;
  const decisionCard = decisionCards.get(ev.event_date);

  // Recompute stats for filtered fights only
  const filtered    = ev.fights.filter(passesFilter);
  const summary     = summarizeFights(ev.fights);
  const totalPnl    = summary.totalPnl;
  const accuracy    = summary.accuracy !== null ? summary.accuracy.toFixed(1) : null;
  const roi         = summary.roi !== null ? summary.roi.toFixed(1) : null;
  const roiLabel    = summary.mode === 'tracked' ? 'ROI (tracked)' : 'ROI';
  const pnlLabel    = summary.mode === 'tracked' ? 'Bet P&amp;L' : 'P&amp;L ($100 flat)';
  const accLabel    = summary.mode === 'tracked' ? 'Bet Accuracy' : 'Accuracy';

  const accFmt    = accuracy !== null ? `${accuracy}%`  : '—';
  const roiFmt    = roi      !== null ? (roi >= 0 ? `+${roi}%` : `${roi}%`) : '—';
  const pnlFmt    = totalPnl !== 0    ? (totalPnl >= 0 ? `+$${totalPnl.toFixed(0)}` : `-$${Math.abs(totalPnl).toFixed(0)}`) : '$0';
  const roiClass  = roi  !== null ? (roi  > 0 ? 'pos' : roi  < 0 ? 'neg' : '') : '';
  const pnlClass  = totalPnl > 0 ? 'pos' : totalPnl < 0 ? 'neg' : '';

  const fightCards = ev.fights.map(f => {
    const decision = getBetDecision(f);
    return renderFightCard(
      f,
      ev.event_date,
      decision.visible,
      decision.multiplier,
      findDecisionFight(decisionCard, f),
    );
  }).join('');

  const decisionControls = renderDecisionControls(decisionCard);

  panelEl.innerHTML = `
    <div class="event-panel">
      <div class="event-panel-header">
        <div>
          <div class="event-panel-title">${ev.event_name || 'UFC Event'}</div>
          <div class="event-panel-date">${ev.event_date}${ev.event_url ? ` · <a href="${ev.event_url}" target="_blank" style="color:var(--text-secondary);text-decoration:none;">odds source ↗</a>` : ''}</div>
          ${decisionControls}
        </div>
        <div class="event-stats-row">
          <div class="ev-stat">
            <span class="ev-stat-val">${filtered.length}<span style="font-size:12px;font-weight:400;color:var(--text-secondary)">/${ev.fights.length}</span></span>
            <span class="ev-stat-lbl">Fights</span>
          </div>
          <div class="ev-stat">
            <span class="ev-stat-val ${roiClass}">${accFmt}</span>
            <span class="ev-stat-lbl">${accLabel}</span>
          </div>
          <div class="ev-stat">
            <span class="ev-stat-val ${roiClass}">${roiFmt}</span>
            <span class="ev-stat-lbl">${roiLabel}</span>
          </div>
          <div class="ev-stat">
            <span class="ev-stat-val ${pnlClass}">${pnlFmt}</span>
            <span class="ev-stat-lbl">${pnlLabel}</span>
          </div>
        </div>
      </div>
      <div class="fights-list">
        ${fightCards}
      </div>
    </div>
  `;

  if (!decisionCards.has(ev.event_date)) loadCachedDecisionCard(ev, idx);
}

/* ── Fight card ──────────────────────────────────────────────────────────────── */
function renderFightCard(f, eventDate, visible = true, multiplier = null, decisionFight = null) {
  const hasPred   = f.model_prob_f1 !== null;
  const hasResult = f.winner !== null;
  const isWin     = f.correct === true;
  const isLoss    = f.correct === false;
  const isPending = !hasResult;
  const trackedBet = getTrackedBetSummary(f);

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
  const isOddsHistoryAvailable = f.source_type === 'the_odds_api' && !!eventDate;
  const f1OddsDisplay = isOddsHistoryAvailable
    ? `<button type="button" class="fighter-odds odds-history-trigger"
         data-fighter="${escAttr(f.fighter1)}"
         data-opponent="${escAttr(f.fighter2)}"
         data-event-date="${escAttr(eventDate)}">${f1OddsFmt} · mkt ${f.market_prob_f1}%</button>`
    : `<span class="fighter-odds">${f1OddsFmt} · mkt ${f.market_prob_f1}%</span>`;
  const f2OddsDisplay = isOddsHistoryAvailable
    ? `<button type="button" class="fighter-odds odds-history-trigger"
         data-fighter="${escAttr(f.fighter2)}"
         data-opponent="${escAttr(f.fighter1)}"
         data-event-date="${escAttr(eventDate)}">${f2OddsFmt} · mkt ${(100 - f.market_prob_f1).toFixed(1)}%</button>`
    : `<span class="fighter-odds">${f2OddsFmt} · mkt ${(100 - f.market_prob_f1).toFixed(1)}%</span>`;

  // Per-fighter ELO chips (higher rating highlighted)
  const hasElo = f.f1_elo !== null && f.f1_elo !== undefined && f.f2_elo !== null && f.f2_elo !== undefined;
  const f1EloHi = hasElo && f.f1_elo > f.f2_elo ? ' elo-higher' : '';
  const f2EloHi = hasElo && f.f2_elo > f.f1_elo ? ' elo-higher' : '';
  const f1EloDisplay = (f.f1_elo !== null && f.f1_elo !== undefined)
    ? `<span class="fighter-elo${f1EloHi}">ELO ${f.f1_elo}</span>` : '';
  const f2EloDisplay = (f.f2_elo !== null && f.f2_elo !== undefined)
    ? `<span class="fighter-elo${f2EloHi}">ELO ${f.f2_elo}</span>` : '';

  // Probability bar width: model if available, else market
  const barWidth = hasPred ? f.model_prob_f1 : (f.market_prob_f1 || 50);
  const mktLabel = hasPred ? `${f.model_prob_f1}%` : '?';
  const mktLabelR = hasPred ? `${(100 - f.model_prob_f1).toFixed(1)}%` : '?';

  const edgeFmt   = f.edge !== null ? (f.edge > 0 ? `+${f.edge}%` : `${f.edge}%`) : null;
  const edgeClass = f.edge !== null ? (f.edge > 0 ? 'pos' : 'neg') : '';
  const f1BetBadge = trackedBet && trackedBet.fighter === f.fighter1 ? '<span class="fighter-bet-badge">BET</span>' : '';
  const f2BetBadge = trackedBet && trackedBet.fighter === f.fighter2 ? '<span class="fighter-bet-badge">BET</span>' : '';

  // Outcome meta
  let outcomeMeta = '';
  if (hasResult) {
    const oClass = isWin ? 'correct' : 'wrong';
    const mthd   = [f.winner, f.method, f.round ? `R${f.round}` : ''].filter(Boolean).join(' · ');
    outcomeMeta = `<span class="fight-meta-item">Result: <span class="meta-outcome ${oClass} meta-val">${mthd}</span></span>`;
  } else if (hasPred) {
    outcomeMeta = `<span class="fight-meta-item" style="color:var(--text-secondary)">Result: <span class="meta-val">TBD</span></span>`;
  }

  const displayedPnl = trackedBet && trackedBet.pnl !== null ? trackedBet.pnl : f.pnl;
  const pnlMeta = displayedPnl !== null
    ? `<span class="fight-meta-item">${trackedBet ? 'Bet P&amp;L' : 'P&amp;L'}: <span class="meta-val ${displayedPnl >= 0 ? 'pos' : 'neg'}" style="color:${displayedPnl >= 0 ? 'var(--accent)' : 'var(--danger)'}">${displayedPnl >= 0 ? '+' : ''}$${displayedPnl}</span></span>`
    : '';
  const betMeta = trackedBet
    ? `<span class="fight-meta-item">Bet: <span class="meta-val">$${trackedBet.stake.toFixed(2)} @ ${fmtOdds(trackedBet.odds)}${trackedBet.listedOdds !== null && trackedBet.listedOdds !== undefined && trackedBet.listedOdds !== trackedBet.odds ? ` <span class="meta-subtle">(listed ${fmtOdds(trackedBet.listedOdds)})</span>` : ''}${trackedBet.opponentListedOdds !== null && trackedBet.opponentListedOdds !== undefined ? ` <span class="meta-subtle">vs ${fmtOdds(trackedBet.opponentListedOdds)}</span>` : ''}</span></span>`
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
  const eloMeta = f.pick_elo_diff !== null && f.pick_elo_diff !== undefined
    ? `<span class="fight-meta-item">ELO Diff: <span class="meta-val ${f.pick_elo_diff > 0 ? 'pos' : f.pick_elo_diff < 0 ? 'neg' : ''}">${f.pick_elo_diff > 0 ? '+' : ''}${f.pick_elo_diff}</span></span>`
    : '';
  const reviewMeta = f.review_label
    ? `<span class="fight-meta-item">Signal: <span class="meta-val">${escHtml(f.review_label)}</span></span>`
    : '';

  const errorNote = f.error
    ? `<div class="fight-error">⚠ ${f.error}</div>`
    : '';

  const noBetBadge = visible ? '' : `<span class="no-bet-badge">no bet</span>`;
  const goldenEloBadge = f.review_tier ? `<span class="bet-size-badge bet-size-low">Golden ELO T${f.review_tier}</span>` : '';
  const decisionBadge = renderDecisionBadge(decisionFight);

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
            <span class="${f1Class} fighter-clickable" data-fighter="${f.fighter1}">${f.fighter1}</span>${f1BetBadge}
            ${f1OddsDisplay}
            ${f1EloDisplay}
          </div>
          <span class="vs-divider">VS</span>
          <div class="fighter-block f2">
            <span class="${f2Class} fighter-clickable" data-fighter="${f.fighter2}">${f.fighter2}</span>${f2BetBadge}
            ${f2OddsDisplay}
            ${f2EloDisplay}
          </div>
        </div>
        ${noBetBadge}
        ${goldenEloBadge}
        ${betSizeBadge}
        ${decisionBadge}
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
        ${betMeta}
        ${edgeMeta}
        ${eloMeta}
        ${reviewMeta}
        ${srcMeta}
        ${fightsMeta}
        ${renderDecisionMeta(decisionFight)}
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

function decisionFightKey(fighter1, fighter2) {
  return [normalizeFightName(fighter1), normalizeFightName(fighter2)].sort().join('::');
}

function findDecisionFight(card, fight) {
  if (!card || card.status !== 'complete') return null;
  const key = decisionFightKey(fight.fighter1, fight.fighter2);
  return (card.fights || []).find(item =>
    decisionFightKey(item.fighter1, item.fighter2) === key
  ) || null;
}

function renderDecisionControls(card) {
  if (!card) {
    return `<div class="decision-card-controls">
      <button class="decision-card-btn" onclick="analyzeDecisionCard(event)">Analyze Finish / Decision</button>
      <span class="decision-card-note">Runs once for the full card and caches results.</span>
    </div>`;
  }
  if (card.status === 'queued' || card.status === 'running') {
    return `<div class="decision-card-controls">
      <button class="decision-card-btn" disabled><span class="decision-spinner"></span> Analyzing card…</button>
      <span class="decision-card-note">Winner predictions remain available while this runs.</span>
    </div>`;
  }
  if (card.status === 'error') {
    return `<div class="decision-card-controls">
      <button class="decision-card-btn" onclick="analyzeDecisionCard(event, true)">Retry Finish / Decision</button>
      <span class="decision-card-note decision-card-error">${escHtml(card.error_message || 'Card analysis failed.')}</span>
    </div>`;
  }
  const eligible = (card.fights || []).filter(item => item.result?.eligible).length;
  return `<div class="decision-card-controls">
    <button class="decision-card-btn decision-card-btn-cached" onclick="analyzeDecisionCard(event, true)">Refresh Finish / Decision</button>
    <span class="decision-card-note">${eligible} threshold signal${eligible === 1 ? '' : 's'} · cached ${formatDecisionTimestamp(card.completed_at)}</span>
  </div>`;
}

function renderDecisionBadge(item) {
  const result = item?.result;
  if (!result || result.bet === 'error') return '';
  const pct = (result.confidence * 100).toFixed(1);
  const selection = result.selection === 'finish' ? 'Finish' : 'Decision';
  const opportunity = result.confidence >= 0.60 && result.eligible;
  const inactiveStatus = result.confidence >= 0.60
    ? 'History Ineligible'
    : 'Below 60% Bar';
  return `
    <div class="decision-signal-block ${opportunity ? 'decision-signal-opportunity' : 'decision-signal-ineligible'}"
         title="${opportunity ? 'Confidence threshold cleared; check market price before wagering.' : 'Below the 60% confidence/history eligibility threshold.'}">
      <span class="decision-signal-label">${selection}</span>
      <strong class="decision-signal-confidence">${pct}%</strong>
      <span class="decision-signal-status">${opportunity ? 'Bet Opportunity' : inactiveStatus}</span>
    </div>`;
}

function renderDecisionMeta(item) {
  if (!item) return '';
  const result = item.result;
  if (!result) return '';
  if (result.bet === 'error') {
    return `<span class="fight-meta-item">Finish/Decision: <span class="meta-val decision-card-error">${escHtml(result.error_code || 'error')}</span></span>`;
  }
  const finish = (result.probabilities.finish * 100).toFixed(1);
  const decision = (result.probabilities.decision * 100).toFixed(1);
  return `<span class="fight-meta-item">Finish/Decision: <span class="meta-val">${finish}% / ${decision}%</span></span>
    <span class="fight-meta-item">Division: <span class="meta-val">${escHtml(item.weight_class || '—')} · #${item.fight_number || '—'}</span></span>`;
}

function formatDecisionTimestamp(value) {
  if (!value) return 'previously';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? 'previously' : parsed.toLocaleString();
}

async function loadCachedDecisionCard(ev, idx) {
  decisionCards.set(ev.event_date, null);
  try {
    const res = await fetch(`/api/decision-cards?event_date=${encodeURIComponent(ev.event_date)}`);
    if (res.status === 404) return;
    if (!res.ok) throw new Error(`${res.status}`);
    const card = await res.json();
    const current = decisionCards.get(ev.event_date);
    if (current?.status === 'queued' || current?.status === 'running') return;
    decisionCards.set(ev.event_date, card);
    if (card.status === 'queued' || card.status === 'running') pollDecisionCard(ev.event_date, card.card_key);
    if (activeIndex === idx) selectEvent(idx);
  } catch (err) {
    console.error('decision-card cache lookup failed:', err);
  }
}

async function analyzeDecisionCard(event, force = false) {
  event?.stopPropagation();
  const ev = allEvents[activeIndex];
  if (!ev) return;

  decisionCards.set(ev.event_date, {
    status: 'queued',
    event_date: ev.event_date,
    fights: [],
  });
  selectEvent(activeIndex);

  try {
    const res = await fetch('/api/decision-cards/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_name: ev.event_name,
        event_date: ev.event_date,
        fights: ev.fights.map(f => ({ fighter1: f.fighter1, fighter2: f.fighter2 })),
        force,
      }),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    const card = await res.json();
    decisionCards.set(ev.event_date, card);
    selectEvent(activeIndex);
    if (card.status === 'queued' || card.status === 'running') {
      pollDecisionCard(ev.event_date, card.card_key);
    }
  } catch (err) {
    decisionCards.set(ev.event_date, {
      status: 'error',
      event_date: ev.event_date,
      error_message: err.message,
      fights: [],
    });
    selectEvent(activeIndex);
  }
}

function pollDecisionCard(eventDate, cardKey) {
  if (!cardKey || decisionPolls.has(cardKey)) return;
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/decision-cards/${encodeURIComponent(cardKey)}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const card = await res.json();
      decisionCards.set(eventDate, card);
      const ev = allEvents[activeIndex];
      if (ev?.event_date === eventDate) selectEvent(activeIndex);
      if (card.status === 'complete' || card.status === 'error') {
        clearInterval(timer);
        decisionPolls.delete(cardKey);
      }
    } catch (err) {
      clearInterval(timer);
      decisionPolls.delete(cardKey);
      console.error('decision-card polling failed:', err);
    }
  }, 2500);
  decisionPolls.set(cardKey, timer);
}

/* ── Helpers ─────────────────────────────────────────────────────────────────── */
function fmtOdds(n) {
  if (n === null || n === undefined) return '—';
  return n > 0 ? `+${n}` : `${n}`;
}

/* ── Add Event modal ────────────────────────────────────────────────────────── */
let userEventsList = [];

function wireAddEvent() {
  const fab      = document.getElementById('addEventFab');
  const overlay  = document.getElementById('addEventOverlay');
  const closeBtn = document.getElementById('addEventClose');
  const cancelBtn= document.getElementById('addEventCancel');
  const submitBtn= document.getElementById('addEventSubmit');
  const bfoInput = document.getElementById('bfoUrl');
  const statsInput = document.getElementById('ufcStatsUrl');
  const errEl    = document.getElementById('addEventError');
  const modalTitle = document.getElementById('modalTitle');
  const eventSelectorWrap = document.getElementById('eventSelectorWrap');
  const eventSelector = document.getElementById('eventSelector');
  const bfoReq = document.getElementById('bfoReq');
  const ufcOptional = document.getElementById('ufcOptional');

  function openModal()  {
    overlay.classList.remove('hidden');
    bfoInput.focus();
    updateModalMode();
  }
  function closeModal() {
    overlay.classList.add('hidden');
    errEl.classList.add('hidden');
    errEl.textContent = '';
    bfoInput.value = '';
    statsInput.value = '';
    updateModalMode();
  }

  // Dynamically update modal based on inputs
  function updateModalMode() {
    const bfoVal = bfoInput.value.trim();
    const statsVal = statsInput.value.trim();

    if (!bfoVal && statsVal) {
      // Only UFC Stats provided → Add Results mode
      modalTitle.textContent = 'Add Results';
      submitBtn.textContent = 'Add Results';
      eventSelectorWrap.classList.remove('hidden');
      bfoReq.classList.add('hidden');
      ufcOptional.classList.add('hidden');
      loadUserEvents();
    } else {
      // Normal mode: Add Event
      modalTitle.textContent = 'Add Event';
      submitBtn.textContent = 'Scrape & Add';
      eventSelectorWrap.classList.add('hidden');
      bfoReq.classList.remove('hidden');
      ufcOptional.classList.remove('hidden');
    }
  }

  async function loadUserEvents() {
    if (userEventsList.length === 0) {
      try {
        const res = await fetch('/api/user-events');
        if (res.ok) userEventsList = await res.json();
      } catch (e) { console.error('Failed to load user events', e); }
    }
    // Populate dropdown
    eventSelector.innerHTML = '<option value="">— Choose an event —</option>' +
      userEventsList.map(e => `<option value="${e.slug}">${e.event_name} (${e.event_date})</option>`).join('');
  }

  bfoInput.addEventListener('input', updateModalMode);
  statsInput.addEventListener('input', updateModalMode);

  fab.addEventListener('click', openModal);
  closeBtn.addEventListener('click', closeModal);
  cancelBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });

  submitBtn.addEventListener('click', async () => {
    const bfoUrl   = bfoInput.value.trim();
    const statsUrl = statsInput.value.trim();
    const isResultsMode = !bfoUrl && statsUrl;

    if (isResultsMode) {
      // Add Results mode
      const slug = eventSelector.value;
      if (!slug) {
        showErr('Please select an event to update.');
        return;
      }
      if (!statsUrl) {
        showErr('UFCStats URL is required.');
        return;
      }

      submitBtn.disabled = true;
      submitBtn.textContent = 'Updating…';
      errEl.classList.add('hidden');

      try {
        const res = await fetch(`/api/user-events/${encodeURIComponent(slug)}/results`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ufc_stats_url: statsUrl }),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || res.statusText);
        }

        closeModal();
        await reloadEvents();
        // Jump to the updated event
        const idx = allEvents.findIndex(e => e.event_url && e.event_url.includes(slug));
        if (idx >= 0) selectEvent(idx);

      } catch (err) {
        showErr(err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Add Results';
      }
    } else {
      // Scrape & Add mode (existing behavior)
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
function escHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escAttr(s) {
  return s === null || s === undefined ? '' : String(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/* ── Odds history modal ─────────────────────────────────────────────────────── */
function wireOddsHistoryModal() {
  const overlay = document.getElementById('oddsHistoryOverlay');
  const closeBtn = document.getElementById('oddsHistoryClose');
  const titleEl = document.getElementById('oddsHistoryTitle');
  const metaEl = document.getElementById('oddsHistoryMeta');
  const bodyEl = document.getElementById('oddsHistoryBody');
  if (!overlay || !closeBtn || !titleEl || !metaEl || !bodyEl) return;

  const closeModal = () => overlay.classList.add('hidden');

  closeBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !overlay.classList.contains('hidden')) closeModal();
  });

  const renderModalBody = (payload, selectedFighter) => {
    const fighterIsF1 = selectedFighter === payload.fighter1;
    const selectedLabel = fighterIsF1 ? payload.fighter1 : payload.fighter2;
    const opponentLabel = fighterIsF1 ? payload.fighter2 : payload.fighter1;
    const currentBet = payload.bet_placed;
    const selectedBetActive = currentBet && currentBet.fighter === selectedFighter;
    const selectedCurrentOdds = fighterIsF1 ? payload.current_fighter1_odds : payload.current_fighter2_odds;
    const snapshotLabel = payload.real_history_count === 1 ? 'real snapshot' : 'real snapshots';
    const estimatedNote = payload.uses_estimated_samples
      ? ' · estimated backfill included'
      : '';
    const savedStake = selectedBetActive && currentBet?.stake ? currentBet.stake : '';
    const savedOdds = selectedBetActive && currentBet?.bet_odds ? currentBet.bet_odds : (selectedCurrentOdds ?? '');

    metaEl.textContent = `${payload.event_name || `${payload.fighter1} vs ${payload.fighter2}`} · ${payload.event_date} · ${payload.real_history_count} ${snapshotLabel}${estimatedNote}`;
    bodyEl.innerHTML = `
      <div class="odds-history-actions">
        <div class="odds-history-form">
          <label class="odds-history-input-group">
            <span>Fund size</span>
            <input type="number" min="0.01" step="0.01" class="modal-input odds-history-stake-input" value="${escAttr(savedStake)}" placeholder="40" />
          </label>
          <label class="odds-history-input-group">
            <span>Bet odds</span>
            <input type="number" step="1" class="modal-input odds-history-custom-odds-input" value="${escAttr(savedOdds)}" placeholder="${escAttr(selectedCurrentOdds ?? '')}" />
          </label>
        </div>
        <div class="odds-history-bet-status">
          ${currentBet
            ? `Bet placed: <strong>${escHtml(currentBet.fighter)}</strong> for <strong>$${escHtml(Number(currentBet.stake || 0).toFixed(2))}</strong> @ <strong>${fmtOdds(currentBet.bet_odds)}</strong>${currentBet.listed_odds !== null && currentBet.listed_odds !== undefined && currentBet.listed_odds !== currentBet.bet_odds ? ` <span class="meta-subtle">(listed ${fmtOdds(currentBet.listed_odds)})</span>` : ''}${currentBet.opponent_listed_odds !== null && currentBet.opponent_listed_odds !== undefined ? ` <span class="meta-subtle">· opp ${fmtOdds(currentBet.opponent_listed_odds)}</span>` : ''}${currentBet.placed_at ? ` · ${escHtml(currentBet.placed_at)}` : ''}`
            : 'Bet placed: not marked'}
        </div>
        <button
          type="button"
          class="modal-submit odds-history-bet-toggle"
          data-active="${selectedBetActive ? '1' : '0'}"
          data-fighter="${escAttr(selectedFighter)}"
          data-opponent="${escAttr(opponentLabel)}"
          data-event-date="${escAttr(payload.event_date)}">
          ${selectedBetActive ? 'Clear bet placed' : 'Save bet placed'}
        </button>
      </div>
      <div class="odds-history-inline-error hidden"></div>
      <table class="odds-history-table">
        <thead>
          <tr>
            <th>Sample</th>
            <th>Captured</th>
            <th>${escHtml(selectedLabel)}</th>
            <th>${escHtml(opponentLabel)}</th>
            <th>Book</th>
          </tr>
        </thead>
        <tbody>
          ${(payload.samples || []).map(sample => `
            <tr>
              <td class="odds-history-sample">${escHtml(sample.label)}</td>
              <td>${escHtml(sample.captured_at || '—')}</td>
              <td>${fighterIsF1 ? fmtOdds(sample.fighter1_odds) : fmtOdds(sample.fighter2_odds)}</td>
              <td>${fighterIsF1 ? fmtOdds(sample.fighter2_odds) : fmtOdds(sample.fighter1_odds)}</td>
              <td class="odds-history-book">${escHtml(sample.bookmaker || '—')}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  };

  document.addEventListener('click', async e => {
    const trigger = e.target.closest('.odds-history-trigger');
    const toggle = e.target.closest('.odds-history-bet-toggle');
    if (!trigger && !toggle) return;

    e.preventDefault();
    e.stopPropagation();

    if (toggle) {
      const fighter = toggle.dataset.fighter;
      const opponent = toggle.dataset.opponent;
      const eventDate = toggle.dataset.eventDate;
      const isActive = toggle.dataset.active === '1';
      if (!fighter || !opponent || !eventDate) return;
      const stakeInput = overlay.querySelector('.odds-history-stake-input');
      const oddsInput = overlay.querySelector('.odds-history-custom-odds-input');
      const inlineError = overlay.querySelector('.odds-history-inline-error');
      const stakeValue = stakeInput?.value.trim() || '';
      const oddsValue = oddsInput?.value.trim() || '';
      if (inlineError) {
        inlineError.classList.add('hidden');
        inlineError.textContent = '';
      }

      if (!isActive) {
        const parsedStake = Number(stakeValue);
        const parsedOdds = oddsValue === '' ? null : Number(oddsValue);
        if (!Number.isFinite(parsedStake) || parsedStake <= 0) {
          if (inlineError) {
            inlineError.textContent = 'Enter a valid fund size before saving the bet.';
            inlineError.classList.remove('hidden');
          }
          return;
        }
        if (parsedOdds !== null && (!Number.isFinite(parsedOdds) || parsedOdds === 0)) {
          if (inlineError) {
            inlineError.textContent = 'Enter valid custom odds or leave the field at the listed line.';
            inlineError.classList.remove('hidden');
          }
          return;
        }
      }

      try {
        const params = new URLSearchParams({ fighter1: fighter, fighter2: opponent, event_date: eventDate, bet_fighter: fighter });
        if (!isActive) {
          params.set('stake', String(Number(stakeValue)));
          if (oddsValue !== '') params.set('custom_odds', String(parseInt(oddsValue, 10)));
        }
        const res = await fetch(`/api/odds-history/bet-toggle?${params.toString()}`);
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);

        await reloadEventsData();
        const refreshed = await fetch(`/api/odds-history?${new URLSearchParams({ fighter1: fighter, fighter2: opponent, event_date: eventDate }).toString()}`);
        const refreshedBody = await refreshed.json().catch(() => ({}));
        if (!refreshed.ok) throw new Error(refreshedBody.detail || `HTTP ${refreshed.status}`);
        renderModalBody(refreshedBody, fighter);
      } catch (err) {
        bodyEl.innerHTML = `<div class="odds-history-empty">${escHtml(err.message || 'Unable to update bet status.')}</div>`;
      }
      return;
    }

    const fighter = trigger.dataset.fighter;
    const opponent = trigger.dataset.opponent;
    const eventDate = trigger.dataset.eventDate;
    if (!fighter || !opponent || !eventDate) return;

    overlay.classList.remove('hidden');
    titleEl.textContent = `Odds History · ${fighter}`;
    metaEl.textContent = `${fighter} vs ${opponent} · ${eventDate}`;
    bodyEl.innerHTML = '<div class="odds-history-loading">Loading history…</div>';

    try {
      const params = new URLSearchParams({ fighter1: fighter, fighter2: opponent, event_date: eventDate });
      const res = await fetch(`/api/odds-history?${params.toString()}`);
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
      renderModalBody(body, fighter);
    } catch (err) {
      bodyEl.innerHTML = `<div class="odds-history-empty">${escHtml(err.message || 'Odds history unavailable.')}</div>`;
    }
  });
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
init().then(() => {
  wireAddEvent();
  wireOddsHistoryModal();
});
