// ── Fighter Page JS ─────────────────────────────────────────────────────────

const $search    = document.getElementById('fighterSearch');
const $dropdown  = document.getElementById('searchDropdown');
const $loading   = document.getElementById('loadingState');
const $error     = document.getElementById('errorState');
const $profile   = document.getElementById('profileContent');
const $header    = document.getElementById('profileHeader');
const $physGrid  = document.getElementById('physGrid');
const $statsGrid = document.getElementById('statsGrid');
const $history   = document.getElementById('historyTableBody');

let searchTimer = null;

// ── Autocomplete search ─────────────────────────────────────────────────────

$search.addEventListener('input', () => {
  clearTimeout(searchTimer);
  const q = $search.value.trim();
  if (q.length < 2) {
    $dropdown.classList.add('hidden');
    return;
  }
  searchTimer = setTimeout(() => doSearch(q), 250);
});

$search.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    $dropdown.classList.add('hidden');
  }
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrapper')) {
    $dropdown.classList.add('hidden');
  }
});

async function doSearch(q) {
  try {
    const res = await fetch(`/api/db/fighters/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    renderDropdown(data);
  } catch {
    $dropdown.classList.add('hidden');
  }
}

function renderDropdown(items) {
  if (!items.length) {
    $dropdown.classList.add('hidden');
    return;
  }
  $dropdown.innerHTML = items.map(f => `
    <div class="search-item" data-name="${escAttr(f.name)}">
      <span>${esc(f.name)}</span>
      <span class="search-record">${esc(f.record)}</span>
    </div>
  `).join('');
  $dropdown.classList.remove('hidden');

  $dropdown.querySelectorAll('.search-item').forEach(el => {
    el.addEventListener('click', () => {
      const name = el.dataset.name;
      $search.value = name;
      $dropdown.classList.add('hidden');
      loadFighter(name);
    });
  });
}

// ── Load fighter profile ────────────────────────────────────────────────────

async function loadFighter(name) {
  $loading.classList.remove('hidden');
  $error.classList.add('hidden');
  $profile.classList.add('hidden');

  try {
    const res = await fetch(`/api/db/fighter/${encodeURIComponent(name)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Fighter not found');
    renderProfile(data);
  } catch (err) {
    $error.textContent = err.message;
    $error.classList.remove('hidden');
  } finally {
    $loading.classList.add('hidden');
  }
}

// ── Render profile ──────────────────────────────────────────────────────────

function renderProfile(f) {
  // Header
  const nicknameHtml = f.nickname ? `<div class="nickname">"${esc(f.nickname)}"</div>` : '';
  $header.innerHTML = `
    <h2>${esc(f.name)}</h2>
    ${nicknameHtml}
    <div class="record-badge">${esc(f.record)}</div>
    <div class="header-details">
      ${f.age ? `<span>Age: ${f.age}</span>` : ''}
      ${f.stance ? `<span>Stance: ${esc(f.stance)}</span>` : ''}
      <span>Fights: ${f.fight_count}</span>
    </div>
  `;

  // Physical attributes
  $physGrid.innerHTML = '';
  const attrs = [
    { label: 'Height', value: f.height_cm ? `${f.height_cm} cm` : '—' },
    { label: 'Weight', value: f.weight_lbs ? `${f.weight_lbs} lbs` : '—' },
    { label: 'Reach', value: f.reach_inches ? `${f.reach_inches}"` : '—' },
  ];
  for (const a of attrs) {
    $physGrid.innerHTML += `
      <div class="phys-card">
        <div class="phys-label">${a.label}</div>
        <div class="phys-value">${esc(a.value)}</div>
      </div>
    `;
  }

  // Career stats
  const stats = [
    { key: 'sig_strikes_landed_per_min', label: 'SLpM', max: 10 },
    { key: 'striking_accuracy',          label: 'Str Acc', max: 100, pct: true },
    { key: 'sig_strikes_absorbed_per_min', label: 'SApM', max: 10 },
    { key: 'striking_defense',           label: 'Str Def', max: 100, pct: true },
    { key: 'takedown_avg_per_15min',     label: 'TD Avg', max: 8 },
    { key: 'takedown_accuracy',          label: 'TD Acc', max: 100, pct: true },
    { key: 'takedown_defense',           label: 'TD Def', max: 100, pct: true },
    { key: 'submission_avg_per_15min',   label: 'Sub Avg', max: 4 },
  ];

  $statsGrid.innerHTML = '';
  for (const s of stats) {
    const val = f[s.key];
    const display = val != null ? (s.pct ? `${val.toFixed(1)}%` : val.toFixed(2)) : '—';
    const pct = val != null ? Math.min((val / s.max) * 100, 100) : 0;
    $statsGrid.innerHTML += `
      <div class="stat-bar-item">
        <div class="stat-bar-label">
          <span>${s.label}</span>
          <span class="stat-bar-value">${display}</span>
        </div>
        <div class="stat-bar-track">
          <div class="stat-bar-fill" style="width:${pct}%"></div>
        </div>
      </div>
    `;
  }

  // Fight history
  $history.innerHTML = '';
  for (const h of f.fight_history) {
    const resultClass = h.result === 'W' ? 'cell-win' :
                        h.result === 'L' ? 'cell-loss' :
                        h.result === 'D' ? 'cell-draw' :
                        h.result === 'NC' ? 'cell-nc' : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${esc(h.date) || '—'}</td>
      <td style="color:var(--text-muted);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(h.event)}</td>
      <td><a class="opponent-link" data-name="${escAttr(h.opponent)}">${esc(h.opponent)}</a></td>
      <td class="${resultClass}">${h.result}</td>
      <td>${esc(h.method || '—')}</td>
      <td>${h.round || '—'}</td>
      <td>${esc(h.closing_odds || '—')}</td>
    `;
    $history.appendChild(tr);
  }

  // Wire up opponent links
  $history.querySelectorAll('.opponent-link').forEach(a => {
    a.addEventListener('click', () => {
      const name = a.dataset.name;
      $search.value = name;
      loadFighter(name);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });

  $profile.classList.remove('hidden');
}

// ── Utilities ────────────────────────────────────────────────────────────────

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function escAttr(s) {
  if (!s) return '';
  return String(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
