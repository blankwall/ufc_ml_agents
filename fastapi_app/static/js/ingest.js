// ── Ingest Page JS ──────────────────────────────────────────────────────────

const $url      = document.getElementById('ufcUrl');
const $btn      = document.getElementById('ingestBtn');
const $loading  = document.getElementById('loadingState');
const $error    = document.getElementById('errorState');
const $results  = document.getElementById('resultsContent');
const $evHeader = document.getElementById('eventHeader');
const $summary  = document.getElementById('summaryStrip');
const $tbody    = document.getElementById('fightTableBody');

$btn.addEventListener('click', async () => {
  const url = $url.value.trim();
  if (!url) return;

  $btn.disabled = true;
  $loading.classList.remove('hidden');
  $error.classList.add('hidden');
  $results.classList.add('hidden');

  try {
    const res = await fetch('/api/db/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ufc_stats_url: url,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

    renderResults(data);
  } catch (err) {
    $error.textContent = err.message;
    $error.classList.remove('hidden');
  } finally {
    $loading.classList.add('hidden');
    $btn.disabled = false;
  }
});

function renderResults(data) {
  const { summary, event, fights } = data;

  // Event header
  if (event) {
    $evHeader.innerHTML = `
      <h2>${esc(event.name || 'Unknown Event')}</h2>
      <div class="event-meta">
        <span>${esc(event.date || '—')}</span>
        <span>${esc(event.location || '—')}</span>
      </div>
    `;
  }

  // Summary strip
  $summary.innerHTML = `
    <span><strong>${summary.fighters_scraped}</strong> fighters scraped</span>
    <span class="sep">·</span>
    <span><strong>${summary.fights_upserted}</strong> fights upserted</span>
    <span class="sep">·</span>
    <span><strong>${summary.fight_stats_upserted}</strong> fight stats</span>
  `;

  // Fight card table
  $tbody.innerHTML = '';
  // Sort by fight_number descending (main event first), but display ascending
  const sorted = [...(fights || [])].sort((a, b) =>
    (a.fight_number || 0) - (b.fight_number || 0)
  );

  for (const f of sorted) {
    const tr = document.createElement('tr');
    const winner = f.winner || '—';
    const f1IsWinner = f.result === 'fighter_1';
    const f2IsWinner = f.result === 'fighter_2';

    tr.innerHTML = `
      <td>${f.fight_number || '—'}</td>
      <td class="${f1IsWinner ? 'cell-win' : f2IsWinner ? 'cell-loss' : ''}">${esc(f.fighter1)}</td>
      <td class="${f2IsWinner ? 'cell-win' : f1IsWinner ? 'cell-loss' : ''}">${esc(f.fighter2)}</td>
      <td>${esc(winner)}</td>
      <td>${esc(f.method || '—')}</td>
      <td>${f.round_finished || '—'}</td>
      <td>${f.time || '—'}</td>
      <td style="color:var(--text-muted)">${esc(weightClass(f))}</td>
    `;
    $tbody.appendChild(tr);
  }

  $results.classList.remove('hidden');
}

function weightClass(f) {
  // Not in the fight rows from get_event_results, return empty
  return '';
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}
