// ── Fighter Ingest & Aliases ────────────────────────────────────────────────

const $sherdogUrl   = document.getElementById('sherdogUrl');
const $requestedName= document.getElementById('requestedName');
const $previewBtn   = document.getElementById('previewBtn');
const $fLoading     = document.getElementById('fPreviewLoading');
const $fError       = document.getElementById('fError');
const $fSuccess     = document.getElementById('fSuccess');
const $fPreview     = document.getElementById('fPreview');
const $fpGrid       = document.getElementById('fpGrid');
const $writeAlias   = document.getElementById('writeAlias');
const $aliasFrom    = document.getElementById('aliasFrom');
const $aliasTo      = document.getElementById('aliasTo');
const $saveFighterBtn = document.getElementById('saveFighterBtn');

const $aliasCount   = document.getElementById('aliasCount');
const $aliasTableBody = document.getElementById('aliasTableBody');
const $newAliasFrom = document.getElementById('newAliasFrom');
const $newAliasTo   = document.getElementById('newAliasTo');
const $addAliasBtn  = document.getElementById('addAliasBtn');

let lastPreview = null;

function fEsc(s) {
  if (s === null || s === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function showF(el) { el.classList.remove('hidden'); }
function hideF(el) { el.classList.add('hidden'); }

function flashError(msg) {
  $fError.textContent = msg;
  showF($fError);
  hideF($fSuccess);
}
function flashSuccess(msg) {
  $fSuccess.textContent = msg;
  showF($fSuccess);
  hideF($fError);
}

// ── Preview ─────────────────────────────────────────────────────────────────
$previewBtn.addEventListener('click', async () => {
  const url = $sherdogUrl.value.trim();
  if (!url) return;

  $previewBtn.disabled = true;
  showF($fLoading);
  hideF($fError); hideF($fSuccess); hideF($fPreview);

  try {
    const res = await fetch('/api/ingest/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sherdog_url: url,
        requested_name: $requestedName.value.trim() || null,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Preview failed');
    lastPreview = data;
    renderPreview(data);
  } catch (err) {
    flashError(err.message);
  } finally {
    hideF($fLoading);
    $previewBtn.disabled = false;
  }
});

function renderPreview(d) {
  const rows = [
    ['Scraped name', d.scraped_name],
    ['Sherdog ID', d.fighter_id],
    ['Requested name', d.requested_name || '—'],
    ['Already in DB', d.already_in_db ? 'Yes' : 'No'],
    ['DB name', d.db_name || '—'],
    ['Requested resolves to', d.requested_name_resolves_to || '— (unresolved)'],
  ];
  let html = rows.map(([k, v]) =>
    `<div class="fp-k">${fEsc(k)}</div><div class="fp-v">${fEsc(v)}</div>`
  ).join('');

  // Fight-history / model-usability hint
  if (d.has_fight_history === false) {
    html += `<div class="fp-k">Fight history</div><div class="fp-v fp-warn">No fights in DB — not model-usable until stats are ingested (use the event ingest above).</div>`;
  } else if (d.has_fight_history === true) {
    html += `<div class="fp-k">Fight history</div><div class="fp-v fp-ok">Present ✓</div>`;
  }
  $fpGrid.innerHTML = html;

  // Prefill alias fields from suggestion
  if (d.suggested_alias) {
    $aliasFrom.value = d.suggested_alias.alias || '';
    $aliasTo.value = d.suggested_alias.canonical || '';
    $writeAlias.checked = !!d.alias_needed;
  } else {
    $aliasFrom.value = d.requested_name || '';
    $aliasTo.value = d.scraped_name || '';
    $writeAlias.checked = false;
  }

  showF($fPreview);
}

// ── Save ────────────────────────────────────────────────────────────────────
$saveFighterBtn.addEventListener('click', async () => {
  if (!lastPreview) return;
  $saveFighterBtn.disabled = true;
  hideF($fError); hideF($fSuccess);

  try {
    const res = await fetch('/api/ingest/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sherdog_url: lastPreview.sherdog_url,
        requested_name: lastPreview.requested_name,
        write_alias: $writeAlias.checked,
        alias_from: $aliasFrom.value.trim() || null,
        alias_to: $aliasTo.value.trim() || null,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Save failed');

    const f = data.fighter || {};
    let msg = `Saved ${f.db_name || f.scraped_name || 'fighter'} to DB.`;
    if (data.alias) msg += ` Alias "${data.alias.alias}" → "${data.alias.canonical}" stored.`;
    flashSuccess(msg);
    if (data.alias) loadAliases();
  } catch (err) {
    flashError(err.message);
  } finally {
    $saveFighterBtn.disabled = false;
  }
});

// ── Alias manager ───────────────────────────────────────────────────────────
async function loadAliases() {
  try {
    const res = await fetch('/api/aliases');
    const data = await res.json();
    const keys = Object.keys(data).sort((a, b) => a.localeCompare(b));
    $aliasCount.textContent = `${keys.length} aliases`;
    $aliasTableBody.innerHTML = keys.map(k => `
      <tr>
        <td>${fEsc(k)}</td>
        <td>${fEsc(data[k])}</td>
        <td style="text-align:right"><button class="alias-del" data-alias="${fEsc(k)}">Delete</button></td>
      </tr>
    `).join('');
    $aliasTableBody.querySelectorAll('.alias-del').forEach(btn => {
      btn.addEventListener('click', () => deleteAlias(btn.dataset.alias));
    });
  } catch (err) {
    flashError('Failed to load aliases: ' + err.message);
  }
}

async function deleteAlias(alias) {
  try {
    const res = await fetch('/api/aliases/' + encodeURIComponent(alias), { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Delete failed');
    loadAliases();
  } catch (err) {
    flashError(err.message);
  }
}

$addAliasBtn.addEventListener('click', async () => {
  const alias = $newAliasFrom.value.trim();
  const canonical = $newAliasTo.value.trim();
  if (!alias || !canonical) return;
  try {
    const res = await fetch('/api/aliases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alias, canonical }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Add failed');
    $newAliasFrom.value = '';
    $newAliasTo.value = '';
    loadAliases();
  } catch (err) {
    flashError(err.message);
  }
});

loadAliases();
