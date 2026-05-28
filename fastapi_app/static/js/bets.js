function formatOdds(value) {
  if (value === null || value === undefined) return '--';
  return value > 0 ? `+${value}` : `${value}`;
}

function formatDateLabel(dateStr) {
  if (!dateStr) return '--';
  const [year, month, day] = dateStr.split('-');
  if (!year || !month || !day) return dateStr;
  return `${month}-${day}`;
}

function valueClass(value) {
  if (value > 0) return 'positive';
  if (value < 0) return 'negative';
  return 'neutral';
}

function formatPercent(value) {
  if (value === null || value === undefined) return '--';
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
}

function formatCurrency(value) {
  if (value === null || value === undefined) return '--';
  return `${value > 0 ? '+' : ''}$${value.toFixed(2)}`;
}

function resultLabel(bet) {
  if (!bet.settled) return 'Pending';
  return bet.won ? 'Won' : 'Lost';
}

function metaParts(fight) {
  return [fight.bet_type, fight.source_label, fight.notes].filter(Boolean);
}

function renderOverall(cards) {
  const settled = cards.reduce((sum, card) => sum + card.settled_count, 0);
  const wins = cards.reduce((sum, card) => sum + card.wins, 0);
  const totalBets = cards.reduce((sum, card) => sum + card.bet_count, 0);
  const totalRisk = cards.reduce((sum, card) => sum + card.total_risk, 0);
  const totalPnl = cards.reduce((sum, card) => sum + card.total_pnl, 0);
  const roi = totalRisk ? (totalPnl / totalRisk) * 100 : null;

  const items = [
    { label: 'Cards', value: cards.length },
    { label: 'Tracked Bets', value: totalBets },
    { label: 'Correct', value: settled ? `${wins}/${settled}` : '--' },
    { label: 'ROI', value: formatPercent(roi), className: valueClass(roi || 0) },
    { label: 'P&L', value: formatCurrency(totalPnl), className: valueClass(totalPnl) },
  ];

  document.getElementById('overallStrip').innerHTML = items.map(item => `
    <div class="overall-stat ${item.className || ''}">
      <span class="overall-label">${item.label}</span>
      <strong class="overall-value">${item.value}</strong>
    </div>
  `).join('');
}

function renderCards(cards) {
  const root = document.getElementById('cardsList');
  root.innerHTML = cards.map((card) => {
    const roiClass = valueClass(card.roi || 0);
    const pnlClass = valueClass(card.total_pnl);
    const correctText = card.settled_count ? `${card.wins}/${card.settled_count} correct` : 'No settled bets';
    const pendingText = card.pending_count ? ` · ${card.pending_count} pending` : '';

    const rows = card.bets.map(fight => {
      const bet = fight.bet;
      const result = resultLabel(bet);
      const resultClass = !bet.settled ? 'neutral' : bet.won ? 'positive' : 'negative';
      const extraMeta = metaParts(fight);
      return `
        <tr>
          <td>
            <div class="bet-matchup">${fight.matchup}</div>
            <div class="bet-subtext">${fight.winner ? `Winner: ${fight.winner}` : 'Awaiting result'}</div>
            ${extraMeta.length ? `<div class="bet-subtext">${extraMeta.join(' · ')}</div>` : ''}
          </td>
          <td>
            <div class="bet-pick">${bet.fighter}</div>
            <div class="bet-subtext">vs ${bet.opponent || '--'}</div>
          </td>
          <td>${formatOdds(bet.odds)}</td>
          <td>${formatOdds(bet.current_odds)}</td>
          <td>$${bet.stake.toFixed(2)}</td>
          <td class="${valueClass(fight.edge || 0)}">${fight.edge !== null && fight.edge !== undefined ? `${fight.edge > 0 ? '+' : ''}${fight.edge.toFixed(1)}%` : '--'}</td>
          <td class="${resultClass}">${result}</td>
          <td class="${valueClass(bet.pnl || 0)}">${formatCurrency(bet.pnl)}</td>
        </tr>
      `;
    }).join('');

    return `
      <details class="bets-card">
        <summary class="bets-card-summary">
          <div>
            <div class="bets-card-date">${formatDateLabel(card.event_date)}</div>
            <div class="bets-card-name">${card.event_name}</div>
          </div>
          <div class="bets-card-metrics">
            <span class="metric-chip ${roiClass}">ROI ${formatPercent(card.roi)}</span>
            <span class="metric-chip">${correctText}${pendingText}</span>
            <span class="metric-chip ${pnlClass}">P&L ${formatCurrency(card.total_pnl)}</span>
          </div>
        </summary>
        <div class="bets-card-body">
          <div class="bets-card-caption">${card.bet_count} tracked bets</div>
          <table class="bets-table">
            <thead>
              <tr>
                <th>Matchup</th>
                <th>Bet</th>
                <th>Bet Odds</th>
                <th>Current Odds</th>
                <th>Stake</th>
                <th>Edge</th>
                <th>Result</th>
                <th>P&L</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </details>
    `;
  }).join('');
}

async function init() {
  try {
    const response = await fetch('/api/bets');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const cards = await response.json();

    document.getElementById('loadingState').classList.add('hidden');

    if (!cards.length) {
      document.getElementById('emptyState').classList.remove('hidden');
      return;
    }

    renderOverall(cards);
    renderCards(cards);
    document.getElementById('content').classList.remove('hidden');
  } catch (error) {
    document.getElementById('loadingState').classList.add('hidden');
    const node = document.getElementById('errorState');
    node.textContent = `Error loading bets: ${error.message}`;
    node.classList.remove('hidden');
  }
}

init();
