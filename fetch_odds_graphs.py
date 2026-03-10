#!/usr/bin/env python3
"""
Fetch odds graphs for UFC fights from bestfightodds.com (opening + closing odds).

Usage:
    python fetch_odds_graphs.py <event_url_or_id> [-o output.json]

Example:
    python fetch_odds_graphs.py https://www.bestfightodds.com/events/ufc-3971
    python fetch_odds_graphs.py ufc-3971 -o data/odds/graphs/ufc-3971_odds.json

Output JSON is used by analysis/clv_analysis.py for Closing Line Value (CLV) analysis.
"""

import requests
import base64
import json
import re
import sys
from datetime import datetime
from typing import Optional, Dict, List, Any

# Character set for ROT-47 decoding
CHARSET = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"


def decode_graph_data(b64_data: str) -> Optional[List[Dict]]:
    """Decode the base64 + ROT-47 encoded graph data"""
    if not b64_data or b64_data == '[]':
        return None
    try:
        decoded_bytes = base64.b64decode(b64_data)
        decoded_str = decoded_bytes.decode("latin-1")

        result = ""
        for c in decoded_str:
            idx = CHARSET.find(c)
            if idx >= 0:
                result += CHARSET[(idx + len(CHARSET) // 2) % len(CHARSET)]
            else:
                result += c

        return json.loads(result)
    except Exception as e:
        print(f"  Decode error: {e}")
        return None


def decimal_to_american(decimal: float) -> str:
    """Convert decimal odds to American odds"""
    if decimal >= 2.0:
        return f"+{int((decimal - 1) * 100)}"
    else:
        return f"-{int(100 / (decimal - 1))}"


def american_to_decimal(american: str) -> float:
    """Convert American odds to decimal"""
    american = american.replace('+', '')
    american = int(american)
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def get_event_url(event_id: str) -> str:
    """Convert event ID to full URL"""
    if event_id.startswith('http'):
        return event_id
    if event_id.startswith('ufc-'):
        return f'https://www.bestfightodds.com/events/{event_id}'
    if event_id.isdigit():
        return f'https://www.bestfightodds.com/events/ufc-{event_id}'
    return f'https://www.bestfightodds.com/events/{event_id}'


def fetch_odds_graphs(event_url: str) -> Dict[str, Any]:
    """Fetch all odds graphs for an event"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15',
        'Referer': 'https://www.bestfightodds.com/'
    }

    # Fetch the event page
    print(f"Fetching event page: {event_url}")
    response = requests.get(event_url, headers=headers)
    html = response.text

    # Find all matchup IDs (5-digit numbers in data-li attributes)
    matchup_ids = set(re.findall(r'data-li="\[\d+,[12],(\d{5})', html))
    print(f"Found {len(matchup_ids)} unique matchups")

    # For each matchup, find the fighter names
    fights = {}
    for mu_id in sorted(matchup_ids):
        # Find fighter names near this matchup
        # Pattern: <span class="t-b-fcc">Fighter Name</span></a></th>...<td...data-li="[bookie,1,MATCHUP_ID]"
        context_pattern = rf'<span class="t-b-fcc">([^<]+)</span></a></th>[^<]*<td[^>]*data-li="\[\d+,1,{mu_id}\]"'
        fighter1_match = re.search(context_pattern, html)

        context_pattern2 = rf'<span class="t-b-fcc">([^<]+)</span></a></th>[^<]*<td[^>]*data-li="\[\d+,2,{mu_id}\]"'
        fighter2_match = re.search(context_pattern2, html)

        fighter1 = fighter1_match.group(1) if fighter1_match else f"Fighter 1"
        fighter2 = fighter2_match.group(1) if fighter2_match else f"Fighter 2"

        fights[mu_id] = [fighter1.strip(), fighter2.strip()]

    # Prepare API headers
    api_headers = {
        **headers,
        'Accept': '*/*',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': event_url
    }

    # Fetch graph data for each fight
    all_fight_data = {}

    for mu_id in sorted(fights.keys()):
        fighters = fights[mu_id]
        print(f"\n{fighters[0]} vs {fighters[1]} (ID: {mu_id})")
        print("-" * 50)

        fight_data = {
            'matchup_id': mu_id,
            'fighters': fighters,
            'graphs': {}
        }

        for p, fighter_name in [(1, fighters[0]), (2, fighters[1])]:
            # Try multiple pt values - different matchups may have different data available
            # pt=1 and pt=5 seem to be the most common for moneyline odds
            best_data = None
            best_pt = None

            for pt in [5, 1, 2, 3]:  # Try in order of preference
                api_url = f'https://www.bestfightodds.com/api/ggd?m={mu_id}&p={p}&pt={pt}&tn=0'

                try:
                    resp = requests.get(api_url, headers=api_headers)
                    if resp.status_code == 200 and resp.text.strip() and resp.text != '[]':
                        graph_data = decode_graph_data(resp.text)

                        if graph_data and isinstance(graph_data, list) and len(graph_data) > 0:
                            data_series = graph_data[0]
                            if 'data' in data_series and data_series['data']:
                                if best_data is None or len(data_series['data']) > len(best_data['data']):
                                    best_data = data_series
                                    best_pt = pt
                except Exception:
                    continue

            if best_data and best_data['data']:
                data_points = best_data['data']
                first_point = data_points[0]
                last_point = data_points[-1]

                first_date = datetime.fromtimestamp(first_point['x']/1000).strftime('%Y-%m-%d %H:%M')
                last_date = datetime.fromtimestamp(last_point['x']/1000).strftime('%Y-%m-%d %H:%M')

                first_odds = decimal_to_american(first_point['y'])
                last_odds = decimal_to_american(last_point['y'])

                print(f"  {fighter_name}:")
                print(f"    Opening ({first_date}): {first_odds} ({first_point['y']:.4f})")
                print(f"    Current  ({last_date}): {last_odds} ({last_point['y']:.4f})")
                print(f"    Data points: {len(data_points)} (pt={best_pt})")

                fight_data['graphs'][fighter_name] = {
                    'series_name': best_data.get('name', 'Mean'),
                    'pt_value': best_pt,
                    'opening': {
                        'date': first_date,
                        'timestamp': first_point['x'],
                        'american_odds': first_odds,
                        'decimal_odds': first_point['y']
                    },
                    'current': {
                        'date': last_date,
                        'timestamp': last_point['x'],
                        'american_odds': last_odds,
                        'decimal_odds': last_point['y']
                    },
                    'total_points': len(data_points),
                    'data_points': data_points
                }
            else:
                print(f"  {fighter_name}: No graph data available")

        all_fight_data[mu_id] = fight_data

    return all_fight_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch odds graphs from bestfightodds.com")
    parser.add_argument("event", help="Event URL, slug (ufc-3971), or ID (3971)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: /tmp/<event_slug>_odds.json)")
    args = parser.parse_args()

    event_input = args.event
    event_url = get_event_url(event_input)

    # Extract event name for output file
    event_name = event_url.split('/')[-1]

    print("=" * 70)
    print(f"ODDS GRAPHS FOR {event_name.upper()}")
    print("=" * 70)

    all_fight_data = fetch_odds_graphs(event_url)

    output_file = args.output or f'/tmp/{event_name}_odds.json'
    with open(output_file, 'w') as f:
        json.dump(all_fight_data, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"Data saved to {output_file}")
    print("Run CLV analysis:  python analysis/clv_analysis.py " + output_file)

    return all_fight_data


if __name__ == '__main__':
    main()
