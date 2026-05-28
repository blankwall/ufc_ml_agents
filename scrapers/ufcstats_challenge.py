from __future__ import annotations

import hashlib
import re
from urllib.parse import urljoin


_NONCE_RE = re.compile(r'var\s+nonce\s*=\s*"([^"]+)"')
_DIFFICULTY_RE = re.compile(r"target\s*=\s*new\s+Array\((\d+)\s*\+\s*1\)\.join\('0'\)")
_MAX_PROOF_OF_WORK_ATTEMPTS = 10_000_000


def is_ufcstats_challenge(html: str) -> bool:
    return all(marker in html for marker in ("Checking your browser", "/__c"))


def solve_ufcstats_challenge(html: str) -> tuple[str, int]:
    nonce_match = _NONCE_RE.search(html)
    difficulty_match = _DIFFICULTY_RE.search(html)
    if not nonce_match or not difficulty_match:
        raise ValueError("UFCStats browser challenge format was not recognized")

    nonce = nonce_match.group(1)
    difficulty = int(difficulty_match.group(1))
    target = "0" * difficulty
    for n in range(_MAX_PROOF_OF_WORK_ATTEMPTS):
        digest = hashlib.sha256(f"{nonce}:{n}".encode("utf-8")).hexdigest()
        if digest.startswith(target):
            return nonce, n

    raise RuntimeError("UFCStats browser challenge proof-of-work exceeded attempt limit")


def fetch_ufcstats_html(session, url: str, *, timeout: int | float) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    html = response.text
    if not is_ufcstats_challenge(html):
        return html

    nonce, n = solve_ufcstats_challenge(html)
    challenge_url = urljoin(url, "/__c")
    challenge_response = session.post(
        challenge_url,
        data={"nonce": nonce, "n": str(n)},
        timeout=timeout,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": url,
        },
    )
    challenge_response.raise_for_status()

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text
