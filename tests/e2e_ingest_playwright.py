"""End-to-end Playwright validation of the /ingest Fighter + Aliases UI.

Drives a real browser against a locally running server (default :8055):
  1. Page + section render.
  2. Alias manager: add via UI -> row appears + count bumps; delete -> row gone.
  3. Fighter preview against the live RJ Harris Sherdog URL -> preview grid shows
     the cleaned name and the suggested alias (dry-run, no DB write).

Run:  BASE=http://127.0.0.1:8055 python tests/e2e_ingest_playwright.py
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("BASE", "http://127.0.0.1:8055")
SHERDOG_URL = "https://www.sherdog.com/fighter/Richard-Harris-429363"
TEST_ALIAS = "E2E Test Alias"
TEST_CANON = "Max Holloway"

failures = []


def check(desc, cond):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {desc}")
    if not cond:
        failures.append(desc)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"{BASE}/ingest", wait_until="networkidle")

        print("\n== 1. Page render ==")
        check("Fighter Ingest heading present", "Fighter Ingest" in page.content())
        check("sherdogUrl input present", page.locator("#sherdogUrl").count() == 1)
        check("alias manager table present", page.locator("#aliasTableBody").count() == 1)

        print("\n== 2. Alias manager add/delete ==")
        # wait for aliases to load
        page.wait_for_function("document.querySelectorAll('#aliasTableBody tr').length > 0")
        start_rows = page.locator("#aliasTableBody tr").count()
        print(f"  (loaded {start_rows} existing aliases)")

        page.fill("#newAliasFrom", TEST_ALIAS)
        page.fill("#newAliasTo", TEST_CANON)
        page.click("#addAliasBtn")
        page.wait_for_function(
            "a => document.querySelector('#aliasTableBody').innerText.includes(a)",
            arg=TEST_ALIAS,
        )
        after_add = page.locator("#aliasTableBody tr").count()
        check("row count increased after add", after_add == start_rows + 1)
        check("new alias visible in table", TEST_ALIAS in page.locator("#aliasTableBody").inner_text())

        # delete it (accept the confirm dialog)
        page.on("dialog", lambda d: d.accept())
        page.click(f'.alias-del[data-alias="{TEST_ALIAS}"]')
        page.wait_for_function(
            "a => !document.querySelector('#aliasTableBody').innerText.includes(a)",
            arg=TEST_ALIAS,
        )
        after_del = page.locator("#aliasTableBody tr").count()
        check("row count back to start after delete", after_del == start_rows)

        print("\n== 3. Fighter preview (live Sherdog, dry-run) ==")
        page.fill("#sherdogUrl", SHERDOG_URL)
        page.fill("#requestedName", "RJ Harris")
        page.click("#previewBtn")
        try:
            page.wait_for_selector("#fPreview", state="visible", timeout=30000)
            page.wait_for_function(
                "document.querySelector('#fpGrid') && document.querySelector('#fpGrid').innerText.length > 0",
                timeout=30000,
            )
            grid = page.locator("#fpGrid").inner_text()
            print("  preview grid text:\n    " + grid.replace("\n", "\n    "))
            check("preview shows cleaned name 'Richard Harris'", "Richard Harris" in grid)
            check("suggested alias fields populated",
                  page.locator("#aliasFrom").input_value() == "RJ Harris"
                  and page.locator("#aliasTo").input_value() == "Richard Harris")
        except Exception as e:
            check(f"fighter preview rendered (error: {e})", False)

        browser.close()

    print("\n" + "=" * 50)
    if failures:
        print(f"E2E FAILED — {len(failures)} check(s) failed:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("E2E PASSED — all checks green.")


if __name__ == "__main__":
    main()
