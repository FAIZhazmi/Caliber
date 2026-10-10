"""Open the deployed Streamlit app in a real browser so it never goes to sleep.

Streamlit Community Cloud only counts a visit that opens the page and its websocket,
so a plain HTTP ping is not enough. If the app is already asleep, this clicks the
"get this app back up" button and waits until the app has actually loaded.
Exits non-zero when the app cannot be woken, so GitHub emails the repo owner.
"""

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://coreintelligent.streamlit.app")
WAKE_BUTTON = "button:has-text('get this app back up')"
APP_FRAME = "iframe[title='streamlitApp']"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(5_000)

        wake = page.locator(WAKE_BUTTON)
        if wake.count() > 0:
            print("App was asleep, clicking wake-up button.")
            wake.first.click()
        else:
            print("App looks awake.")

        try:
            page.wait_for_selector(APP_FRAME, timeout=180_000)
            frame = page.frame_locator(APP_FRAME)
            frame.locator("[data-testid='stApp']").wait_for(timeout=120_000)
        except PlaywrightTimeout:
            page.screenshot(path="keep_alive_failure.png")
            print("App did not finish loading within the timeout.", file=sys.stderr)
            browser.close()
            return 1

        # Stay connected briefly so the visit registers as real traffic.
        page.wait_for_timeout(20_000)
        print("App is up.")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
