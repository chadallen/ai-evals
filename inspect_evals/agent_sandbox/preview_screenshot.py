"""Capture one page inside the isolated preview network."""

import asyncio
import base64
import sys
from urllib.parse import urlsplit

from playwright.async_api import async_playwright


async def capture(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("preview URL must be an absolute HTTP or HTTPS URL")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            await page.goto(url, wait_until="networkidle", timeout=15_000)
            image = await page.screenshot(full_page=True, type="png")
        finally:
            await browser.close()
    print(base64.b64encode(image).decode("ascii"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: preview_screenshot.py URL")
    asyncio.run(capture(sys.argv[1]))
