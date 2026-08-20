"""Capture real screenshots of the running Gradio app via headless Chromium."""
from playwright.sync_api import sync_playwright
import time

URL = "http://127.0.0.1:7861"

SHOTS = [
    ("This movie was absolutely fantastic - the acting was superb, the story kept me "
     "on the edge of my seat, and I would happily watch it again.", "gradio.png"),
    ("A boring, predictable mess with wooden acting. I nearly fell asleep and would "
     "not recommend it to anyone.", "gradio_negative.png"),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1100, "height": 720},
                            device_scale_factor=2)
    page.goto(URL, wait_until="networkidle")
    for review, fname in SHOTS:
        box = page.locator("textarea").first
        box.fill(review)
        page.get_by_role("button", name="Submit").click()
        # wait for the output textbox to contain "Prediction"
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll('textarea, .output-class, [data-testid]'))"
            ".some(e => (e.value||e.textContent||'').includes('Prediction'))",
            timeout=15000)
        time.sleep(1.2)
        page.screenshot(path=fname, full_page=True)
        print("saved", fname)
    browser.close()
