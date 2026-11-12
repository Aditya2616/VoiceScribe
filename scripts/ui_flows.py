"""Drive the real UI end to end in a headless Chromium: demo click-to-seek, upload, and microphone recording
(with Chromium's fake audio device). Starts its own server on port 8107 using the tiny.en model for speed."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8107"
ROOT = Path(__file__).resolve().parents[1]


def wait_up() -> None:
    for _ in range(80):
        try:
            urllib.request.urlopen(f"{BASE}/api/health", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server did not start")


def main() -> int:
    env = {**os.environ, "VOICESCRIBE_WHISPER": "tiny.en"}
    proc = subprocess.Popen([sys.executable, "-m", "voicescribe.cli", "serve"], env=env, cwd=ROOT)
    failures: list[str] = []
    errors: list[str] = []
    try:
        wait_up()
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
            )
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["microphone"])
            page = ctx.new_page()
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function("!document.querySelector('#play').disabled")

            # 1. demo: seek to the end, items are all visible; click the first item and check the audio jumps there
            page.evaluate("document.querySelector('#scrub').value = 1000; document.querySelector('#scrub').dispatchEvent(new Event('input', {bubbles: true}))")
            page.wait_for_timeout(500)
            n = page.inner_text("#n-act")
            print("demo action items shown at the end:", n)
            if int(n) < 5:
                failures.append("demo items did not appear")
            first = page.locator("#actions .card").first
            when = first.locator(".chip.time").inner_text()
            first.click()
            page.wait_for_timeout(800)
            cur = page.evaluate("document.querySelector('#audio').currentTime")
            mm, ss = when.split(":")
            target = int(mm) * 60 + int(ss)
            print(f"clicked item at {when}; audio.currentTime={cur:.2f}")
            if abs(cur - target) > 3:
                failures.append("click-to-seek did not move the audio to the item")
            page.evaluate("document.querySelector('#audio').pause()")

            # 2. upload
            page.click('[data-mode="upload"]')
            page.set_input_files("#file", str(ROOT / "data" / "demo" / "meeting.mp3"))
            page.click("#process")
            page.wait_for_selector("#pipe:not([hidden])", timeout=10000)
            page.wait_for_function("document.querySelector('#pipe').hidden", timeout=600000)
            note = page.inner_text("#intake-note")
            sentences = page.locator("#transcript .sent").count()
            print("upload: sentences =", sentences, "| note =", note or "-", "| source =", page.inner_text("#source-tag"))
            if sentences < 10:
                failures.append("upload produced too few sentences")
            page.screenshot(path=str(ROOT / "outputs" / "shots" / "upload-done.png"))

            # 3. record from the fake microphone for ~3 s
            page.click('[data-mode="record"]')
            page.click("#rec")
            page.wait_for_timeout(3000)
            page.click("#rec")
            page.wait_for_function("!document.querySelector('#process').disabled", timeout=10000)
            page.click("#process")
            page.wait_for_function("document.querySelector('#pipe').hidden && !document.querySelector('#process').disabled", timeout=300000)
            print("record: note =", page.inner_text("#intake-note") or "-", "| transcript =", page.inner_text("#transcript")[:60].replace("\n", " "))
            browser.close()
    finally:
        proc.terminate()
    if errors:
        failures.append("console errors: " + "; ".join(errors))
    print("FAILURES:" if failures else "all UI flows OK")
    for f in failures:
        print(" -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
