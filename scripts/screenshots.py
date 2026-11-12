"""Capture screenshots of the real UI with a headless Chromium of its own, and report console errors.

Usage: python scripts/screenshots.py [--base http://127.0.0.1:8107] [--out docs/assets] [--serve]
--serve starts `voicescribe serve` on port 8107 for the duration of the run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = [
    ("desktop", {"width": 1440, "height": 900}, False),
    ("phone", {"width": 390, "height": 844}, True),
]


def wait_up(base: str, tries: int = 60) -> None:
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"{base}/api/health", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server did not start")


def seek(page, seconds: float) -> None:
    page.evaluate(
        """(s) => { const el = document.querySelector('#scrub'); el.value = String(Math.round(s / document.querySelector('#audio').duration * 1000));
        el.dispatchEvent(new Event('input', {bubbles: true})); }""",
        seconds,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8107")
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "voicescribe.cli", "serve"]) if args.serve else None
    errors: list[str] = []
    try:
        if proc:
            wait_up(args.base)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
            for name, size, touch in SHOTS:
                ctx = browser.new_context(viewport=size, has_touch=touch, is_mobile=touch, device_scale_factor=2 if touch else 1)
                page = ctx.new_page()
                page.on("console", lambda m, n=name: errors.append(f"[{n}] {m.text}") if m.type == "error" else None)
                page.on("pageerror", lambda e, n=name: errors.append(f"[{n}] pageerror {e}"))
                page.goto(args.base, wait_until="networkidle")
                page.wait_for_function("document.querySelector('#play') && !document.querySelector('#play').disabled")
                page.screenshot(path=str(out / f"{name}-start.png"))
                seek(page, 53)
                page.click("#play")
                page.wait_for_timeout(5000)
                page.screenshot(path=str(out / f"{name}-playing.png"))
                seek(page, 117)
                page.wait_for_timeout(600)
                page.screenshot(path=str(out / f"{name}-end.png"), full_page=True)
                overflow = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
                print(f"{name}: horizontal overflow = {overflow}")
                small = page.evaluate(
                    """() => [...document.querySelectorAll('button, .mode, input[type=range], label.btn, a.btn')]
                    .filter(e => e.offsetParent !== null).map(e => [e.id || e.className || e.tagName, e.getBoundingClientRect()])
                    .filter(([n, r]) => r.height < 44 || (r.width < 44 && !/scrub/.test(n))).map(([n, r]) => n + ' ' + Math.round(r.width) + 'x' + Math.round(r.height))"""
                )
                print(f"{name}: controls under 44 px in a dimension: {small or 'none'}")
                if touch:  # reveal stays locked to the audio clock: words shown == words whose start <= currentTime
                    page.evaluate("document.querySelector('#audio').pause()")
                    seek(page, 61.3)
                    page.wait_for_timeout(500)
                    res = page.evaluate(
                        """async () => { const m = await (await fetch('demo/meeting.json')).json(); const t = document.querySelector('#audio').currentTime;
                        const want = m.transcript.sentences.flatMap(s => s.words).filter(w => w.start <= t).length;
                        return [want, document.querySelectorAll('#transcript .w.on').length, t]; }"""
                    )
                    print(f"{name}: reveal sync (expected words, shown words, t) = {res}")
                ctx.close()
            browser.close()
    finally:
        if proc:
            proc.terminate()
    if errors:
        print("console errors:")
        print("\n".join(errors))
        return 1
    print("no console errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
