"""
End-to-end PDF builder (robust to the broken interactive shell).
  1. fetch the 3 Google font families, embed latin/latin-ext/thai woff2 as base64
  2. inline that CSS into diffsoup_doc.html (replace the @import line)
  3. render to PDF with headless Chrome
Everything is logged to make_log.txt so it can be inspected with the Read tool.
"""
import os, re, sys, base64, urllib.request, subprocess, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LOG = open(os.path.join(HERE, "make_log.txt"), "w", encoding="utf-8")


def log(*a):
    print(*a, file=LOG, flush=True)


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
FAMILIES = [
    "family=Google+Sans:wght@400;500;700",
    "family=Noto+Sans+Thai:wght@400;500;700",
    "family=Google+Sans+Code:wght@400;500;700",
]
KEEP = {"latin", "latin-ext", "thai"}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=40).read()


def build_embedded_css():
    css = "\n".join(fetch(f"https://fonts.googleapis.com/css2?{f}&display=swap").decode()
                    for f in FAMILIES)
    blocks = re.split(r'(/\*[^*]*\*/)', css)
    out, seen = [], 0
    for i in range(1, len(blocks), 2):
        sub = blocks[i].strip("/* \n")
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        if sub not in KEEP:
            continue
        m = re.search(r'src:\s*url\((https://[^)]+\.woff2)\)', body)
        if not m:
            continue
        b64 = base64.b64encode(fetch(m.group(1))).decode()
        inner = body.split('{', 1)[1].replace(m.group(1), f"data:font/woff2;base64,{b64}")
        out.append(f"/* {sub} */\n@font-face {{{inner}")
        seen += 1
    log(f"embedded {seen} font faces")
    return "\n".join(out)


def main():
    html_path = os.path.join(HERE, "diffsoup_doc.html")
    html = open(html_path, encoding="utf-8").read()

    try:
        css = build_embedded_css()
        # Replace the @import line with the inlined @font-face rules.
        html2 = re.sub(r"@import url\([^;]+\);", css, html, count=1)
        if html2 == html:
            log("WARN: @import not found, fonts not inlined")
        html = html2
        log("fonts inlined OK")
    except Exception:
        log("FONT EMBED FAILED, keeping @import fallback:\n" + traceback.format_exc())

    final_html = os.path.join(HERE, "diffsoup_final.html")
    open(final_html, "w", encoding="utf-8").write(html)
    log("wrote final html bytes=" + str(os.path.getsize(final_html)))

    out_pdf = os.path.join(PROJ, "DiffSoup_อธิบายโค้ด.pdf")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
        "--no-pdf-header-footer", "--virtual-time-budget=15000",
        f"--print-to-pdf={out_pdf}", "file://" + final_html,
    ]
    log("running chrome...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    log("chrome rc=" + str(r.returncode))
    log("chrome stderr tail:\n" + (r.stderr or "")[-1500:])
    if os.path.exists(out_pdf):
        log(f"PDF OK: {out_pdf}  ({os.path.getsize(out_pdf):,} bytes)")
    else:
        log("PDF NOT CREATED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
    finally:
        LOG.close()
