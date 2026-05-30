"""Fetch Google Fonts (Google Sans, Noto Sans Thai, Google Sans Code),
embed the latin/latin-ext/thai woff2 subsets as base64, write embedded.css.
Writes a status line to _status.txt so we can verify via the Read tool
(the interactive shell garbles stdout)."""
import re, urllib.request, base64, os, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
FAMILIES = [
    "family=Google+Sans:wght@400;500;700",
    "family=Noto+Sans+Thai:wght@400;500;700",
    "family=Google+Sans+Code:wght@400;500;700",
]
KEEP = {"latin", "latin-ext", "thai"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=30).read()


def main():
    css_parts = []
    for fam in FAMILIES:
        url = f"https://fonts.googleapis.com/css2?{fam}&display=swap"
        css_parts.append(get(url).decode("utf-8"))
    css = "\n".join(css_parts)

    blocks = re.split(r'(/\*[^*]*\*/)', css)
    out = []
    seen = 0
    for i in range(1, len(blocks), 2):
        sub = blocks[i].strip("/* \n")
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        if sub not in KEEP:
            continue
        m = re.search(r'src:\s*url\((https://[^)]+\.woff2)\)', body)
        if not m:
            continue
        data = get(m.group(1))
        b64 = base64.b64encode(data).decode()
        inner = body.split('{', 1)[1]
        inner = inner.replace(m.group(1), f"data:font/woff2;base64,{b64}")
        out.append(f"/* {sub} */\n@font-face {{{inner}")
        seen += 1

    path = os.path.join(HERE, "embedded.css")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    with open(os.path.join(HERE, "_status.txt"), "w") as f:
        f.write(f"OK faces={seen} bytes={os.path.getsize(path)}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(os.path.join(HERE, "_status.txt"), "w") as f:
            f.write("ERROR\n" + traceback.format_exc())
