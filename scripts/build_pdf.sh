#!/usr/bin/env bash
# Build paper.pdf (IEEEtran 2 kolom) dari draft markdown.
# Pemakaian: build_pdf.sh [input.md] [output.pdf]
#   default: build_pdf.sh draft.md paper.pdf
# Membutuhkan: pandoc (~/bin), TinyTeX (~/.TinyTeX).
set -euo pipefail
cd "$(dirname "$0")/../paper"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/bin:$PATH"
IN="${1:-draft.md}"
OUT="${2:-paper.pdf}"
BASE="${OUT%.pdf}"

pandoc "$IN" -s -o "${BASE}_raw.tex" \
  --citeproc --bibliography references.bib \
  -V documentclass=IEEEtran -V classoption=conference \
  -V header-includes='\usepackage{times}' \
  -V linkcolor=black -V urlcolor=black

export RAW_TEX="${BASE}_raw.tex" OUT_TEX="${BASE}.tex"
python3 - <<'EOF'
import os
import re

lines = open(os.environ["RAW_TEX"]).read().splitlines()
out = []
state = "normal"
spec = ""
caption = None
body = []
i = 0
SKIP_TOK = (r"\endhead", r"\endfirsthead", r"\endfoot", r"\endlastfoot")
NARROW_MAX = 58  # karakter -> satu kolom; lebih lebar -> dua kolom


def extract_spec(lines, i):
    """Ambil spesifikasi kolom longtable (brace-aware, multi-baris)."""
    s = lines[i]
    start = s.index("[]{") + 3
    buf = s[start:]
    depth = 1
    j = 0
    while True:
        while j < len(buf) and depth > 0:
            if buf[j] == "{":
                depth += 1
            elif buf[j] == "}":
                depth -= 1
            j += 1
        if depth <= 0:
            return buf[:j - 1], i
        i += 1
        buf += lines[i]


def clean_caption(lines, i):
    cap = lines[i]
    while not cap.rstrip().endswith((r"\\", r"\tabularnewline")) \
            and i + 1 < len(lines):
        i += 1
        cap += " " + lines[i].strip()
    cap = re.sub(r"\\tabularnewline\s*$", "", cap.strip())
    cap = re.sub(r"\\\\\s*$", "", cap)
    return cap.strip(), i


def flush():
    width = max((len(l) for l in body), default=0)
    env = "table" if width <= NARROW_MAX else "table*"
    spec_float = "[!tb]" if env == "table" else "[!t]"
    out.append(r"\begin{%s}%s" % (env, spec_float))
    out.append(r"\centering")
    if caption:
        out.append(caption)
    out.append(r"\begin{tabular}{" + spec + "}")
    out.extend(body)
    out.append(r"\end{tabular}")
    out.append(r"\end{%s}" % env)


while i < len(lines):
    ln = lines[i]
    if state == "normal":
        if ln.startswith(r"\begin{longtable}"):
            spec, i = extract_spec(lines, i)
            caption = None
            body = []
            state = "await_caption"
        else:
            out.append(ln)
    elif state == "await_caption":
        if ln.startswith(SKIP_TOK) or ln.strip() == "":
            pass
        elif ln.startswith(r"\caption"):
            caption, i = clean_caption(lines, i)
            state = "await_body"
        else:
            state = "await_body"
            continue
    elif state == "await_body":
        if ln.startswith(SKIP_TOK) or ln.strip() == "":
            pass
        else:
            state = "in_body"
            continue
    elif state == "in_body":
        if ln.startswith(r"\endfirsthead"):
            state = "skip_cont_header"
        elif ln.startswith((r"\endhead", r"\endfoot", r"\endlastfoot")):
            pass
        elif ln.startswith(r"\end{longtable}"):
            flush()
            state = "normal"
        else:
            body.append(ln)
    elif state == "skip_cont_header":
        if ln.startswith(r"\endhead"):
            state = "in_body"
    i += 1

open(os.environ["OUT_TEX"], "w").write("\n".join(out) + "\n")
print("tables:", sum(1 for l in out if re.match(r"\\begin\{table\*?\}", l)),
      "| single-col:", sum(1 for l in out if l == r"\begin{table}[!tb]"),
      "| captions:", sum(1 for l in out if l.startswith(r"\caption")))
EOF

pdflatex -interaction=nonstopmode -halt-on-error "${BASE}.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "${BASE}.tex" >/dev/null
echo "OK: paper/${OUT} ($(pdfinfo "${OUT}" 2>/dev/null | grep Pages || echo 'pages?'))"
