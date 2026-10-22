#!/usr/bin/env bash
# Build paper.pdf (IEEEtran 2 kolom) dari draft.md.
# Membutuhkan: pandoc (~/bin), TinyTeX (~/.TinyTeX).
set -euo pipefail
cd "$(dirname "$0")/../paper"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/bin:$PATH"

pandoc draft.md -s -o paper_raw.tex \
  --citeproc --bibliography references.bib \
  -V documentclass=IEEEtran -V classoption=conference \
  -V header-includes='\usepackage{times}' \
  -V linkcolor=black -V urlcolor=black

python3 - <<'EOF'
import re

lines = open("paper_raw.tex").read().splitlines()
out = []
state = "normal"
spec = ""
caption_pending = None
i = 0
SKIP_TOK = (r"\endhead", r"\endfirsthead", r"\endfoot", r"\endlastfoot")


def braces_balanced(s):
    return s.count("{") == s.count("}")


def extract_spec(lines, i):
    """Ambil spesifikasi kolom longtable (brace-aware, multi-baris).
    Return (spec, i_baris_terakhir)."""
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


while i < len(lines):
    ln = lines[i]
    if state == "normal":
        if ln.startswith(r"\begin{longtable}"):
            spec, i = extract_spec(lines, i)
            out.append(r"\begin{table*}[t]")
            out.append(r"\centering")
            state = "await_caption"
        else:
            out.append(ln)
    elif state == "await_caption":
        if ln.startswith(SKIP_TOK) or ln.strip() == "":
            pass
        elif ln.startswith(r"\caption"):
            cap = ln
            while not cap.rstrip().endswith((r"\\", r"\tabularnewline")) \
                    and i + 1 < len(lines):
                i += 1
                cap += " " + lines[i].strip()
            cap = re.sub(r"\\tabularnewline\s*$", "", cap.strip())
            cap = re.sub(r"\\\\\s*$", "", cap)
            caption_pending = cap.strip()
            state = "await_tabular"
        else:
            if caption_pending:
                out.append(caption_pending)
                caption_pending = None
            out.append(r"\begin{tabular}{" + spec + "}")
            state = "in_tabular"
            continue  # proses ulang baris ini
    elif state == "await_tabular":
        if ln.startswith(SKIP_TOK) or ln.strip() == "":
            pass
        else:
            if caption_pending:
                out.append(caption_pending)
                caption_pending = None
            out.append(r"\begin{tabular}{" + spec + "}")
            state = "in_tabular"
            continue
    elif state == "in_tabular":
        if ln.startswith(r"\endfirsthead"):
            state = "skip_cont_header"
        elif ln.startswith((r"\endhead", r"\endfoot", r"\endlastfoot")):
            pass
        elif ln.startswith(r"\end{longtable}"):
            out.append(r"\end{tabular}")
            out.append(r"\end{table*}")
            state = "normal"
        else:
            out.append(ln)
    elif state == "skip_cont_header":
        if ln.startswith(r"\endhead"):
            state = "in_tabular"
    i += 1

open("paper.tex", "w").write("\n".join(out) + "\n")
print("tables converted:",
      sum(1 for l in out if l.startswith(r"\begin{table*}")),
      "| captions:", sum(1 for l in out if l.startswith(r"\caption")))
EOF

pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
echo "OK: paper/paper.pdf ($(pdfinfo paper.pdf 2>/dev/null | grep Pages || echo 'pages?'))"
