#!/usr/bin/env python3
"""Production-switch decision: stock 1.5.1 + DFlash2 vs skinny, from quality_gate.py records.
  quality_compare.py <skinny.json> <stock.json> [<skinny-tools.json>]
Thresholds (fixed 2026-10-18, before the stock arm ran; told to Brandon): stock has no gross failure (0 request errors,
needles 5/5 at both depths, tools 4/4) AND needles >= skinny at both depths AND AIME correct >= skinny - 1 AND
AIME finish=length <= skinny + 1. n = 9 AIME runs: this catches gross breakage, not small quality shifts.
Exit 0 = PASS, 1 = FAIL.
"""
import json, sys

sk, st = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
skt = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
fl = lambda r: sum(x.get("finish") == "length" for x in r.get("aime", []))
checks = [
    ("stock: no gross failure", not st.get("gross_failure", True), f"gross_failure={st.get('gross_failure')}"),
    ("stock: 0 AIME request errors", st["aime_errors"] == 0, f"{st['aime_errors']} errors"),
    ("needles >= skinny @64K/@120K", all(a >= b for a, b in zip(st["needle_found"], sk["needle_found"])),
     f"stock {st['needle_found']} vs skinny {sk['needle_found']}"),
    ("AIME correct >= skinny - 1", st["aime_correct"] >= sk["aime_correct"] - 1,
     f"stock {st['aime_correct']}/{st['aime_total']} vs skinny {sk['aime_correct']}/{sk['aime_total']}"),
    ("AIME finish=length <= skinny + 1", fl(st) <= fl(sk) + 1, f"stock {fl(st)} vs skinny {fl(sk)}"),
    ("stock tools 4/4", st.get("tools_pass") == 4, f"stock {st.get('tools_pass')}/4, skinny {skt.get('tools_pass', 'n/a')}/4"),
]
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
print(f"stock AIME tau (thinking on, T 0.6): {st.get('aime_tau')}")
verdict = all(ok for _, ok, _ in checks)
print("VERDICT:", "PASS — switch allowed" if verdict else "FAIL — keep skinny")
sys.exit(0 if verdict else 1)
