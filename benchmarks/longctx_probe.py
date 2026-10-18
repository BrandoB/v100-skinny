#!/usr/bin/env python3
"""Stage 0 long-context decode probe (Volta roadmap Task 0.0 Step 2).

  python longctx_probe.py <base> <model> <approx_words>

Fills the prompt with repeats of benchmarks/stage0_prompt.txt and reports decode tok/s at that context.
36,800 words is ~48K tokens with this checkpoint's tokenizer; the record carries the server's prompt_tokens,
so read the context size from there, not from the word count.
Caveat: the filler is repetitive, which inflates MTP acceptance. Compare partition arms at a fixed K only.
A per-run nonce leads the prompt so prefix caching can't serve reps 2-3 from cache (TTFT stays meaningful).
Always prints one JSON line, including on failure ({"error": ...}), so a bad rep is recorded, not lost.
"""
import json, sys, time, urllib.request, uuid
from pathlib import Path

base, model, words = sys.argv[1], sys.argv[2], int(sys.argv[3])
para = (Path(__file__).resolve().parent / "stage0_prompt.txt").read_text()
filler = (para + "\n") * (words // max(1, len(para.split())) + 1)
nonce = uuid.uuid4().hex
msgs = [{"role": "user", "content": f"[run {nonce}]\n" + filler + "\n\nSummarize the text above in 400 words."}]
body = json.dumps({"model": model, "messages": msgs, "max_tokens": 512, "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True},
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
t0, first, usage = time.time(), None, None
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            if first is None and ev.get("choices") and ev["choices"][0]["delta"].get("content"):
                first = time.time()
    t1 = time.time()
    if first is None or usage is None:
        raise RuntimeError(f"incomplete stream: first_token={'yes' if first else 'no'} usage={'yes' if usage else 'no'}")
    n = usage["completion_tokens"]
    print(json.dumps({"words": words, "nonce": nonce, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": n,
                      "ttft_s": round(first - t0, 2), "decode_tps": round((n - 1) / (t1 - first), 2)}))
except Exception as e:  # recorded, not raised: one bad rep must not lose the arm's record
    print(json.dumps({"words": words, "nonce": nonce, "error": f"{type(e).__name__}: {e}",
                      "elapsed_s": round(time.time() - t0, 2)}))
    sys.exit(1)
