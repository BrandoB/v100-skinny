#!/usr/bin/env python3
"""Concurrency sweep for the v100-skinny vLLM lane (s358, 2026-09-14).

Fires N concurrent streaming chat requests (same prompt) and reports, per
round: per-stream decode tok/s, aggregate tok/s, TTFT, and MTP acceptance
scraped from vLLM's Prometheus counters before/after the round.

  # manual :8096 lane, booted with MNS=4 MBT=8192 (capture sizes follow MNS)
  .venv-sm70/bin/python benchmarks/concurrency_sweep.py --streams 1,2,4
  # through llama-swap (metrics via the upstream passthrough)
  .venv-sm70/bin/python benchmarks/concurrency_sweep.py --base http://127.0.0.1:8082 \
      --metrics-url http://127.0.0.1:8082/upstream/qwen3.8-27b-skinny/metrics --streams 1

Short prompts on purpose: the KV cache holds ~1.5 full contexts, so long
prompts at N=4 would preempt and muddy the read. Writes a JSON record next
to the markdown summary under benchmarks/results/.
"""
import argparse, json, re, threading, time, urllib.request
from datetime import datetime
from pathlib import Path

PROMPT = ("Write a detailed, step-by-step explanation of how a four-stroke engine "
          "works, then list ten common failure modes with a one-line cause each. "
          "Be thorough and keep going until you reach the token limit.")

SPEC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total")


def scrape(url: str) -> dict:
    try:
        txt = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception as e:
        return {"_error": str(e)}
    out = {}
    for ln in txt.splitlines():
        for k in SPEC:
            if ln.startswith(k + "{") or ln.startswith(k + " "):
                out[k] = float(ln.rsplit(" ", 1)[1])
    return out


def one_stream(base, model, max_tokens, idx, results, thinking):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}]}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; chunks = 0; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0].get("delta") or {}
                if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning"):
                    chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
    except Exception as e:
        err = str(e)
    t_end = time.perf_counter()
    toks = (usage or {}).get("completion_tokens") or chunks  # chunks undercount under MTP
    decode_s = (t_end - t_first) if t_first else 0.0
    results[idx] = {"stream": idx, "ttft_s": round((t_first - t0), 3) if t_first else None,
                    "wall_s": round(t_end - t0, 2), "completion_tokens": toks,
                    "usage_exact": usage is not None,
                    "decode_tok_s": round((toks - 1) / decode_s, 1) if decode_s > 0 and toks > 1 else None,
                    "error": err}


def run_round(n, args):
    before = scrape(args.metrics_url)
    results = [None] * n
    ths = [threading.Thread(target=one_stream, args=(args.base, args.model, args.max_tokens, i, results, args.thinking))
           for i in range(n)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    after = scrape(args.metrics_url)
    ok = [r for r in results if r and not r["error"]]
    total_toks = sum(r["completion_tokens"] for r in ok)
    per = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
    spec = {}
    if "_error" not in before and "_error" not in after and all(k in before and k in after for k in SPEC):
        d = {k: after[k] - before[k] for k in SPEC}
        drafts, dtoks, acc = (d[SPEC[0]], d[SPEC[1]], d[SPEC[2]])
        spec = {"drafts": drafts, "draft_tokens": dtoks, "accepted": acc,
                "accept_rate": round(acc / dtoks, 3) if dtoks else None,
                "accepted_per_draft": round(acc / drafts, 2) if drafts else None}
    return {"streams": n, "wall_s": round(wall, 2), "errors": n - len(ok),
            "aggregate_tok_s": round(total_toks / wall, 1) if wall else None,
            "per_stream_decode_tok_s": per,
            "per_stream_mean": round(sum(per) / len(per), 1) if per else None,
            "ttft_s": [r["ttft_s"] for r in ok], "usage_exact": all(r["usage_exact"] for r in ok),
            "spec": spec, "streams_detail": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8096")
    ap.add_argument("--model", default="qwen3.8-27b-skinny")
    ap.add_argument("--metrics-url", default=None, help="default: <base>/metrics")
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", action="store_true", help="leave thinking on (default: off, so tokens are all visible output)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.metrics_url = args.metrics_url or args.base.rstrip("/") + "/metrics"
    rounds = []
    for n in [int(x) for x in args.streams.split(",") if x.strip()]:
        print(f"== {n} stream(s) …", flush=True)
        r = run_round(n, args); rounds.append(r)
        print(f"   wall {r['wall_s']}s · aggregate {r['aggregate_tok_s']} tok/s · per-stream "
              f"{r['per_stream_decode_tok_s']} (mean {r['per_stream_mean']}) · ttft {r['ttft_s']} · "
              f"errors {r['errors']} · usage_exact {r['usage_exact']} · spec {r['spec'] or 'n/a'}", flush=True)
    out = Path(__file__).resolve().parent / "results"; out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    rec = {"ts": stamp, "base": args.base, "model": args.model, "max_tokens": args.max_tokens,
           "thinking": args.thinking, "rounds": rounds}
    (out / f"concurrency_{stamp}.json").write_text(json.dumps(rec, indent=1))
    md = ["| streams | aggregate tok/s | per-stream mean | per-stream | ttft s | accept rate | acc/draft | errors |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rounds:
        md.append(f"| {r['streams']} | {r['aggregate_tok_s']} | {r['per_stream_mean']} | {r['per_stream_decode_tok_s']} | "
                  f"{r['ttft_s']} | {r['spec'].get('accept_rate') if r['spec'] else 'n/a'} | "
                  f"{r['spec'].get('accepted_per_draft') if r['spec'] else 'n/a'} | {r['errors']} |")
    (out / f"concurrency_{stamp}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"record: {out}/concurrency_{stamp}.json")


if __name__ == "__main__":
    main()
