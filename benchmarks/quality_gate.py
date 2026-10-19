#!/usr/bin/env python3
"""Production-switch quality gate: the same suite against any OpenAI-compatible lane (skinny via llama-swap :8082, or
stock 1Cat 1.5.1 on the manual :8096 lane). Two parts, both scored against ground truth, never against the model:

  1. AIME 2026 fixtures 01/15/30 (benchmarks/ninfer_fixtures, truth from aime_score.py) x seeds, thinking ON,
     reasoning_effort medium, aime_score.py's sampling (T 0.6, top_p 0.95, top_k 20, presence 1.0), 24K max tokens.
     Runs up to 3 at once (production lanes are MNS=4).
  2. Needle retrieval (fp8 KV risk): five "secret code" sentences planted at 10/30/50/70/90 % of a ~64K-token and a
     ~120K-token slice of the llama-benchy book; thinking OFF, T=0; score = codes returned exactly.

  3. Tool calls (OpenCode streams them; 1.5.1's qwen3_coder parser is 928 lines vs 1.2.2's 663): one function schema,
     server-default chat kwargs (what OpenCode sends), T=0, twice non-stream + twice stream; PASS = right function
     name + JSON-parseable args with the asked city/unit.
  Also: spec-decode counters (/metrics, or QG_METRICS) before/after the AIME batch -> tau under thinking-on traffic;
  finish=length counted separately (a truncated trace can still carry a \\boxed guess).

  quality_gate.py <base> <model> <tag> [--tools-only]   -> results/stage0/quality-<tag>.json + one summary line
  Exit 1 on GROSS failure only (any request error, any needle < 5/5, any tool probe failing); the skinny-vs-stock
  comparison and its thresholds live in quality_compare.py.
Run with the benchy venv's python (it has `tokenizers`); HF stays offline (local tokenizer.json).
"""
import json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE, MODEL, TAG = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
TOOLS_ONLY = "--tools-only" in sys.argv[4:]
METRICS = os.environ.get("QG_METRICS", BASE + "/metrics")
ROOT = Path(__file__).resolve().parent.parent
FIXDIR = ROOT / "benchmarks" / "ninfer_fixtures"
TRUTH = {"01": 277, "15": 83, "30": 393}          # same as benchmarks/aime_score.py:36
SEEDS = [int(s) for s in os.environ.get("QG_SEEDS", "1001,2002,3003").split(",")]
BOOK = Path.home() / ".cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt"
TOKENIZER = "/mnt/models/RadixArk-Qwen3.8-27B-NVFP4/tokenizer.json"
BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def post(body, timeout):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def aime(fx, seed):
    msgs = json.load(open(FIXDIR / f"long_decode_aime26_{fx}.json"))
    t0 = time.time()
    try:
        d = post({"model": MODEL, "messages": msgs, "max_completion_tokens": 24000,
                  "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
                  "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.0, "seed": seed}, 7200)
        m = d["choices"][0]["message"]; fin = d["choices"][0].get("finish_reason")
        ans = None
        for text in (m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or ""):
            hits = BOXED.findall(text)
            if hits:
                g = re.search(r"-?\d+", hits[-1].replace(",", "").replace("$", ""))
                if g: ans = int(g.group(0)); break
        gen = d.get("usage", {}).get("completion_tokens")
        return {"fixture": fx, "seed": seed, "answer": ans, "truth": TRUTH[fx], "correct": ans == TRUTH[fx],
                "finish": fin, "completion_tokens": gen, "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"fixture": fx, "seed": seed, "error": f"{type(e).__name__}: {e}", "correct": False, "wall_s": round(time.time() - t0, 1)}


def needle(n_tokens, seed):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    ids = tok.encode(BOOK.read_text(encoding="utf-8", errors="ignore")).ids[:n_tokens]
    text = tok.decode(ids)
    rnd = random.Random(seed)
    doors = ["red", "blue", "green", "yellow", "purple"]
    codes = {c: f"{rnd.randrange(100000, 999999)}" for c in doors}
    for frac, c in sorted(zip([0.1, 0.3, 0.5, 0.7, 0.9], doors), reverse=True):   # insert back-to-front: offsets stay valid
        pos = text.rfind(". ", 0, int(len(text) * frac)) + 2
        text = text[:pos] + f"The secret code for the {c} door is {codes[c]}. " + text[pos:]
    q = ("\n\nAnswer from the text above only. List the secret code for each of the red, blue, green, yellow and purple "
         "doors, one per line as 'color: code'.")
    t0 = time.time()
    try:
        d = post({"model": MODEL, "messages": [{"role": "user", "content": text + q}], "max_tokens": 200, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}}, 1800)
        out = d["choices"][0]["message"].get("content") or ""
        hit = [c for c in doors if re.search(rf"{c}\W+{codes[c]}\b", out, re.I)]
        return {"tokens": n_tokens, "prompt_tokens": d.get("usage", {}).get("prompt_tokens"), "found": len(hit), "of": 5,
                "missed": [c for c in doors if c not in hit], "answer": out.strip()[:300], "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"tokens": n_tokens, "error": f"{type(e).__name__}: {e}", "found": 0, "of": 5, "wall_s": round(time.time() - t0, 1)}


TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city", "unit"]}}}
TOOL_MSG = [{"role": "user", "content": "What is the weather in Paris right now, in celsius? Use the tool."}]


def tool_ok(name, args_text):
    try:
        a = json.loads(args_text or "")
    except Exception:
        return False, f"args not JSON: {str(args_text)[:80]!r}"
    good = name == "get_weather" and "paris" in str(a.get("city", "")).lower() and a.get("unit") == "celsius"
    return good, f"{name} {a}"


def tool_probe(stream):
    body = {"model": MODEL, "messages": TOOL_MSG, "tools": [TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 1024}
    try:
        if not stream:
            m = post(body, 600)["choices"][0]["message"]
            tc = (m.get("tool_calls") or [None])[0]
            if not tc:
                return {"stream": False, "pass": False, "detail": f"no tool_calls; content={str(m.get('content'))[:80]!r}"}
            ok, det = tool_ok(tc["function"]["name"], tc["function"].get("arguments"))
            return {"stream": False, "pass": ok, "detail": det}
        body.update(stream=True)
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        name, args = None, ""
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                if not raw.startswith(b"data:") or raw[5:].strip() == b"[DONE]":
                    continue
                ch = json.loads(raw[5:]).get("choices") or [{}]
                for tc in ch[0].get("delta", {}).get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or name
                    args += fn.get("arguments") or ""
        ok, det = tool_ok(name, args)
        return {"stream": True, "pass": ok, "detail": det}
    except Exception as e:
        return {"stream": stream, "pass": False, "detail": f"error {type(e).__name__}: {e}"}


def spec_counters():
    try:
        t = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return None
    acc = drafts = 0.0
    for line in t.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts += float(line.rsplit(" ", 1)[1])
    return acc, drafts


t_start = time.time()
res = {"tag": TAG, "base": BASE, "model": MODEL}
if not TOOLS_ONLY:
    c0 = spec_counters()
    with ThreadPoolExecutor(3) as ex:
        aime_rows = list(ex.map(lambda a: aime(*a), [(fx, s) for fx in TRUTH for s in SEEDS]))
    c1 = spec_counters()
    tau = round(1 + (c1[0] - c0[0]) / (c1[1] - c0[1]), 3) if c0 and c1 and c1[1] > c0[1] else None
    needle_rows = [needle(64000, 7), needle(120000, 11)]
    res.update(aime=aime_rows, needle=needle_rows, aime_correct=sum(r["correct"] for r in aime_rows),
               aime_total=len(aime_rows), aime_errors=sum("error" in r for r in aime_rows),
               aime_finish_length=sum(r.get("finish") == "length" for r in aime_rows), aime_tau=tau,
               needle_found=[r["found"] for r in needle_rows])
res["tools"] = [tool_probe(False), tool_probe(False), tool_probe(True), tool_probe(True)]
res["tools_pass"] = sum(t["pass"] for t in res["tools"])
res["wall_s"] = round(time.time() - t_start, 1)
gross = res["tools_pass"] < 4 or (not TOOLS_ONLY and (res["aime_errors"] > 0 or min(res["needle_found"]) < 5
                                                      or any("error" in r for r in res["needle"])))
res["gross_failure"] = gross
out = ROOT / "results" / "stage0" / f"quality-{TAG}.json"
out.write_text(json.dumps(res, indent=1))
summary = f"tools {res['tools_pass']}/4"
if not TOOLS_ONLY:
    summary = (f"AIME {res['aime_correct']}/{res['aime_total']} (errors {res['aime_errors']}, finish=length "
               f"{res['aime_finish_length']}, tau {res['aime_tau']}) | needle 64K {res['needle_found'][0]}/5, "
               f"120K {res['needle_found'][1]}/5 | " + summary)
print(f"{TAG}: {summary} | gross_failure={gross} | {res['wall_s']} s -> {out}")
sys.exit(1 if gross else 0)
