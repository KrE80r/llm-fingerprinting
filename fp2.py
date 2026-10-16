"""fp2: exact-model verification for Claude 5/5.5 and the GPT-6/5.6 family. No timing is used (A003); billed thinking-token counts are provider accounting, not a weights-level signal.
"""
import base64, collections, itertools, json, math, os, random, statistics as st, sys, time, urllib.request, urllib.error
import concurrent.futures as cf

OR_MODEL = {"opus-5.5": "anthropic/claude-opus-5.5", "sonnet-5.5": "anthropic/claude-sonnet-5.5",
            "opus-5": "anthropic/claude-opus-5", "sonnet-5": "anthropic/claude-sonnet-5",
            "haiku-5.5": "anthropic/claude-haiku-5.5", "opus-4.8": "anthropic/claude-opus-4.8"}
AS_MODEL = {"opus-5.5": "claude-opus-5-5", "sonnet-5.5": "claude-sonnet-5-5", "opus-5": "claude-opus-5",
            "sonnet-5": "claude-sonnet-5", "haiku-5.5": "claude-haiku-5-5"}
RESELLER_R = os.environ.get("FPX_RESELLER_R_PEER", "reseller-r-peer-id-redacted-000000000000")  # real peer id kept private
# GPT family: same short name on both paths; OR prefixes "openai/". Reseller R serves these via its
# `openai-responses` plugin (a ChatGPT/Codex backend: requires stream + store:false + instructions).
GPT = ["gpt-6.1-sol", "gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra"]


_KEY = None


def _key():
    # the inherited env var is stale on this machine; ~/.zshrc holds the live one
    global _KEY
    if _KEY is None:
        import subprocess
        _KEY = subprocess.run(["zsh", "-ic", 'printf %s "$OPENROUTER_API_KEY"'], capture_output=True, text=True).stdout
    return _KEY

def or_usage():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"authorization": "Bearer " + _key()})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"]


def request(path, model, body):
    """Build (url, headers, body) for a path; `model` is a short name from OR_MODEL / GPT."""
    body = dict(body)
    if path.startswith("url:"):  # generic endpoint: url:<base, e.g. https://host/v1>; key from $FPX_KEY; native model ids
        base = path[4:].rstrip("/")
        k = os.environ.get("FPX_KEY", "")
        h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if k: h.update({"authorization": "Bearer " + k, "x-api-key": k})
        if model in GPT:
            body["model"] = model
            body.setdefault("store", False); body.setdefault("stream", True); body.setdefault("instructions", "Be helpful.")
            return base + "/responses", h, body
        body["model"] = AS_MODEL.get(model, model)
        return base + "/messages", h, body
    if path.startswith("px:"):  # local test proxy (trials/G011): model sent as the short name it claims
        body["model"] = model
        if model in GPT: body.setdefault("store", False); body.setdefault("stream", True); body.setdefault("instructions", "Be helpful.")
        return f"http://127.0.0.1:{path[3:]}/v1/" + ("responses" if model in GPT else "messages"), \
            {"content-type": "application/json", "anthropic-version": "2023-06-01"}, body
    if model in GPT:  # OpenAI Responses API on both paths
        body["model"] = model
        body.setdefault("store", False); body.setdefault("stream", True); body.setdefault("instructions", "Be helpful.")
        h = {"content-type": "application/json"}
        if path == "or":
            body["model"] = "openai/" + model
            body.setdefault("provider", {"order": ["openai"], "allow_fallbacks": False})
            h["authorization"] = "Bearer " + _key()
            return "https://openrouter.ai/api/v1/responses", h, body
        if path.startswith("as:"):
            h["x-antseed-pin-peer"] = path[3:]
            return "http://127.0.0.1:8377/v1/responses", h, body
        raise ValueError(path)
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if path == "or":
        body["model"] = OR_MODEL[model]
        body.setdefault("provider", {"order": ["anthropic"], "allow_fallbacks": False})
        h["authorization"] = "Bearer " + _key()
        url = "https://openrouter.ai/api/v1/messages"
    elif path.startswith("as:"):
        body["model"] = AS_MODEL[model]
        h["x-antseed-pin-peer"] = path[3:]
        url = "http://127.0.0.1:8377/v1/messages"
    else:
        raise ValueError(path)
    return url, h, body


def parse_sse(chunks):
    """chunks: [(t_ns, bytes)] -> [(t_ns_of_completing_chunk, event, data_dict)]. Coalescing-proof:
    events are defined by the byte stream, not by chunk boundaries."""
    buf, out = b"", []
    for t, b in chunks:
        buf += b
        while b"\n\n" in buf:
            raw, buf = buf.split(b"\n\n", 1)
            ev, data = None, []
            for line in raw.decode("utf-8", "replace").splitlines():
                if line.startswith("event:"): ev = line[6:].strip()
                elif line.startswith("data:"): data.append(line[5:].strip())
            d = "\n".join(data)
            try: d = json.loads(d)
            except Exception: pass
            if ev or data: out.append((t, ev, d))
    return out


def call(path, model, body, timeout=300):
    """Returns dict: status, headers, t0_ns, chunks [(t_ns, b64)], events, text, thinking, usage, error."""
    url, h, b = request(path, model, body)
    req = urllib.request.Request(url, json.dumps(b).encode(), h)
    rec = {"path": path, "model": model, "body": {k: v for k, v in b.items() if k != "messages"}}
    t0 = time.perf_counter_ns(); rec["t0_ns"] = t0; rec["wall"] = time.time()
    chunks = []
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        rec["status"], rec["headers"] = r.status, dict(r.headers)
        rec["t_hdr_ns"] = time.perf_counter_ns() - t0
        while True:
            c = r.read1(65536)
            if not c: break
            chunks.append((time.perf_counter_ns() - t0, c))
    except urllib.error.HTTPError as e:
        rec["status"], rec["headers"] = e.code, dict(e.headers)
        rec["error"] = e.read().decode("utf-8", "replace")
    except Exception as e:
        rec["status"], rec["error"] = -1, repr(e)[:300]
    rec["chunks"] = [(t, base64.b64encode(c).decode()) for t, c in chunks]
    raw = b"".join(c for _, c in chunks)
    if chunks and not b.get("stream"):
        try: rec["json"] = json.loads(raw)
        except Exception: rec["error"] = raw.decode("utf-8", "replace")[:2000]
    if b.get("stream"):
        evs = parse_sse(chunks)
        rec["events"] = [(t, e, d) for t, e, d in evs]
        summarize(rec, evs)
    elif "json" in rec:
        j = rec["json"]
        rec["usage"], rec["id"] = j.get("usage"), j.get("id")
        rec["text"] = "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
        rec["thinking_blocks"] = [c for c in j.get("content", []) if c.get("type") in ("thinking", "redacted_thinking")]
    return rec


def summarize(rec, evs):
    text, think, sig, usage = [], [], [], {}
    for t, e, d in evs:
        if not isinstance(d, dict): continue
        if d.get("type") == "message_start":
            m = d["message"]; rec["id"] = m.get("id"); usage.update(m.get("usage") or {})
        elif d.get("type") == "content_block_delta":
            dl = d["delta"]
            if dl.get("type") == "text_delta": text.append(dl["text"])
            elif dl.get("type") == "thinking_delta": think.append(dl["thinking"])
            elif dl.get("type") == "signature_delta": sig.append(dl["signature"])
        elif d.get("type") == "message_delta":
            usage.update(d.get("usage") or {}); rec["stop_reason"] = d.get("delta", {}).get("stop_reason")
        elif d.get("type") == "error":
            rec["error"] = json.dumps(d)
        # OpenAI Responses API events
        elif d.get("type") == "response.created":
            rec["id"] = d["response"].get("id"); rec["created"] = d["response"]
        elif d.get("type") == "response.output_text.delta":
            text.append(d["delta"])
        elif d.get("type") == "response.output_item.done" and d["item"].get("type") == "reasoning":
            sig.append(d["item"].get("encrypted_content") or ""); rec.setdefault("reasoning_items", []).append(d["item"])
        elif d.get("type") in ("response.completed", "response.incomplete", "response.failed"):
            resp = d["response"]; rec["final"] = resp
            u = resp.get("usage") or {}
            usage.update(u)
            # normalise to the Anthropic-shaped fields the behavior layer reads
            usage["output_tokens_details"] = {"thinking_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
            if resp.get("error"): rec["error"] = json.dumps(resp["error"])
    rec["text"], rec["thinking"], rec["signature"], rec["usage"] = "".join(text), "".join(think), "".join(sig), usage


# ---------------------------------------------------------------- gates (label layer)
OK = [{"role": "user", "content": "Reply with: ok"}]
GATES = {
    "adaptive": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
    "think_off": {"thinking": {"type": "disabled"}},
    "between_tools": {"thinking": {"type": "between_tools"}},
    "speed_fast": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}, "speed": "fast"},
}


def gates(path, model):
    def one(g):
        r = call(path, model, {"max_tokens": 64, "messages": OK, **GATES[g]}, timeout=120)
        return g, {"status": r["status"], "error": (r.get("error") or "")[:400], "usage": r.get("usage")}
    with cf.ThreadPoolExecutor(3) as ex:   # + 3 rate streams in identify() = 6 concurrent max
        return dict(ex.map(one, GATES))


def gate_label(g):
    """Decision list from T001/T004 (identical on OR and native Anthropic paths)."""
    inv = lambda x: x["status"] == 400 and "invalid_request" in x["error"]
    if g["between_tools"]["status"] == 200: return "sonnet-5.5"
    if inv(g["think_off"]): return "opus-5.5" if inv(g["between_tools"]) else None
    if g["think_off"]["status"] != 200 or g["adaptive"]["status"] != 200: return None
    # adaptive thinking costs +2 input tokens on 5.5-generation models, 0 on 5-generation
    if g["adaptive"]["usage"]["input_tokens"] > g["think_off"]["usage"]["input_tokens"]: return "haiku-5.5"
    speed = (g["speed_fast"]["usage"] or {}).get("speed") if g["speed_fast"]["status"] == 200 else None
    return {"fast": "opus-5", "standard": "sonnet-5"}.get(speed, "opus-5|sonnet-5")


# ---------------------------------------------------------------- habit (provider-accounting signal)
HABIT_PROMPT = "Write the integers from 1 to 150 as English words, separated by commas, nothing else."
# T005 OR medians: thinking tokens on HABIT_PROMPT at effort low. Only the thinks / doesn't-think bucket is used.
THINK = {"opus-5.5": 0, "sonnet-5.5": 0, "opus-5": 233, "sonnet-5": 215, "haiku-5.5": 366}
THINK_MIN = 20       # thinking-token count separating "does not think" from "thinks" on HABIT_PROMPT


def habit_probe(path, model):
    return call(path, model, {"max_tokens": 3000, "stream": True, "thinking": {"type": "adaptive"},
                              "output_config": {"effort": "low"},
                              "messages": [{"role": "user", "content": HABIT_PROMPT}]}, timeout=180)


def thinking_tokens(r):
    if r.get("status") != 200: return None
    u = r.get("usage") or {}
    return (u.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0


# ---------------------------------------------------------------- GPT family (G001-G006)
# effort=none is rejected only by gpt-6.1-sol and gpt-6-astra (OR path). Those two are observationally
# identical on every channel we tested (G001-G006), so they form one class.
GPT_TWIN = "gpt-6.1-sol|gpt-6-astra"
# G009: a model "thinks" only if EVERY run used >GPT_THINK_MIN reasoning tokens (gpt-6-sol thinks on ~21% of runs).
GPT_THINK = {"gpt-6-sol": 0, "gpt-6-luna": 34, "gpt-5.6-terra": 55, "gpt-5.6-sol": 27, GPT_TWIN: 0}
GPT_THINK_MIN = 10
GPT_THINKERS = "|".join(m for m, t in GPT_THINK.items() if t > GPT_THINK_MIN)   # cannot be split without timing


def gpt_gates(path, model):
    r = call(path, model, {"input": OK, "reasoning": {"effort": "none"}}, timeout=120)
    return {"effort_none": {"status": r["status"], "error": (r.get("error") or "")[:400]}}


def gpt_gate_label(g):
    x = g["effort_none"]
    if x["status"] == 400 and "mandatory" in x["error"]: return GPT_TWIN
    return "not-twin" if x["status"] == 200 else None


def gpt_habit_probe(path, model):
    return call(path, model, {"input": [{"role": "user", "content": HABIT_PROMPT}], "reasoning": {"effort": "low"}}, timeout=240)


# ---------------------------------------------------------------- identify / validate / score
def resolve(path):
    return "as:" + RESELLER_R if path == "resr" else path


def identify(path, model, k=3):
    path = resolve(path)
    gpt = model in GPT
    with cf.ThreadPoolExecutor(3) as ex:
        fut = [ex.submit(gpt_habit_probe if gpt else habit_probe, path, model) for _ in range(k)]
        g = (gpt_gates if gpt else gates)(path, model)
        streams = [f.result() for f in fut]
    return {"path": path, "claimed": model, "gates": g, "streams": streams, **verdict(g, streams)}


def verdict(g, streams):
    """Gate label refined by the thinking habit. Labels may be classes ("a|b") when nothing timing-free splits them."""
    think = [t for t in (thinking_tokens(s) for s in streams) if t is not None]
    gpt = "effort_none" in g
    if not think:
        gl = gpt_gate_label(g) if gpt else gate_label(g)
        return {"gate_label": gl, "habit": None, "consistent": False, "label": None if gl == "not-twin" else gl}
    if gpt:
        thinks = min(think) > GPT_THINK_MIN
        gl = gpt_gate_label(g)
        if gl == "not-twin": label = GPT_THINKERS if thinks else "gpt-6-sol"   # gate + habit jointly narrow the class
        elif gl == GPT_TWIN: label = gl if not thinks else f"MISMATCH(gates={gl}, habit=thinks)"
        else: label = None
    else:
        thinks = st.median(think) > THINK_MIN
        gl = gate_label(g)
        expect = {m: t > THINK_MIN for m, t in THINK.items()}
        members = gl.split("|") if gl else []
        fit = [m for m in members if expect[m] == thinks]
        label = ("|".join(fit) if fit else f"MISMATCH(gates={gl}, habit={'thinks' if thinks else 'no-think'})") if gl else None
    habit = {"thinking_tokens": think, "thinks": thinks}
    return {"gate_label": gl, "habit": habit, "label": label,
            "consistent": bool(label) and not label.startswith("MISMATCH")}


def validate(out, tag, targets):
    with open(out, "a") as f:
        for t in targets:
            p, m = t.split(":", 1) if not t.startswith("as:") else (t.rsplit(":", 1)[0], t.rsplit(":", 1)[1])
            r = identify(p, m)
            r.update(session=tag, wall=time.time())
            f.write(json.dumps(r) + "\n"); f.flush()
            print(tag, t, "->", r["label"], r["habit"], flush=True)


def truth(r):
    """Ground-truth class: the two observationally identical GPT models are scored as one class."""
    return GPT_TWIN if r["claimed"] in GPT_TWIN.split("|") else r["claimed"]


def score(files):
    rows = [json.loads(l) for fn in files for l in open(fn)]
    labels = sorted({truth(r) for r in rows}, key=lambda x: list(THINK).index(x) if x in THINK else 99)
    def matrix(name, pred):
        cm, ok = collections.Counter(), 0
        for r in rows:
            p = pred(r); cm[(truth(r), p)] += 1; ok += p == truth(r)
        preds = sorted({p for _, p in cm} - set(labels), key=str)
        print(f"\n{name}: exact accuracy {ok}/{len(rows)} = {ok / len(rows):.1%}")
        print("true \\ pred".ljust(14) + "".join(str(p)[:13].rjust(14) for p in labels + preds))
        for t in labels:
            print(t[:13].ljust(14) + "".join(str(cm[(t, p)] or ".").rjust(14) for p in labels + preds))
    matrix("gates", lambda r: verdict(r["gates"], r["streams"])["gate_label"])
    matrix("gates + habit", lambda r: verdict(r["gates"], r["streams"])["label"])
    wrong = sum(1 for r in rows if (lab := verdict(r["gates"], r["streams"])["label"]) and not lab.startswith("MISMATCH")
                and truth(r) not in lab.split("|") and lab != truth(r))
    print(f"\nlabels that exclude the true model: {wrong}/{len(rows)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "identify":
        r = identify(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
        print(json.dumps({k: r[k] for k in ("path", "claimed", "gate_label", "habit", "consistent", "label")}, indent=1))
    elif cmd == "validate":
        validate(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "score":
        score(sys.argv[2:])
    else:
        print(__doc__)
