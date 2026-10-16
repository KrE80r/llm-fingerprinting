"""fpx: verify a claimed model, or identify an unknown one, behind a chat API endpoint.

Layers (see docs/PAPER.md):
  L1 artifact forensics   reasoning-artifact format/structure (Fernet tokens, signatures, plain text)
  L2 tokenizer accounting usage.prompt_tokens deltas on 8 probe strings (template overhead cancels)
  L2b token-ID namespace  sparse logit_bias acceptance + forced/forbidden-token controls (optional)
  L3 serving stack        chat-template Jacobian, tool-call id grammar, raw upstream error, finish reason, fingerprint
  L4 reasoning habit      reasoning tokens spent on a fixed prompt (fp2; decode speed was removed in A003)
  L5 capability gates     per-model API feature gates (Claude, GPT; fp2.py)
  L6 cache-transfer oracle  prime label A, read cached tokens under label B on the same endpoint
A label is reported only when the layers agree; otherwise MISMATCH / unknown / insufficient_data.

Endpoint styles:
  OpenAI chat-completions  identify / verify-ref / probe   (OpenRouter by default, or --base URL --key-env VAR)
  Anthropic messages, OpenAI responses  verify / cache-oracle
      path: or | resr | as:<peerId> | px:<port> | url:<base-url>  (url: uses $FPX_KEY and native model ids)
usage:
  python3 fpx.py identify     <refs.json> <model-id> [--base URL] [--key-env VAR] [--provider NAME]   # unknown/stealth
  python3 fpx.py token-id     <token-id-refs.json> <model-id> [--base URL] [--key-env VAR] [--provider NAME]
  python3 fpx.py verify-ref   <refs.json> <model-id> <claimed-ref> [--base URL] [--key-env VAR]
  python3 fpx.py verify       <path> <claimed>       # Claude/GPT exact model via gates + reasoning habit (7 req)
  python3 fpx.py cache-oracle <path> <claimed> <alt> # substitution check: does label <alt> share <claimed>'s cache?
  python3 fpx.py audit        <path> <claimed> <alt>... [--out f.jsonl]  # STACKED verdict: gates+habit+replay+cache
  python3 fpx.py probe        <out.jsonl> <tag> <model-id> [...]   # collect reference/held-out probes
  python3 fpx.py build        <refs.json> <probe.jsonl>...         # freeze reference DB
  python3 fpx.py evaluate     <refs.json> <probe.jsonl>... [--layers tok,stack,art]
"""
import base64, collections, concurrent.futures as cf, itertools, json, math, os, random, re, statistics as st, subprocess, sys, time
import fp2

PROBES = {  # tokenizer probes; feature = prompt_tokens(p) - prompt_tokens("ok")
    "digits": "12345678901234567890 31415926535897932384",
    "zh": "我们今天去北京大学看看图书馆的历史和文化",
    "code": "def f(x):\n    return [i**2 for i in range(x) if i % 3 == 0]\n",
    "rare": "🐟🦀🧬 ﷽ ‽ ∮ ⊕ ℵ₀ 𝔉𝔯𝔞𝔨𝔱𝔲𝔯",
    "ru": "Съешь же ещё этих мягких французских булок, да выпей чаю",
    "ws": "x          \n\n\t\t      \n    y",
    "ar": "السلام عليكم ورحمة الله وبركاته يا أصدقائي",
    "url": "https://example.com/api/v2/items?id=42&sort=desc#section-3",
}
REASON_Q = "Is 391 prime? Think it through, then answer yes or no."



def endpoint(base, key_env):
    if base is None:
        return "https://openrouter.ai/api/v1/chat/completions", fp2._key()
    base = base.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    return url, os.environ.get(key_env or "", "") if key_env else ""


def chat(*a, **kw):
    """G4: retry rate limits and server errors (twice, 3 s then 9 s). They say nothing about the model (A001)."""
    for wait in (3, 9, None):
        rec = _chat(*a, **kw)
        st = rec.get("status") or 0
        if wait is None or not (st == 429 or st >= 500): return rec
        time.sleep(wait)


def _chat(url, key, model, messages, stream=False, max_tokens=16, reasoning=None, provider=None, timeout=180, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream, **(extra or {})}
    if reasoning: body["reasoning"] = reasoning
    if provider: body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if stream: body["stream_options"] = {"include_usage": True}
    h = {"content-type": "application/json"}
    if key: h["authorization"] = "Bearer " + key
    import urllib.request, urllib.error
    rec, chunks, t0 = {"model": model}, [], time.perf_counter_ns()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h), timeout=timeout)
        rec["status"], rec["headers"] = r.status, dict(r.headers)
        while (c := r.read1(65536)): chunks.append((time.perf_counter_ns() - t0, c))
    except urllib.error.HTTPError as e:
        rec["status"], rec["error"] = e.code, e.read().decode("utf-8", "replace")[:8000]
    except Exception as e:
        rec["status"], rec["error"] = -1, repr(e)[:300]
    raw = b"".join(c for _, c in chunks)
    if not stream:
        try: rec["json"] = json.loads(raw)
        except Exception: rec.setdefault("error", raw.decode("utf-8", "replace")[:8000])
        if "json" in rec and "error" in rec["json"]: rec["error"] = json.dumps(rec["json"]["error"])[:8000]
    else:
        rec["events"] = [(t, e, d) for t, e, d in fp2.parse_sse(chunks)]
    return rec


# ---------------------------------------------------------------- collection
TOOL = {"tools": [{"type": "function", "function": {"name": "emit", "description": "Emit a value.", "parameters": {
            "type": "object", "properties": {"v": {"type": "string"}}, "required": ["v"]}}}],
        "tool_choice": {"type": "function", "function": {"name": "emit"}}}
BAD_SCHEMA = {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "strict": True,
              "schema": {"type": "bogus_type"}}}}
# E004: option gates found in the E002 sweep. Only the HTTP status is kept (error text drifts).
PNG_1PX = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
IMAGE_MSG = [{"role": "user", "content": [{"type": "text", "text": "Reply with: ok"}, {"type": "image_url", "image_url": {"url": PNG_1PX}}]}]
# G8: weights-level behaviour battery (temperature 0). Read as coarse categories; together ONE independent block.
BEHAV = {
    "echo": ("Repeat exactly, nothing else: Ωmega ß ﬁ ①② 𝔘 e\u0301 \u202ex\u202c", {}),
    "json": ("Reply with a JSON object with key a set to 1.", {"response_format": {"type": "json_object"}}),
    "cutoff": ("What is the most recent month and year you have knowledge of? Reply as YYYY-MM only.", {}),
    "bits": ("Return exactly eight characters, 1 for true and 0 for false, one per statement, nothing else: "
             "(1) 341 is prime. (2) 2^341 is congruent to 2 modulo 341. (3) Every finite subgroup of the multiplicative "
             "group of a field is cyclic. (4) Context-free languages are closed under intersection. (5) JSON Pointer /a~1b "
             "addresses the key a/b. (6) UTF-8 encodes U+1F600 in three bytes. (7) In SQL, NULL = NULL evaluates TRUE. "
             "(8) In Python 3, round(2.5) == 2.", {}),
    "count": ("How many times does the letter r appear in: strawberry raspberry? Reply with digits only.", {}),
    "list": ("List three primary colors.", {}),
    "pow": ("What is 7 to the power 7? Reply with digits only.", {}),
    # G11 capability panel: chosen on the G11 discovery sweep (stable, disjoint answers between unnamed siblings).
    "c_mul": ("What is 3847 * 2963? Reply with the answer only.", {}),
    "c_rev": ("Write the word 'unbelievably' backwards. Reply with the answer only.", {}),
    "c_days": ("How many days are there from 2024-02-27 to 2024-03-02, not counting the start day? Reply with the answer only.", {}),
    "c_letter": ("What is the 7th letter of the word 'parallelogram'? Reply with the answer only.", {}),
    "c_code": ("What does this Python print: print(sorted('banana', key=lambda c: -ord(c))[2]) Reply with the answer only.", {}),
    "c_sum": ("What is the sum of the digits of 2 to the power 30? Reply with the answer only.", {}),
    "c_word": ("How many vowels are in the word 'onomatopoeia'? Reply with the answer only.", {}),
    "c_hex": ("What is 0xBEEF in decimal? Reply with the answer only.", {}),
    # G13 hard panel: no answer differences between close siblings (all correct), kept for its reasoning lengths (G14).
    **{k: (q + " Reply with the answer only.", {}) for k, q in {
        "h_mul": "What is 987654 * 123457?", "h_seven": "How many times does the digit 7 appear when writing all integers from 1 to 1000?",
        "h_prime": "What is the 1000th prime number?", "h_dow": "What day of the week was 1999-12-31?",
        "h_mod": "What is 17 to the power 5, modulo 1000?", "h_perm": "How many distinct arrangements are there of the letters of BALLOON?",
        "h_rev": "Reverse the string 'Supercalifragilistic' character by character.", "h_sqrt": "What is the square root of 2 to 12 decimal places?",
        "h_s": "How many times does the letter s (upper or lower case) appear in: She sells sea shells by the sea shore",
        "h_skip": "Write 'mississippi' keeping only the 1st, 3rd, 5th, 7th, 9th and 11th letters."}.items()},
    # G15 knowledge panel: newest version the weights know of (training cutoff). Capability block. Of 13 items only
    # this one was stable within models and split sibling releases; the rest wobbled (reasoning at temperature 0).
    # (k_k8s looked stable for GLM-5 in discovery but wobbled 1.31-1.33 elsewhere and caused drift-wrongs: dropped.)
    "k_ios": ("What is the newest iOS major version you know of? Reply with the number only.", {}),
}


def probe(model, base=None, key_env=None, provider=None):
    url, key = endpoint(base, key_env)
    one = lambda text: chat(url, key, model, [{"role": "user", "content": text}], max_tokens=16, provider=provider)
    msg = lambda ms, **kw: chat(url, key, model, ms, provider=provider, **kw)
    with cf.ThreadPoolExecutor(6) as ex:
        tok = dict(zip(["ok", *PROBES], ex.map(one, ["ok", *PROBES.values()])))
        reason = ex.submit(chat, url, key, model, [{"role": "user", "content": REASON_Q}], False, 1500, {"effort": "low"}, provider)
        # A003: no streams. Decode speed tracked host, load and time, not the model, and never changed a decision.
        stack = {  # serving-stack probes (round M2): template Jacobian, tool-call serializer, raw upstream error
            "sys": ex.submit(msg, [{"role": "system", "content": "x"}, {"role": "user", "content": "ok"}]),
            "asst": ex.submit(msg, [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "ok"}]),
            "tool": ex.submit(msg, [{"role": "user", "content": "Call emit with v set to: hello world"}], max_tokens=300, extra=TOOL),
            "err": ex.submit(msg, [{"role": "user", "content": "ok"}], extra=BAD_SCHEMA),
            "image": ex.submit(msg, IMAGE_MSG),
            "effort_none": ex.submit(msg, [{"role": "user", "content": "Reply with: ok"}], reasoning={"effort": "none"}),
        }
        behav = behav_battery(ex, msg)
        reason = reason.result()
        stack = {k: v.result() for k, v in stack.items()}
        behav = {k: v.result() for k, v in behav.items()}
    return {"model": model, "tok": {k: ((v.get("json") or {}).get("usage") or {}).get("prompt_tokens") for k, v in tok.items()},
            "tok_err": {k: v.get("error") for k, v in tok.items() if v.get("status") != 200},
            "ok_json": tok["ok"].get("json"), "stack": stack, "reason": reason, "behav": behav, "wall": time.time()}

def behav_battery(ex, msg):
    return {k: ex.submit(msg, [{"role": "user", "content": q}], max_tokens=600 if k.startswith(("c_", "h_", "k_")) else 400,
                         extra={"temperature": 0, **x}) for k, (q, x) in BEHAV.items()}


def behav_sig(p):
    """G8: coarse categories of the temperature-0 battery. Missing (None) when the reply is transient or empty."""
    import unicodedata
    out = {}
    for k, r in (p.get("behav") or {}).items():
        st = r.get("status") or 0
        if st in (-1, 402, 408, 429) or st >= 500: continue
        if st != 200: out[k] = f"E{st}"; continue
        t = ((((r.get("json") or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not t: continue
        if k == "echo":
            target = BEHAV["echo"][0].split(": ", 1)[1]
            out[k] = "exact" if t == target else "normalized" if unicodedata.normalize("NFC", t) == unicodedata.normalize("NFC", target) \
                else "partial" if target.startswith(t) or t in target else "other"
        elif k == "json":
            out[k] = "fence" if t.startswith("```") else "c_sp" if t == '{"a": 1}' else "c" if t == '{"a":1}' else \
                "p2" if "\n  \"" in t else "p4" if "\n    \"" in t else "p0" if "\n\"" in t else "other"
        elif k == "cutoff":
            m = re.search(r"(19|20)\d\d", t); out[k] = m.group(0) if m else "none"   # year only: months wobble (E005)
        elif k == "bits":
            m = re.search(r"[01]{8}", t); out[k] = m.group(0) if m else "bad"
        elif k in ("count", "pow") or k.startswith(("c_", "h_")):
            # G19 (A005): the LAST number is the answer ("7^7 = 823543", "4 days" -> "4"); text answers normalised.
            nums = re.findall(r"\d+(?:\.\d+)?", t.replace(",", ""))
            out[k] = nums[-1] if nums else re.sub(r"[\s.*`'\"]", "", t).lower()[:24] if k.startswith(("c_", "h_")) else "none"
        elif k.startswith("k_"):                     # G15: the version only ("Python 3.13" == "3.13")
            m = re.search(r"\d+(\.\d+)?", t); out[k] = m.group(0) if m else "none"
        elif k == "list":
            out[k] = ("num" if re.match(r"1[.)]", t) else "dash" if t.startswith("- ") else "star" if t.startswith("* ")
                      else "dot" if t.startswith("•") else "lines" if "\n" in t else "inline") + ("+bold" if "**" in t else "")
    # G14: hidden reasoning length per question (log2 bin; 0 = no reasoning; cap = hit max_tokens). A weights habit.
    for k, r in (p.get("behav") or {}).items():
        if not r or r.get("status") != 200 or not r.get("json"): continue
        j = r["json"]
        if (j.get("choices") or [{}])[0].get("finish_reason") == "length": out["rt_" + k] = "cap"; continue
        v = ((j.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(v, int): out["rt_" + k] = str(v.bit_length())
    # G10: which reasoning each request exposes (none / encrypted / summary / text / signature-only / plain).
    # Adaptive-thinking models decide this per prompt, so it is a weights-level habit. Part of the behaviour block.
    def rshape(r):
        if not r or r.get("status") != 200 or not r.get("json"): return None
        m = ((r["json"].get("choices") or [{}])[0]).get("message") or {}
        s = {("E" if "encrypted" in (d.get("type") or "") else "S" if "summary" in (d.get("type") or "")
              else ("T" if d.get("text") else "G") if "text" in (d.get("type") or "") else "?")
             for d in m.get("reasoning_details") or []}
        return "".join(sorted(s)) or ("P" if m.get("reasoning") else "-")
    st = p.get("stack") or {}
    for k, r in [("ok", {"status": 200, "json": p.get("ok_json")}), *((k, st.get(k)) for k in ("sys", "asst", "tool")),
                 *(("b_" + k, r) for k, r in (p.get("behav") or {}).items())]:
        v = rshape(r)
        if v is not None: out["think_" + k] = v
    fin = (((p.get("ok_json") or {}).get("choices") or [{}])[0]).get("finish_reason")
    if fin: out["think_finish"] = fin
    return out


def _err_text(rec, model):
    """upstream error text: nested error.metadata.raw or error.message (G19 parser fix), model slug removed"""
    s = rec.get("error") or ""
    if not isinstance(s, str): s = json.dumps(s)    # some providers return the error as an object
    try:
        e = json.loads(s); e = e.get("error", e) if isinstance(e, dict) else {}
        txt = (e.get("metadata") or {}).get("raw") or e.get("message") or ""
    except Exception:                               # bodies cut at 600 bytes before G19: same text, escapes undone
        m = re.search(r'"raw":\s*"(.*)', s) or re.search(r'"message":\s*"(.*)', s)
        txt = (m.group(1) if m else s).replace('\\"', '"').replace("\\n", "\n")
    for name in (model, model.split("/")[-1]): txt = txt.replace(name, "<M>")   # full slug first: fixed order
    return txt


FINISH_VOCAB = {"end_turn": "anthropic", "max_tokens": "anthropic", "stop_sequence": "anthropic", "stop": "chat",
                "length": "chat", "STOP": "gemini", "MAX_TOKENS": "gemini", "completed": "responses",
                "max_output_tokens": "responses", "incomplete": "responses"}


# ---------------------------------------------------------------- features
def artifact(rec):
    """L1: structural description of the reasoning artifact. Never decrypts; reads only public framing."""
    j = rec.get("json") or {}
    msg = ((j.get("choices") or [{}])[0]).get("message") or {}
    out = {"types": [], "formats": [], "enc": None}
    for d in msg.get("reasoning_details") or []:
        out["types"].append(d.get("type")); out["formats"].append(d.get("format"))
        blob = d.get("data") or d.get("signature")
        if blob and not out["enc"]:
            try:
                raw = base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4))
            except Exception:
                try: raw = base64.b64decode(blob + "=" * (-len(blob) % 4))
                except Exception: raw = b""
            fernet = len(raw) > 57 and raw[0] == 0x80 and abs(int.from_bytes(raw[1:9], "big") - time.time()) < 10 * 365 * 86400
            out["enc"] = "fernet" if fernet else f"blob:{raw[:2].hex()}" if raw else "opaque"
    if not out["types"] and msg.get("reasoning"): out["types"] = ["reasoning.plain"]
    u = j.get("usage") or {}
    out["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return out


def lab_from_artifact(a):
    f = set(x for x in a["formats"] if x and x != "unknown")
    if a["enc"] == "fernet" or "openai-responses-v1" in f: return "openai"
    for fmt, lab in (("anthropic", "anthropic"), ("google", "google"), ("gemini", "google"), ("xai", "x-ai"), ("grok", "x-ai")):
        if any(fmt in x for x in f): return lab
    return None


def _shape(s):
    """Normalise an identifier/message to its grammar: digits->9, letters->a/A runs kept by class, length bucketed."""
    if not s: return None
    s = re.sub(r"[0-9]+", "9", str(s)); s = re.sub(r"[a-z]{2,}", "a", s); s = re.sub(r"[A-Z]{2,}", "A", s)
    return s[:60]


def stack_sig(p):
    """Serving-stack signature (round M2): categorical, timing-free."""
    st_ = p.get("stack")
    if not st_: return {}
    pt = lambda r: (((r.get("json") or {}).get("usage") or {}).get("prompt_tokens"))
    base = p["tok"].get("ok")
    sig = {}
    # A1 fix: rate limits, billing errors, server errors and transport failures are missing evidence, not a fingerprint.
    transient = lambda r: r.get("status") in (-1, 402, 408, 429) or (r.get("status") or 0) >= 500
    for k in ("sys", "asst"):
        if transient(st_[k]): continue
        v = pt(st_[k]); sig[k] = (v - base) if (v is not None and base is not None) else f"status{st_[k].get('status')}"
    tc = ((((st_["tool"].get("json") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [{}])[0]
    tid = tc.get("id") if tc else None
    if tid:
        m = re.match(r"^([A-Za-z]+[_\-:.])", tid); body = tid[len(m.group(1)):] if m else tid
        cls = "hex" if re.fullmatch(r"[0-9a-f]+", body) else "num" if body.isdigit() else "b62" if body.isalnum() else "mixed"
        sig["tool_id"] = f"{m.group(1) if m else ''}{cls}{len(body)}"
    elif not transient(st_["tool"]):
        sig["tool_id"] = f"status{st_['tool'].get('status')}"
    if not transient(st_["err"]):
        raw = _err_text(st_["err"], p["model"])
        sig["err"] = f"{st_['err'].get('status')}:" + (_shape(raw)[:40] if raw else "")
    ch = (((p.get("ok_json") or {}).get("choices") or [{}])[0])
    # G14: the finish-reason VOCABULARY is the reply adapter; WHICH reason came back (did the 16-token budget run out,
    # e.g. because the model reasoned) is behaviour and goes to the reasoning block as think_finish.
    sig["finish"] = FINISH_VOCAB.get(ch.get("native_finish_reason"), ch.get("native_finish_reason"))
    # G16: only the FORMAT of the fingerprint is stable; its hex is a per-deploy build hash (Kimi lost names in E003 and
    # G16 when Moonshot redeployed). "fpv0_08c4dda0" -> "fpv0_h8".
    fp = (p.get("ok_json") or {}).get("system_fingerprint")
    sig["sysfp"] = re.sub(r"[0-9a-f]{6,}", lambda m: f"h{len(m.group(0))}", str(fp)) if fp else None
    # E001: does the reasoning probe come back with a separate reasoning part? Categorical, no thresholds.
    r = p.get("reason") or {}
    if r.get("status") == 200 and r.get("json"):
        msg = (((r["json"].get("choices") or [{}])[0]).get("message") or {})
        sig["reasoning_channel"] = "separate" if (msg.get("reasoning") or msg.get("reasoning_details")) else "none"
    # E004: option gates, status only. Absent in probes collected before E004 (then simply not compared).
    for k in ("image", "effort_none"):
        if k in st_ and not transient(st_[k]):
            sig[k] = st_[k].get("status")
    # G4: bytes that were stored but never used (G1 byte mining). base = absolute prompt tokens of the plain "ok"
    # request: the provider's hidden prompt + chat template, which the tokenizer layer subtracts away. Stable in
    # 36/36 models over Oct 11-13. tool_keys = keys of the forced tool-call reply message (e.g. reasoning parts).
    okj = p.get("ok_json") or {}
    if (okj.get("usage") or {}).get("prompt_tokens") is not None: sig["base"] = okj["usage"]["prompt_tokens"]
    tj = st_.get("tool") or {}
    if tj.get("status") == 200 and tj.get("json"):
        sig["tool_keys"] = ",".join(sorted((((tj["json"].get("choices") or [{}])[0]).get("message") or {})))
        # G10: size of the hidden tool-use prompt (tool request tokens minus the plain baseline).
        tp = (tj["json"].get("usage") or {}).get("prompt_tokens")
        if tp is not None and sig.get("base") is not None: sig["tool_tmpl"] = tp - sig["base"]
    return sig


def features(p):
    base = p["tok"].get("ok")
    tokv = {k: (p["tok"][k] - base) if (p["tok"].get(k) is not None and base is not None) else None for k in PROBES}
    a = artifact(p["reason"])
    return {"tok": tokv, "art": a, "art_lab": lab_from_artifact(a), "stack": stack_sig(p),
            "reason_tok": a["reasoning_tokens"], "behav": behav_sig(p)}



# ------------------------------------------------------- token-ID namespace
def _choice_text(row):
    try:
        content = row["json"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def classify_token_id_rows(rows, db):
    """Match only a stable, bias-controlled token-ID acceptance vector."""
    if db.get("schema") != "fpx.token-id-reference/v1":
        raise ValueError("expected fpx.token-id-reference/v1")
    token_ids = db["protocol"]["token_ids"]
    repeats = sorted({row["repeat"] for row in rows})
    if len(repeats) < 2:
        return {"verdict": "ABSTAIN", "reason": "incomplete_repeats", "namespace": "unknown"}
    index = {(row["repeat"], row["token_id"], row["bias"]): row for row in rows}
    signatures, echoes, forced = [], [], []
    for repeat in repeats:
        positive = [index.get((repeat, token_id, 100)) for token_id in token_ids]
        negative = index.get((repeat, 32, -100))
        if any(row is None for row in positive) or negative is None:
            return {"verdict": "ABSTAIN", "reason": "incomplete_rows", "namespace": "unknown"}
        if any(row.get("status") not in (200, 400) for row in positive) or negative.get("status") not in (200, 400):
            return {"verdict": "ABSTAIN", "reason": "transport_error", "namespace": "unknown"}
        control = positive[token_ids.index(32)]
        text = _choice_text(control)
        if (control.get("status") != 200 or negative.get("status") != 200 or not text
                or _choice_text(negative) == text):
            return {"verdict": "ABSTAIN", "reason": "control_failed", "namespace": "unknown"}
        forced.append(text)
        signatures.append(tuple(row.get("status") == 200 for row in positive))
        echoes.append(tuple(_choice_text(row) if row.get("status") == 200 else "" for row in positive))
    if len(set(forced)) != 1 or len(set(signatures)) != 1 or len(set(echoes)) != 1:
        return {"verdict": "ABSTAIN", "reason": "unstable", "namespace": "unknown"}
    signature = signatures[0]
    matches = [(name, ref) for name, ref in db["namespaces"].items()
               if tuple(ref["acceptance"]) == signature]
    if not matches:
        return {"verdict": "INCOMPATIBLE_WITH_ENROLLED", "reason": "no_match",
                "namespace": "unknown", "observed_acceptance": list(signature),
                "observed_echoes": list(echoes[0])}
    if len(matches) != 1:
        return {"verdict": "ABSTAIN", "reason": "reference_collision",
                "namespace": "unknown", "observed_acceptance": list(signature)}
    namespace, ref = matches[0]
    if sum(ref["acceptance"]) < db["protocol"].get("min_accepted_ids", 2):
        return {"verdict": "ABSTAIN", "reason": "underdetermined_reference",
                "namespace": "unknown", "observed_acceptance": list(signature)}
    if ref.get("echoes") and tuple(ref["echoes"]) != echoes[0]:
        return {"verdict": "ABSTAIN", "reason": "echo_mismatch",
                "namespace": "unknown", "observed_acceptance": list(signature),
                "observed_echoes": list(echoes[0])}
    return {"verdict": "CONSISTENT_WITH", "namespace": namespace,
            "family_candidates": ref.get("family_candidates", []), "model": "unknown",
            "observed_acceptance": list(signature), "observed_echoes": list(echoes[0]),
            "scope": "API-visible token-ID namespace; not weights, provider, family, or exact-model proof"}


def load_token_id_db(path):
    db = json.load(open(path))
    if db.get("schema") != "fpx.token-id-reference/v1" or not isinstance(db.get("namespaces"), dict):
        raise ValueError(f"{path}: expected fpx.token-id-reference/v1")
    return db


def identify_token_id_namespace(refs_path, model, base=None, key_env=None, provider=None):
    """Run the N001 two-repeat target-only probe; references stay on disk."""
    db = load_token_id_db(refs_path)
    url, key = endpoint(base, key_env)
    rows = []
    for repeat in ("r1", "r2"):
        for token_id, bias in [(token_id, 100) for token_id in db["protocol"]["token_ids"]] + [(32, -100)]:
            row = chat(url, key, model, [{"role": "user", "content": "x"}], max_tokens=1,
                       provider=provider,
                       extra={"temperature": 0, "logit_bias": {str(token_id): bias}})
            row.update({"repeat": repeat, "token_id": token_id, "bias": bias})
            rows.append(row)
    return {"result": classify_token_id_rows(rows, db), "observations": rows}


# ---------------------------------------------------------------- reference DB + classifier
FAMILY_BY_LAB = {
    "anthropic": "claude", "openai": "gpt", "google": "gemini", "x-ai": "grok",
    "z-ai": "glm", "thudm": "glm", "qwen": "qwen", "mistralai": "mistral",
    "deepseek": "deepseek", "meta-llama": "llama", "meta": "muse", "minimax": "minimax",
    "xiaomi": "mimo", "moonshotai": "kimi", "cohere": "command", "microsoft": "phi",
}
MAX_FAMILY_DISTANCE = 2.0
MAX_MODEL_DISTANCE = 0.5
# (A003) MIN_MODEL_MARGIN was dropped: without speed the score is pure structure, so the margin equals the lead.
MIN_STRUCTURAL_LEAD = 2.0   # R2 (A001): survives any single categorical field drifting toward a sibling
MIN_FAMILY_GAP = 1.0        # A001: nearest other family must be this much further (no-artifact family claims)
STRONG_CAPABILITY = 4        # G12: capability evidence this strong counts as two blocks
BEHAV_MIN_ITEMS = 1          # G18: one disjoint item per block (items need MIN_ITEM_OBS sessions); was 2 until G16
STACK_FIELDS = {"sys", "asst", "tool_id", "err", "finish", "sysfp", "reasoning_channel"}   # a reference needs all of them


def lab_of(model_id): return model_id.split("/")[0]
def family_of(model_id, lab=None): return FAMILY_BY_LAB.get(lab or lab_of(model_id), "unknown")


def make_reference_db(refs):
    return {
        "schema": "fpx.reference-db/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "id": "fpx-stack-v1",
            "tokenizer_probes": PROBES,
            "layers": ["tokenizer", "serving_stack", "reasoning_artifact"],
            "requests_per_identification": 14,
        },
        "calibration": {
            "policy": "open-set abstention; thresholds require fresh held-out validation",
            "max_family_distance": MAX_FAMILY_DISTANCE,
            "max_model_distance": MAX_MODEL_DISTANCE,
            "decode_speed": "removed (A003)",
            "min_structural_lead": MIN_STRUCTURAL_LEAD,
            "min_family_gap": MIN_FAMILY_GAP,
        },
        "models": refs,
    }


def load_reference_db(path):
    db = json.load(open(path))
    if db.get("schema") != "fpx.reference-db/v1" or not isinstance(db.get("models"), dict):
        raise ValueError(f"{path}: expected fpx.reference-db/v1")
    return db["models"]


def usable_ref(ref):
    return any(sum(v is not None for v in p["tok"].values()) >= len(PROBES) // 2 for p in ref["probes"])


def build(refs_path, files):
    """Reference DB keeps every reference probe because host variants can shift template token counts."""
    refs = collections.defaultdict(lambda: {"lab": None, "family": None, "probes": [], "provenance": []})
    seen, dropped, extra, samples = set(), 0, [], []
    for fn in files:
        for l in open(fn):
            p = json.loads(l)
            if p.get("behav_only"):                       # G14: behaviour-only reference session (no structural fields)
                extra.append(p); continue
            if "task" in p and "set" in p:                # G17: one-word sample row (grouped below)
                samples.append(p); continue
            f = features(p); seen.add(p["model"])
            # Count by name, not by number: a new field must not make an incomplete probe look complete (E001).
            # A001: a reference probe must be complete. Probes hit by 429/402/5xx would match too easily on fewer fields.
            if any(v is None for v in f["tok"].values()) or not STACK_FIELDS <= set(f["stack"]):
                dropped += 1; continue
            r = refs[p["model"]]
            r["lab"], r["family"] = lab_of(p["model"]), family_of(p["model"])
            r["probes"].append({k: f[k] for k in ("tok", "art_lab", "reason_tok", "stack", "behav")})
            r["provenance"].append({"source": os.path.relpath(fn), "session": p.get("session"),
                                    "provider": p.get("provider_pin"), "collected_at_unix": p.get("wall")})
    for p in extra:
        if p["model"] in refs: refs[p["model"]].setdefault("behav_extra", []).append(behav_sig(p))
    for (m, s), d in group_samples(samples).items():
        if m in refs: refs[m].setdefault("dist", []).append(d)
    skipped = sorted(seen - set(refs))
    json.dump(make_reference_db(refs), open(refs_path, "w"), indent=1)
    print(f"froze {len(refs)} references ({sum(len(r['probes']) for r in refs.values())} probes); dropped {dropped} "
          f"incomplete probes; not enrolled (no complete probe): {', '.join(skipped) or 'none'} -> {refs_path}")
    collision_audit(refs)


def collision_audit(refs):
    """A001: list reference pairs too close for an exact name (lead < MIN_STRUCTURAL_LEAD) or cross-family pairs inside
    the family bound. Classification abstains on these by rule; the audit makes the blind spots explicit at build time."""
    ok = lambda p: sum(v is not None for v in p["tok"].values()) >= len(PROBES) // 2
    d = lambda a, b: tok_dist(a["tok"], b["tok"]) + stack_mismatch(a.get("stack") or {}, b.get("stack") or {})
    for (m1, r1), (m2, r2) in itertools.combinations(sorted(refs.items()), 2):
        ps1, ps2 = [p for p in r1["probes"] if ok(p)], [p for p in r2["probes"] if ok(p)]
        if not ps1 or not ps2: continue
        x = min(d(a, b) for a in ps1 for b in ps2)
        if x < MIN_STRUCTURAL_LEAD:
            print(f"  collision audit: {m1} ~ {m2} structure {x:.2f} -> never named apart")
        elif r1["family"] != r2["family"] and x <= MAX_FAMILY_DISTANCE:
            print(f"  collision audit: cross-family {m1} ~ {m2} structure {x:.2f} -> family needs the gap rule")


def tok_dist(a, b):
    ks = [k for k in PROBES if a.get(k) is not None and b.get(k) is not None]
    return sum(abs(a[k] - b[k]) for k in ks) / max(len(ks), 1) if ks else math.inf


def stack_mismatch(a, b):
    ks = [k for k in a if k in b]
    return sum(a[k] != b[k] for k in ks) if ks else 0


# G4 (forge review): fields from one causal source drift together, so the exact-name lead is counted in blocks.
BLOCK_OF = {"base": "template", "sys": "template", "asst": "template", "tool_id": "tool", "tool_keys": "tool", "tool_tmpl": "tool",
            "reasoning_channel": "reply_adapter", "finish": "reply_adapter", "sysfp": "reply_adapter",
            "err": "validator", "image": "validator", "effort_none": "validator"}


MIN_ITEM_OBS = 3    # G18 (was 2): G14: "never answers that way" needs at least two reference sessions answering the item
_UNION = {}


def behav_union(ref):
    """G10: every behaviour answer this model gave in any reference session (absorbs temperature-0 wobble).
    G14: items answered in fewer than MIN_ITEM_OBS reference sessions are left out (one session shows no wobble)."""
    hit = _UNION.get(id(ref))
    if hit and hit[0] is ref: return hit[1]
    u, n = collections.defaultdict(set), collections.Counter()
    for b in behav_sessions(ref):
        for k, v in b.items(): u[k].add(v); n[k] += 1
    u = {k: v for k, v in u.items() if n[k] >= MIN_ITEM_OBS}
    _UNION[id(ref)] = (ref, u)
    return u


def disjoint(k, a, b):
    """Reference answer sets that never meet. Reasoning lengths wobble by one bin, so their ranges need an empty bin
    between them (G14)."""
    if a & b: return False
    if not k.startswith("rt_"): return True
    r = lambda s: [99 if v == "cap" else int(v) for v in s]
    return min(r(a)) >= max(r(b)) + 2 or min(r(b)) >= max(r(a)) + 2


def block_dist(f, ref):
    """Number of independent structural blocks that differ (tokenizer counts as one block), nearest reference probe."""
    def one(p):
        a, b = f.get("stack") or {}, p.get("stack") or {}
        blocks = {BLOCK_OF.get(k, k) for k in a if k in b and a[k] != b[k]}
        t = tok_dist(f["tok"], p["tok"])
        return len(blocks) + (1 if (math.isinf(t) or t >= 0.5) else 0)
    return min(one(p) for p in ref["probes"])


CAPABILITY_ITEMS = ("c_", "h_", "k_", "count", "pow", "bits", "cutoff")   # answers decided by the weights (right/wrong, knowledge)
REASONING_ITEMS = ("rt_", "think_")    # G14: whether and how long the model reasons on each question (one habit)
MIXED_REASONING = 2                    # G14: >= this many questions longer AND shorter than the sibling: two blocks


def behav_group(k):
    return "cap" if k.startswith(CAPABILITY_ITEMS) else "reason" if k.startswith(REASONING_ITEMS) else "style"


def behav_sessions(ref):
    return [p.get("behav") or {} for p in ref["probes"]] + (ref.get("behav_extra") or [])


def behav_split(f, best_ref, other_ref):
    """G12/G14: behaviour evidence for best over other, in up to three blocks: capability (right/wrong answers),
    reasoning (think shapes + reasoning lengths) and style (formats, refusals; a hidden prompt can move these).
    Only items whose reference answer sets are disjoint count. Per block: hits_best - hits_other >= BEHAV_MIN_ITEMS
    and hits_other <= 1 gives one block. Two blocks: capability with >= STRONG_CAPABILITY items (only a weights change
    can do that), or reasoning lengths that are longer than the other model's on >= MIXED_REASONING questions AND
    shorter on >= MIXED_REASONING others (one effort knob moves every length the same way, so it cannot fake that)."""
    ub, uo, fb = behav_union(best_ref), behav_union(other_ref), f.get("behav") or {}
    h = {g: [0, 0] for g in ("cap", "reason", "style")}; updown = [0, 0]
    rank = lambda v: 99 if v == "cap" else int(v)
    for k, v in fb.items():
        if ub.get(k) and uo.get(k) and disjoint(k, ub[k], uo[k]):
            g = behav_group(k); h[g][0] += v in ub[k]; h[g][1] += v in uo[k]
            if k.startswith("rt_") and v in ub[k]:
                updown[0] += rank(v) > max(map(rank, uo[k])); updown[1] += rank(v) < min(map(rank, uo[k]))
    def blk(hb, ho, two):
        if ho > 1 or hb - ho < BEHAV_MIN_ITEMS: return 0
        return 2 if two else 1
    return (blk(*h["cap"], h["cap"][0] - h["cap"][1] >= STRONG_CAPABILITY)
            + blk(*h["reason"], min(updown) >= MIXED_REASONING) + blk(*h["style"], False))

# G17: one-word answer distributions (temperature 1). Twins with identical single answers still differ in how often
# they pick each answer. Weights-level evidence, counted as ONE block (a hidden prompt could shift it).
SAMPLE_SYSTEM = "Answer with exactly one word."
SAMPLE_TASKS = {
    "number_1_100": "Name a random number between 1 and 100. Answer with only the number.",
    "number_1_10": "Name a random number between 1 and 10. Answer with only the number.",
    "favorite_number": "What is your favorite number? Answer with only the number.",
    "letter": "Name a random letter of the English alphabet. Answer with only the letter.",
    "word": "Name a random word. Answer with only the word.", "color": "Name a random color. Answer with only the color.",
    "favorite_color": "What is your favorite color? Answer with only the color.",
    "animal": "Name a random animal. Answer with only the animal.", "city": "Name a random city. Answer with only the city name.",
    "coin": "Flip a coin. Answer with only 'heads' or 'tails'."}
DIST_MARGIN = 0.10    # G17b: calibrated on held-out reference sets only


def sample_rows(model, tag, n, base=None, key_env=None, provider=None):
    """n temperature-1 samples per task, chat path, effort low; per-sample rows (the format build() groups)."""
    url, key = endpoint(base, key_env)
    def one(a):
        task, i = a
        r = chat(url, key, model, [{"role": "system", "content": SAMPLE_SYSTEM}, {"role": "user", "content": SAMPLE_TASKS[task]}],
                 max_tokens=400, reasoning={"effort": "low"}, provider=provider, extra={"temperature": 1})
        m = (((r.get("json") or {}).get("choices") or [{}])[0].get("message") or {})
        return {"set": tag, "model": model, "task": task, "rep": i, "status": r.get("status"), "text": m.get("content")}
    with cf.ThreadPoolExecutor(6) as ex:
        return list(ex.map(one, [(t, i) for t in SAMPLE_TASKS for i in range(n)]))


def group_samples(rows):
    """per-sample rows {model, set, task, status, text} -> {(model, set): {task: [normalised answers]}}"""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("status") == 200 and r.get("text"):
            g[(r["model"], r["set"])][r["task"]].append(r["text"].strip().lower().strip(".!\"'*` "))
    return {k: dict(v) for k, v in g.items()}


def _jsd(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b); s = 0.0
    for k in set(ca) | set(cb):
        p, q = ca[k] / len(a), cb[k] / len(b); m = (p + q) / 2
        s += (p * math.log2(p / m) if p else 0) + (q * math.log2(q / m) if q else 0)
    return s / 2


_DJ = {}


def dist_jsd(obs, sets):
    hit = _DJ.get((id(obs), id(sets)))                 # certification re-asks the same (observation, reference) pair
    if hit and hit[0] is obs and hit[1] is sets: return hit[2]
    pooled = collections.defaultdict(list)
    for s in sets:
        for t, v in s.items(): pooled[t] += v
    ts = [t for t in obs if obs[t] and pooled.get(t)]
    v = sum(_jsd(obs[t], pooled[t]) for t in ts) / len(ts) if ts else math.nan
    _DJ[(id(obs), id(sets))] = (obs, sets, v)
    return v


def dist_split(f, best_ref, other_ref):
    """1 block for best if the observed distribution is closer to best's reference by at least DIST_MARGIN."""
    if not f.get("dist") or not best_ref.get("dist") or not other_ref.get("dist"): return 0
    return int(dist_jsd(f["dist"], other_ref["dist"]) - dist_jsd(f["dist"], best_ref["dist"]) >= DIST_MARGIN)



LAYERS = {"tok", "stack", "art"}   # ablation switch (fpx.py evaluate --layers tok,art ...)


def model_score(f, ref):
    """Lower = closer: tokenizer distance (token units) + serving-stack field mismatches, nearest reference probe.
    Returns (score, structure); they are equal since decode speed was removed (A003)."""
    def one(p):
        d = tok_dist(f["tok"], p["tok"]) if "tok" in LAYERS else 0.0
        if "stack" in LAYERS: d += stack_mismatch(f.get("stack") or {}, p.get("stack") or {})
        return d
    dt = min(one(p) for p in ref["probes"])
    if math.isinf(dt): dt = 50.0                       # reference without valid probes: far, but comparable
    return dt, dt


def valid(f):
    """A probe is usable only if the endpoint actually answered the tokenizer probes (M002: 404/429 probes)."""
    return sum(v is not None for v in f["tok"].values()) >= len(PROBES) // 2


def _apply(f, su, bu):
    g = dict(f); g["stack"] = {**(f.get("stack") or {}), **su}; g["behav"] = {**(f.get("behav") or {}), **bu}
    return g


def sibling_changes(f, r):
    """Every single change of observation f toward sibling reference r, as (stack update, behaviour update): one field,
    one whole structural block, one behaviour answer, the whole style block (a hidden prompt), the whole reasoning block
    (a host switching reasoning on/off or changing effort moves all of it together; G19, A005). Deduplicated."""
    fs, fb, out = f.get("stack") or {}, f.get("behav") or {}, {}
    def add(su, bu): out.setdefault(json.dumps([su, bu], sort_keys=True, default=str), (su, bu))
    for sp in r["probes"]:
        ss = sp.get("stack") or {}
        diff = [k for k in ss if fs.get(k, ss[k]) != ss[k]]
        for k in diff: add({k: ss[k]}, {})
        for blk in set(BLOCK_OF.values()):
            ks = [k for k in diff if BLOCK_OF.get(k) == blk]
            if len(ks) > 1: add({k: ss[k] for k in ks}, {})
    for sb in behav_sessions(r):
        bdiff = [k for k in sb if k in fb and fb[k] != sb[k]]
        for k in bdiff: add({}, {k: sb[k]})
        style = [k for k in bdiff if behav_group(k) == "style"]
        if len(style) > 1: add({}, {k: sb[k] for k in style})       # the whole style block moves (a hidden prompt)
    # All capability answers turning into the sibling's is a weights swap, not drift: naming the sibling is then correct.
    return list(out.values())


def knob_changes(f):
    """G14: one reasoning-effort change moves every reasoning length the same way; effort 'none' removes reasoning."""
    fb = f.get("behav") or {}
    rts = {k: int(v) for k, v in fb.items() if k.startswith("rt_") and v != "cap"}
    for d in (-2, -1, 1, 2): yield {}, {k: str(max(v + d, 1)) for k, v in rts.items() if v}
    yield {}, {**{k: "0" for k in rts}, **{k: "-" for k in fb if k.startswith("think_b_")}}


def knob_variants(f):
    for su, bu in knob_changes(f): yield _apply(f, su, bu)


def certify_name(f, cand, best):
    """G10: withhold the name if one change toward a sibling (field, structural block, behaviour answer, style block)
    or one effort-knob change would name another model."""
    fam = cand[best].get("family") or family_of(best)
    wrong = lambda g: classify(g, cand, certify=False)["model"] not in (best, "unknown")
    if any(wrong(_apply(f, su, bu)) for su, bu in knob_changes(f)): return False
    return not any(wrong(_apply(f, su, bu)) for m, r in cand.items()
                   if m != best and (r.get("family") or family_of(m)) == fam for su, bu in sibling_changes(f, r))


def classify(f, refs, exclude=None, certify=True):
    """Open-set classifier: name a family/model only inside frozen acceptance bounds; otherwise abstain."""
    if not valid(f):
        return {"lab": "insufficient_data", "family": "insufficient_data", "model": "insufficient_data",
                "lab_by_artifact": f["art_lab"]}
    cand = {m: r for m, r in refs.items() if m != exclude}
    if not cand:
        return {"lab": "insufficient_reference", "family": "insufficient_reference",
                "model": "unknown", "lab_by_artifact": f["art_lab"]}
    art = f["art_lab"] if "art" in LAYERS else None
    if art:
        art_cand = {m: r for m, r in cand.items() if r["lab"] == art}
        if not art_cand:
            return {"lab": art, "family": FAMILY_BY_LAB.get(art, "unknown"), "model": "unknown",
                    "lab_by_artifact": art, "confidence": "medium", "evidence": "reasoning_artifact",
                    "why_model_unknown": "no reference model from the artifact-identified lab"}
        cand = art_cand
    scored = sorted((model_score(f, r) + (m,) for m, r in cand.items()))
    (s1, dt1, best), runner = scored[0], (scored[1] if len(scored) > 1 else (math.inf, math.inf, None))
    margin = runner[0] - s1
    family_of_m = lambda m: cand[m].get("family") or family_of(m)
    # A001: without an artifact, the nearest other family must be clearly further away (no inclusive-boundary ties).
    other_family = min((dt for _, dt, m in scored if family_of_m(m) != family_of_m(best)), default=math.inf)
    family_ok = art is not None or (dt1 <= MAX_FAMILY_DISTANCE and other_family - dt1 >= MIN_FAMILY_GAP)
    # R2 (A001): the runner-up must be at least two structural units behind, so no single drifting field
    # (finish reason, error text, tool-ID, fingerprint) can hand the name to a sibling.
    # G4: ...and at least two independent blocks behind, so one causal change (e.g. a new tool serializer that
    # alters both tool_id and tool_keys) cannot either.
    # G12: when several candidates tie on structure, the behaviour evidence decides which one is the best match.
    # G14: behaviour evidence for the other candidate counts against the lead (contradicting evidence -> abstain).
    lead_of = lambda c: min((block_dist(f, cand[m]) - block_dist(f, cand[c]) + behav_split(f, cand[c], cand[m])
                             - behav_split(f, cand[m], cand[c]) + dist_split(f, cand[c], cand[m])
                             - dist_split(f, cand[m], cand[c]) for m in cand if m != c), default=math.inf)
    ties = [m for _, dt, m in scored if dt - dt1 < 0.5]
    if len(ties) > 1:
        best = max(ties, key=lead_of); dt1 = model_score(f, cand[best])[1]
    block_lead = lead_of(best)
    # R4 (G8): the lead is counted in independent blocks only (block lead >= 2 implies >= 2 differing fields).
    model_ok = dt1 <= MAX_MODEL_DISTANCE and block_lead >= MIN_STRUCTURAL_LEAD
    # G10/G19 certification is a redundant safety check against a single changed evidence item.
    if model_ok and certify:
        model_ok = certify_name(f, cand, best)
    lab = cand[best]["lab"] if family_ok else "unknown"
    family = family_of_m(best) if family_ok else "unknown"
    return {"lab": lab, "family": family, "lab_by_artifact": f["art_lab"],
            "model": best if model_ok else "unknown", "candidate_model": best,
            "confidence": "high" if model_ok else "medium" if family_ok else "low",
            "score": round(s1, 2), "structural_distance": round(dt1, 2), "margin": round(margin, 2),
            "runner_up": runner[2], "unknown_model": not model_ok,
            "why_model_unknown": None if model_ok else
                f"requires structural_distance<={MAX_MODEL_DISTANCE} "
                f"and a structural lead >={MIN_STRUCTURAL_LEAD} over the runner-up"}


def verify_ref(f, refs, claimed):
    """CONSISTENT only if the claimed model is the nearest reference and every available layer agrees."""
    if not valid(f): return {"claimed": claimed, "verdict": "insufficient_data"}
    r = refs[claimed]; s, dt = model_score(f, r); best = classify(f, refs)
    checks = {"tokenizer+stack": dt <= 0.5,
              "artifact": f["art_lab"] is None or f["art_lab"] == r["lab"],
              "nearest_is_claimed": best["model"] == claimed}
    return {"claimed": claimed, "verdict": "CONSISTENT" if all(checks.values()) else "MISMATCH",
            "checks": checks, "nearest": best}


def verify_family(path, claimed):
    """Claude / GPT exact-model verification: capability gates + reasoning habit (fp2). A class label ("a|b") means
    the claim is consistent with a group that no timing-free signal splits; use audit (replay, cache) to narrow it."""
    r = fp2.identify(path, claimed)
    label, twin = r["label"] or "", fp2.GPT_TWIN.split("|")
    group = label.split("|") if label and not label.startswith("MISMATCH") else []
    if claimed in group or (claimed in twin and label == fp2.GPT_TWIN):
        verdict = "CONSISTENT" if (len(group) == 1 or label == fp2.GPT_TWIN) else "CONSISTENT_WITH_GROUP"
    else:
        verdict = "MISMATCH" if group else "UNRESOLVED"
    note = ("gpt-6.1-sol and gpt-6-astra are observationally identical (G001-G006); cannot be told apart" if claimed in twin
            else f"timing-free signals cannot split {label}; run audit (replay, cache) to narrow" if verdict == "CONSISTENT_WITH_GROUP" else None)
    return {"claimed": claimed, "verdict": verdict, "confidence": "high" if r["consistent"] else "low",
            "gate_layer": r["gate_label"], "habit": r["habit"], "note": note}


def cache_oracle(path, claimed, alt, reps=2):
    """Novel substitution test (G010/G011): providers keep per-model prompt caches. Prime a unique ~2k-token
    prefix under `claimed`, confirm it hits under `claimed`, then read cached tokens under `alt` on the SAME endpoint.
    alt hit => both labels reach the same deployment (substitution or alias)."""
    import uuid
    path = fp2.resolve(path)
    fill = "The quick brown fox jumps over the lazy dog near the quiet river bank. " * 150
    def send(m, prefix):
        q = prefix + "\nReply with: ok"
        if m in fp2.GPT:
            r = fp2.call(path, m, {"input": [{"role": "user", "content": q}], "reasoning": {"effort": "low"}}, timeout=180)
            return ((r.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens") or 0
        r = fp2.call(path, m, {"max_tokens": 32, "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"},
                               "messages": [{"role": "user", "content": [{"type": "text", "text": q, "cache_control": {"type": "ephemeral"}}]}]}, timeout=180)
        return (r.get("usage") or {}).get("cache_read_input_tokens") or 0
    runs = []
    for _ in range(reps):
        pre = f"Session {uuid.uuid4()}. " + fill
        send(claimed, pre); time.sleep(3); ctrl = send(claimed, pre); time.sleep(3); probe_ = send(alt, pre)
        runs.append({"control_cached": ctrl, "alt_cached": probe_})
    if not all(r["control_cached"] for r in runs):
        verdict = "NO_CACHE_SIGNAL"      # endpoint hides or disables caching: oracle abstains
    elif all(r["alt_cached"] for r in runs):
        verdict = "SHARED_DEPLOYMENT"    # claimed and alt are served by the same model state
    elif not any(r["alt_cached"] for r in runs):
        verdict = "SEPARATE_DEPLOYMENTS"
    else:
        verdict = "INCONSISTENT"
    return {"claimed": claimed, "alt": alt, "verdict": verdict, "runs": runs}


# T002 (OpenRouter, deterministic): target label -> set of source models whose signed thinking it KEEPS on replay
CLAUDE_KEEP = {"opus-5.5": {"opus-5.5", "sonnet-5.5", "opus-5", "sonnet-5", "haiku-5.5"},
               "sonnet-5.5": {"sonnet-5.5", "sonnet-5", "haiku-5.5"}, "opus-5": {"opus-5", "sonnet-5"},
               "sonnet-5": {"sonnet-5"}, "haiku-5.5": {"sonnet-5", "haiku-5.5"}}
REPLAY_Q = "Compute 3847 * 2963 exactly. Show only the final number."


def _keeps(target, source):
    if target in fp2.GPT:   # G003: OpenAI encrypted reasoning kept iff same generation
        return target.startswith("gpt-6") == source.startswith("gpt-6")
    return source in CLAUDE_KEEP[target]


def replay_oracle(path, claimed, alts):
    """Crypto layer (T002/G003, no official key needed): produce a signed reasoning block under the claimed label,
    replay it unchanged into the claimed label (control) and each sibling label on the same endpoint. The keep/drop
    vector (read from input_tokens with vs without the block) must match the claimed model as producer."""
    path = fp2.resolve(path); gpt = claimed in fp2.GPT
    if gpt:
        g = fp2.call(path, claimed, {"input": [{"role": "user", "content": REPLAY_Q}], "reasoning": {"effort": "medium"},
                                     "include": ["reasoning.encrypted_content"]}, timeout=240)
        items = [i for i in g.get("reasoning_items") or [] if i.get("encrypted_content")]
        if not items: return {"result": "ABSTAIN", "why": "no encrypted reasoning returned"}
        block = {"type": "reasoning", "summary": items[0].get("summary") or [], "encrypted_content": items[0]["encrypted_content"]}
        def conv(with_block):
            return {"input": [{"role": "user", "content": REPLAY_Q}] + ([block] if with_block else []) +
                    [{"role": "assistant", "content": g.get("text") or "ok"}, {"role": "user", "content": "Now reply with: ok"}],
                    "reasoning": {"effort": "low"}}
    else:
        g = fp2.call(path, claimed, {"max_tokens": 4000, "stream": True, "thinking": {"type": "adaptive", "display": "summarized"},
                                     "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": REPLAY_Q}]}, timeout=240)
        if not (g.get("thinking") and g.get("signature")): return {"result": "ABSTAIN", "why": "no signed thinking returned"}
        def conv(with_block):
            asst = ([{"type": "thinking", "thinking": g["thinking"], "signature": g["signature"]}] if with_block else []) + \
                   [{"type": "text", "text": g.get("text") or "ok"}]
            return {"max_tokens": 200, "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"},
                    "messages": [{"role": "user", "content": REPLAY_Q}, {"role": "assistant", "content": asst},
                                 {"role": "user", "content": "Now reply with: ok"}]}
    kept, raw = {}, {}
    for t in [claimed] + alts:
        a, b = fp2.call(path, t, conv(True), timeout=180), fp2.call(path, t, conv(False), timeout=180)
        ia, ib = (a.get("usage") or {}).get("input_tokens"), (b.get("usage") or {}).get("input_tokens")
        raw[t] = {"with": ia, "without": ib, "status": a["status"]}
        kept[t] = None if (ia is None or ib is None) else (ia - ib > 20)
    if not kept.get(claimed):
        return {"result": "ABSTAIN", "why": "control: claimed label did not keep its own block", "kept": kept, "raw": raw}
    cands = [m for m in (fp2.GPT if gpt else CLAUDE_KEEP) if all(kept[t] is None or _keeps(t, m) == kept[t] for t in kept)]
    if claimed not in cands:
        res = "FAIL"
    elif any(a in cands for a in alts):
        res = "ABSTAIN"     # keep/drop pattern cannot tell claimed from this sibling (e.g. same GPT generation)
    else:
        res = "PASS"
    return {"result": res, "producer_candidates": cands, "kept": kept, "raw": raw}


def fuse_results(results):
    """Conservative evidence-group fusion; one contradiction or fewer than three supports abstains."""
    if results.count("FAIL") >= 2:
        return "CONTRADICTED"
    if "FAIL" in results:
        return "INDETERMINATE"
    return "CONSISTENT_WITH" if results.count("PASS") >= 3 else "INDETERMINATE"


def audit(path, claimed, alts):
    """Stack four evidence groups without claiming cryptographic identity.
    Three non-contradictory supports -> CONSISTENT_WITH; two contradictions -> CONTRADICTED; else abstain."""
    def match(lbl):
        if lbl in (None, "unknown") or str(lbl).startswith("MISMATCH"): return "ABSTAIN"
        if lbl == "not-twin": return "FAIL" if claimed in fp2.GPT_TWIN.split("|") else "ABSTAIN"   # gate rules out only the twin
        group = str(lbl).split("|")
        if claimed not in group: return "FAIL"
        return "PASS" if len(group) == 1 or lbl == fp2.GPT_TWIN else "ABSTAIN"   # a wider class supports nothing specific
    ident = fp2.identify(path, claimed)
    layers = {"label (capability gates)": {"result": match(ident["gate_label"]), "label": ident["gate_label"]},
              "habit (reasoning tokens)": {"result": match(ident["label"]), "label": ident["label"], "habit": ident["habit"]},
              "crypto (reasoning replay)": replay_oracle(path, claimed, alts)}
    cache = [cache_oracle(path, claimed, a) for a in alts]
    cres = ["FAIL" if c["verdict"] == "SHARED_DEPLOYMENT" else "PASS" if c["verdict"] == "SEPARATE_DEPLOYMENTS" else "ABSTAIN"
            for c in cache]
    layers["state (cross-label cache)"] = {"result": "FAIL" if "FAIL" in cres else "PASS" if cres and all(r == "PASS" for r in cres)
                                           else "ABSTAIN", "per_alt": cache}
    res = [l["result"] for l in layers.values()]
    verdict = fuse_results(res)
    evidence_counts = {x.lower(): res.count(x) for x in ("PASS", "FAIL", "ABSTAIN")}
    need = {"label (capability gates)": "route gate-shaped requests to the real model or forge provider error texts",
            "habit (reasoning tokens)": "run a model with the claimed model's reasoning habit on a fixed prompt",
            "crypto (reasoning replay)": "return reasoning signed by the claimed model (i.e. run it)",
            "state (cross-label cache)": "keep claimed and sibling labels on separate caches (i.e. separate backends)"}
    return {"path": fp2.resolve(path), "claimed": claimed, "alts": alts, "verdict": verdict,
            "evidence_counts": evidence_counts, "layers": layers,
            "to_fake_this_verdict": [need[k] for k, l in layers.items() if l["result"] == "PASS"]}


def leave_one_lab_out(refs, observations):
    """Measure open-set behavior after removing every reference from the probe's true lab."""
    outcomes, cm, abstained = collections.Counter(), collections.Counter(), 0
    for true_lab, f in observations:
        c = classify(f, {m: r for m, r in refs.items() if r["lab"] != true_lab})
        if c["lab"] == "insufficient_data":
            abstained += 1
            continue
        outcome = "correct" if c["lab"] == true_lab else "unknown" if c["lab"] in ("unknown", "insufficient_reference") else "wrong"
        outcomes[outcome] += 1
        cm[(true_lab, c["lab"])] += 1
    return outcomes, cm, abstained


def evaluate(refs_path, files):
    refs = load_reference_db(refs_path)
    rows = [json.loads(l) for fn in files for l in open(fn)]
    nlab = collections.Counter(r["lab"] for r in refs.values() if usable_ref(r))
    def run(loo=False):
        lab_ok = mod_ok = model_pred = n = abst = 0; cm = collections.Counter()
        for p in rows:
            if p["model"] not in refs: continue
            if loo and (nlab[lab_of(p["model"])] < 2 or not usable_ref(refs[p["model"]])): continue  # need a known sibling
            f = features(p)
            c = classify(f, refs, exclude=p["model"] if loo else None)
            if c["lab"] == "insufficient_data": abst += 1; continue
            n += 1; lab_ok += c["lab"] == lab_of(p["model"])
            if c["model"] != "unknown":
                model_pred += 1; mod_ok += c["model"] == p["model"]
            cm[(lab_of(p["model"]), c["lab"])] += 1
        return lab_ok / n, mod_ok / model_pred if model_pred else 0, model_pred / n, n, cm, abst
    def show(title, cm):
        labs = sorted({a for a, _ in cm} | {b for _, b in cm})
        print(title + "\n" + "".ljust(12) + "".join(x[:9].rjust(10) for x in labs))
        for a in labs: print(a[:11].ljust(12) + "".join(str(cm[(a, b)] or ".").rjust(10) for b in labs))
    lab, mod, coverage, n, cm, ab = run()
    print(f"layers={sorted(LAYERS)}  answered probes: {n} (abstained insufficient_data: {ab})  "
          f"lab accuracy {lab:.1%}  exact-model precision {mod:.1%} at {coverage:.1%} coverage")
    show("in-DB lab confusion (true \\ pred):", cm)
    l2, _, _, n2, cm2, ab2 = run(loo=True)
    print(f"leave-one-model-out over labs with >=2 valid reference models ({n2} probes, {ab2} abstained): lab {l2:.1%}")
    show("LOMO lab confusion (true \\ pred):", cm2)
    loo_lab, cm3, ab3 = leave_one_lab_out(
        refs, ((lab_of(p["model"]), features(p)) for p in rows if p["model"] in refs))
    print(f"leave-one-lab-out ({sum(loo_lab.values())} probes, {ab3} insufficient_data): "
          f"correct artifact family {loo_lab['correct']}, unknown {loo_lab['unknown']}, wrong family {loo_lab['wrong']}")
    show("LOLO lab confusion (true \\ pred):", cm3)



def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    base, kenv, prov = _opt("--base"), _opt("--key-env"), _opt("--provider")
    if _opt("--layers"): LAYERS.clear(); LAYERS.update(_opt("--layers").split(","))
    if cmd == "probe":
        out, tag, model = sys.argv[2:5]
        p = probe(model, base, kenv, prov); p["session"], p["provider_pin"] = tag, prov
        open(out, "a").write(json.dumps(p) + "\n")
        print(tag, model, json.dumps({k: v for k, v in features(p).items() if k != "art"}))
    elif cmd == "build":
        build(sys.argv[2], sys.argv[3:])
    elif cmd == "identify":
        refs = load_reference_db(sys.argv[2]); f = features(probe(sys.argv[3], base, kenv, prov)); c = classify(f, refs)
        # G17 stage 2: family known, model not, and the family has one-word distribution references -> sample (100 req).
        if c["model"] == "unknown" and any(r.get("dist") and r.get("family") == c["family"] for r in refs.values()):
            f["dist"] = next(iter(group_samples(sample_rows(sys.argv[3], "id", 10, base, kenv, prov)).values()), None)
            c = classify(f, refs)
        print(json.dumps({"features": f, "result": c}, indent=1, default=str))
    elif cmd == "token-id":
        r = identify_token_id_namespace(sys.argv[2], sys.argv[3], base, kenv, prov)
        if _opt("--out"): json.dump(r, open(_opt("--out"), "w"), indent=1)
        print(json.dumps(r["result"], indent=1))
    elif cmd == "verify-ref":
        refs = load_reference_db(sys.argv[2]); f = features(probe(sys.argv[3], base, kenv, prov))
        print(json.dumps(verify_ref(f, refs, sys.argv[4]), indent=1, default=str))
    elif cmd == "verify":
        print(json.dumps(verify_family(sys.argv[2], sys.argv[3]), indent=1, default=str))
    elif cmd == "cache-oracle":
        print(json.dumps(cache_oracle(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
    elif cmd == "audit":
        alts = list(itertools.takewhile(lambda a: not a.startswith("--"), sys.argv[4:]))
        r = audit(sys.argv[2], sys.argv[3], alts)
        if _opt("--out"): open(_opt("--out"), "a").write(json.dumps(r, default=str) + "\n")
        brief = {k: v for k, v in r.items() if k != "layers"}
        brief["layers"] = {k: {kk: vv for kk, vv in v.items() if kk in ("result", "label", "habit", "producer_candidates", "kept", "why")}
                           | ({"per_alt": [(c["alt"], c["verdict"]) for c in v["per_alt"]]} if "per_alt" in v else {})
                           for k, v in r["layers"].items()}
        print(json.dumps(brief, indent=1, default=str))
    elif cmd == "evaluate":
        args, skip = [], False
        for a in sys.argv[3:]:
            if skip: skip = False; continue
            if a.startswith("--"): skip = True; continue
            args.append(a)
        evaluate(sys.argv[2], args)
    else:
        print(__doc__)
