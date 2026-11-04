"""Build boundary_references.json for the fpx output-boundary layer (L2c).

  python3 build_boundaries.py collect      <text.txt> <model-id> <out.jsonl> [provider]  # additivity, resumable (~2·len(text) req)
  python3 build_boundaries.py collect-cuts <text.txt> <model-id> <out.jsonl> [provider]  # output sweep (fpx.cut_rows)
  python3 build_boundaries.py build <out.json> <text1.txt> [<text2.txt> ...] -- [<name>@<text#>=<file.jsonl> ...]

Offline tokenizers are used only where their 8-probe token deltas equal the served counts of the listed enrolled
models (references.json). Closed tokenizers are measured live by additivity: excess(c) = prompt_tokens(text[:c]) +
prompt_tokens(text[c:]) - prompt_tokens(text) equals the chat-template constant at every true boundary, so positions
with any other excess are certainly not boundaries. The constant is prompt_tokens(c) - 1 for a single ASCII character
(the text must start and end with one; both ends must agree). A live entry is therefore a superset (Llama-4 check:
all 95 true boundaries plus 13 of 233 non-boundaries): it can under-count contradictions, never invent one.
Boundaries that fall inside a character are floored to the last whole character, matching fpx.cut_points.
Claude does not generate its canonical tokenization (it emits a lone trailing-space token where its prompt counter does
not split), so its entry comes only from complete output sweeps: every budget 1..N with no hidden reasoning, which
observes every boundary the model generates. For the other closed tokenizers, output cuts of the reference model are
added to the additivity set (held-out Gemini/Grok siblings generated 1-3 boundaries that additivity missed).
A tokenizer without a reference for some text is stored as null there and is never contradicted on it.
Needs transformers + tiktoken; fpx.py itself does not.
"""
import collections, concurrent.futures as cf, functools, json, os, sys
import fpx

OFFLINE = {  # name: (tokenizer source, models whose served tokenizer it is)
    "llama4": ("unsloth/Llama-4-Scout-17B-16E-Instruct", ["meta/muse-glimmer-30b", "meta/muse-spark-1.3",
                                                         "meta-llama/llama-4-scout", "meta-llama/llama-4-maverick"]),
    "llama3": ("unsloth/Llama-3.1-8B-Instruct", ["meta-llama/llama-3.1-8b-instruct"]),
    "o200k": ("tiktoken:o200k_base", ["openai/gpt-5-nano", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-6-astra",
                                      "openai/gpt-6-luna", "openai/gpt-6-sol", "openai/gpt-6.1-sol"]),
    "kimi": ("moonshotai/Kimi-K3", ["moonshotai/kimi-k2.6", "moonshotai/kimi-k2.7-code", "moonshotai/kimi-k3"]),
    "tekken": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", ["mistralai/ministral-8b-2512", "mistralai/mistral-medium-3-5"]),
    "nemotron": ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", ["nvidia/nemotron-3-ultra-550b-a55b"]),
    "deepseek": ("deepseek-ai/DeepSeek-V4.1-Flash", ["deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-v4.1-flash"]),
    "mimo": ("XiaomiMiMo/MiMo-V2.6-Flash-RL", ["xiaomi/mimo-v2.5", "xiaomi/mimo-v2.6-flash"]),
    "glm": ("zai-org/GLM-5.3", ["z-ai/glm-5", "z-ai/glm-5.3", "z-ai/glm-5.3-flash", "z-ai/glm-5.3-flashx"]),
    "minimax": ("MiniMaxAI/MiniMax-M3", ["minimax/minimax-m2.5", "minimax/minimax-m2.7", "minimax/minimax-m3"]),
}
LIVE = {  # name: (method, measured model, models with identical served token counts)
    "claude": ("output", "anthropic/claude-haiku-5.5", ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5",
                                                        "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5", "anthropic/claude-opus-5.5"]),
    "gemini": ("additivity", "google/gemini-2.5-flash", ["google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]),
    "mistral-large": ("additivity", "mistralai/mistral-large-4-0", ["mistralai/mistral-large-4-0"]),
    "qwen": ("additivity", "qwen/qwen3.7-flash", ["qwen/qwen3.5-flash-02-23", "qwen/qwen3.7-flash"]),
    "grok-4.20": ("additivity", "x-ai/grok-4.20", ["x-ai/grok-4.20", "x-ai/grok-4.3"]),
    "grok-4.7": ("additivity", "x-ai/grok-4.7", ["x-ai/grok-4.6", "x-ai/grok-4.7"]),
}


@functools.lru_cache(maxsize=None)
def _tokenizer(source):
    """(encode, decode-prefix-of-ids) for a tiktoken encoding or a Hugging Face repo."""
    if source.startswith("tiktoken:"):
        import tiktoken
        enc = tiktoken.get_encoding(source.split(":", 1)[1])
        return enc.encode, lambda ids: enc.decode_bytes(ids).decode("utf-8", errors="ignore")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    return (lambda s: tok.encode(s, add_special_tokens=False),
            lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False).rstrip("\ufffd"))


def offline_boundaries(source, text):
    encode, decode = _tokenizer(source); ids = encode(text)
    prefix = lambda i: decode(ids[:i])
    out = {len(prefix(i)) for i in range(1, len(ids))}
    assert all(text.startswith(prefix(i)) for i in range(1, len(ids))), f"{source}: decode is not a prefix of the text"
    return sorted(c for c in out if 0 < c < len(text))


def live_boundaries(path, text):
    pt = {}
    for line in open(path):
        r = json.loads(line)
        if r["pt"]: pt[r["s"]] = r["pt"]
    # a position stays unmeasured when either side is refused (e.g. Claude content_filter): it may be a boundary
    missing = {c for c in range(1, len(text)) if text[:c] not in pt or text[c:] not in pt}
    if text not in pt or len(missing) > len(text) // 10:
        raise SystemExit(f"{path}: {len(missing)} positions unmeasured; rerun collect")
    const = pt[text[:1]] - 1   # one ASCII character is one token in every BPE vocabulary
    if not (text[0].isascii() and text[-1].isascii()) or pt[text[-1:]] - 1 != const:
        raise SystemExit(f"{path}: template constant disagrees between the first and last character")
    return sorted(c for c in range(1, len(text)) if c in missing or pt[text[:c]] + pt[text[c:]] - pt[text] == const), len(missing)


def collect(text, model, out, provider=None):
    url, key = fpx.endpoint(None, "OPENROUTER_API_KEY")
    strings = sorted({text} | {text[:c] for c in range(1, len(text))} | {text[c:] for c in range(1, len(text))})
    done = {json.loads(l)["s"] for l in open(out) if json.loads(l)["pt"]} if os.path.exists(out) else set()
    def one(s):
        r = fpx._chat(url, key, model, [{"role": "user", "content": s}], max_tokens=16, timeout=60, provider=provider)
        return s, ((r.get("json") or {}).get("usage") or {}).get("prompt_tokens")
    with open(out, "a") as f, cf.ThreadPoolExecutor(8) as ex:
        for s, n in ex.map(one, [s for s in strings if s not in done]):
            f.write(json.dumps({"s": s, "pt": n}, ensure_ascii=False) + "\n"); f.flush()


def output_boundaries(path, text):
    """Exact boundary set from a sweep that saw every budget up to the full echo, each spent entirely on visible text."""
    rows = {r["n"]: r for r in map(json.loads, open(path)) if r["status"] == 200}
    full = min((n for n, r in rows.items() if r["finish"] == "stop" and r["text"].strip() == text), default=None)
    gaps = [n for n in range(1, full or 1) if n not in rows or rows[n]["finish"] != "length" or rows[n]["completion_tokens"] != n]
    if full is None or gaps: raise SystemExit(f"{path}: sweep incomplete or reasoning-shifted at budgets {gaps[:10]}")
    return sorted(fpx.cut_points(rows.values(), text))


def file_boundaries(name, path, text):
    if "pt" in json.loads(open(path).readline()):
        bounds, unmeasured = live_boundaries(path, text)
        return bounds, f"additivity ({unmeasured} positions unmeasured)"
    if LIVE[name][0] == "output":
        return output_boundaries(path, text), "complete output sweep"
    return sorted(fpx.cut_points(map(json.loads, open(path)), text)), "observed output cuts"


def build(out, texts, live_args):
    toks = {name: {"source": [f"offline:{src}"], "models": models, "boundaries": [offline_boundaries(src, t) for t in texts]}
            for name, (src, models) in OFFLINE.items()}
    for arg in live_args:
        key, path = arg.split("=", 1); name, i = key.split("@"); i = int(i) - 1
        _, model, models = LIVE[name]
        t = toks.setdefault(name, {"source": [], "models": models, "boundaries": [None] * len(texts)})
        bounds, how = file_boundaries(name, path, texts[i])
        t["boundaries"][i] = sorted(set(t["boundaries"][i] or []) | set(bounds))
        t["source"].append(f"text{i + 1}: {how}, {model}")
    same = collections.defaultdict(list)   # tokenizers indistinguishable on every text become one entry
    for name, t in toks.items(): same[json.dumps(t["boundaries"])].append(name)
    merged = {"+".join(names): {"source": [s for n in names for s in toks[n]["source"]],
                                "models": sorted({m for n in names for m in toks[n]["models"]}),
                                "boundaries": toks[names[0]]["boundaries"]} for names in same.values()}
    json.dump({"schema": "fpx.boundary-reference/v2", "texts": texts,
               "protocol": {"prompt": fpx.CUT_PROMPT, "coarse_budgets": fpx.CUT_COARSE, "max_dense": fpx.MAX_DENSE,
                            "min_cuts": fpx.MIN_CUTS, "min_split": fpx.MIN_SPLIT},
               "tokenizers": merged}, open(out, "w"), ensure_ascii=False, indent=1)
    # weakest directed separation: boundaries of a that b lacks, over texts where both have references
    sep = lambda a, b: sum(len(set(x) - set(y)) for x, y in zip(merged[a]["boundaries"], merged[b]["boundaries"]) if x and y)
    worst = min((sep(a, b), a, b) for a in merged for b in merged if a != b)
    print(f"{len(merged)} tokenizers; weakest pair: {worst[0]} boundaries of {worst[1]} absent from {worst[2]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect": collect(open(sys.argv[2]).read(), sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    elif cmd == "collect-cuts":
        with open(sys.argv[4], "w") as f:
            for r in fpx.cut_rows(sys.argv[3], open(sys.argv[2]).read(), None, "OPENROUTER_API_KEY", sys.argv[5] if len(sys.argv) > 5 else None):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif cmd == "build":
        sep = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
        build(sys.argv[2], [open(p).read() for p in sys.argv[3:sep]], sys.argv[sep + 1:])
    else: print(__doc__)
