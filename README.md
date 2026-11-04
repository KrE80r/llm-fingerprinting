# fpx: which model is behind this API?

A small black-box tool for checking LLM API endpoints, plus the technical report that describes and evaluates it.

- **Report:** [`docs/PAPER.pdf`](docs/PAPER.pdf) ([markdown](docs/PAPER.md))
- **Blog post:** <https://kre80r.com/is-it-really-opus/>

## What it does

| Command | Question | Requests |
|---|---|---|
| `fpx.py identify` | Which lab / model is this unknown endpoint? Names a model only when the evidence is clear, otherwise `unknown`. | 43 (+100 for close siblings; +140–520 when token counts are fake) |
| `fpx.py cuts` | Which tokenizer generated the replies? Works even when the provider invents its usage numbers. | 140–520 |
| `fpx.py verify` | Is this Claude 5/5.5 or GPT-6/5.6 label the model it claims? | 4–7 |
| `fpx.py cache-oracle` | Do two model names on one endpoint share a prompt cache (a sign of a swap)? | 6 |
| `fpx.py token-id` | Optional: does the endpoint expose the OpenAI o200k token namespace? | 32 |

`identify` compares the target with the saved references in `references.json` (40 models, 12 companies). It does not call the reference models again.

**Fake token counts.** Most layers read `usage.prompt_tokens`, so `identify` first sends one random 800-letter string that no real tokenizer can encode in fewer than 411 tokens. A lower count proves the provider estimates usage (OpenRouter's stealth slot `stealth/crouton-alpha` reports about 320), and `identify` then refuses to name anything from those layers.

**Output boundaries.** Instead, it asks the endpoint to echo two fixed texts under every `max_tokens` budget where the echo is visible but unfinished. A reply cut by `max_tokens` ends on a boundary of the tokenizer that generated it, whatever the provider reports. Only cuts inside a word count, because some layers release text word by word. A tokenizer is named only when every cut is one of its boundaries and every other enrolled tokenizer is contradicted by at least 3 cuts. `boundary_references.json` holds 15 tokenizers (9 open, plus Claude, Gemini, Mistral Large, Qwen and two Grok tokenizers measured live by `build_boundaries.py`). Blind check with references frozen: 20 endpoints from enrolled families gave 14 named, 6 abstained, 0 wrong; 9 endpoints from other labs gave 8 abstentions and one reused tokenizer (Inception Mercury-3 cuts on DeepSeek boundaries). It names a tokenizer, not a lab: tokenizers are shared (Nvidia Nemotron-3 uses Mistral's Tekken).

## Run

Python 3.10+, standard library only (`build_boundaries.py`, which rebuilds the boundary references, also needs `transformers` and `tiktoken`).

```bash
export SUSPECT_KEY=...
python3 fpx.py identify references.json <model-id> --base https://host/v1 --key-env SUSPECT_KEY
python3 fpx.py cuts boundary_references.json <model-id> --base https://host/v1 --key-env SUSPECT_KEY
python3 fpx.py --help
python3 -m unittest test_fpx
```

## Limits (read the report before trusting a result)

- Built for honest, cooperative endpoints. A provider that rewrites replies on purpose can fool it.
- All `identify` results in the report were measured through OpenRouter with pinned providers. Other routes are untested.
- In the last blind test it named 32 of 40 models in both sessions with no wrong name (29 of 40 without gateway-written fields).
- The cache check can be confused by sellers that add their own cached text to every request; send a brand-new prompt first.
- References drift as providers change their software.
- The output-boundary layer names a tokenizer, which several labs can share. It abstains on GPT vs MiMo, GLM vs MiMo and Qwen vs MiMo when an endpoint gives too few mid-word cuts.

## Licence and citation

Code: MIT (`LICENSE`). Report: CC BY 4.0 (`LICENSE-paper`). Please cite as given in `CITATION.cff` (KrE80r, <https://kre80r.com>).
