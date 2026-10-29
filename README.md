# fpx: which model is behind this API?

A small black-box tool for checking LLM API endpoints, plus the technical report that describes and evaluates it.

- **Report:** [`docs/PAPER.pdf`](docs/PAPER.pdf) ([markdown](docs/PAPER.md))
- **Blog post:** <https://kre80r.com/is-it-really-opus/>

## What it does

| Command | Question | Requests |
|---|---|---|
| `fpx.py identify` | Which lab / model is this unknown endpoint? Names a model only when the evidence is clear, otherwise `unknown`. | 42 (+100 for close siblings) |
| `fpx.py verify` | Is this Claude 5/5.5 or GPT-6/5.6 label the model it claims? | 4–7 |
| `fpx.py cache-oracle` | Do two model names on one endpoint share a prompt cache (a sign of a swap)? | 6 |
| `fpx.py token-id` | Optional: does the endpoint expose the OpenAI o200k token namespace? | 32 |

`identify` compares the target with the saved references in `references.json` (40 models, 12 companies). It does not call the reference models again.

## Run

Python 3.10+, standard library only.

```bash
export SUSPECT_KEY=...
python3 fpx.py identify references.json <model-id> --base https://host/v1 --key-env SUSPECT_KEY
python3 fpx.py --help
python3 -m unittest test_fpx
```

## Assess sellers without overclaiming

For marketplaces, target the seller explicitly with `--pin PEER_ID` on `probe`,
`identify`, `verify-ref`, or `token-id`. Each response must echo the requested
`x-antseed-peer-id`; an absent or different echo is not identification evidence.

```bash
python3 collect_sellers.py --base https://router.example/v1 --out-dir seller-runs/current
python3 seller_audit.py seller-runs/current/rows.jsonl \
  --catalog seller-runs/current/catalog.json --refs references.json \
  --out seller-runs/current/assessment.json --markdown seller-runs/current/assessment.md
```

This separates **operational readiness** from **exact-model confidence**. Two
successful tasks do not authenticate a model, a tokenizer match does not prove
honesty, and a negative baseline-subtracted delta is not a negative raw count.
Without independent exact-model calibration, identity recommendations stay
`unknown`. See [evidence requirements and interpretation](docs/SELLER_AUDIT.md).
Raw runs under `seller-runs/` are not committed; publish code, not seller evidence.

## Limits (read the report before trusting a result)

- Built for honest, cooperative endpoints. A provider that rewrites replies on purpose can fool it.
- All `identify` results in the report were measured through OpenRouter with pinned providers. Other routes are untested.
- In the last blind test it named 32 of 40 models in both sessions with no wrong name (29 of 40 without gateway-written fields).
- The cache check can be confused by sellers that add their own cached text to every request; send a brand-new prompt first.
- References drift as providers change their software.

## Licence and citation

Code: MIT (`LICENSE`). Report: CC BY 4.0 (`LICENSE-paper`). Please cite as given in `CITATION.cff` (KrE80r, <https://kre80r.com>).
