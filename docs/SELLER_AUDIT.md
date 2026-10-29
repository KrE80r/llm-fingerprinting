# Seller assessment: identity is not availability

A catalog entry, matching tokenizer counts, a model's self-description, reasoning tokens or a router-supplied peer header cannot authenticate model weights. Report what the observations support, without inferring intent.

## Collect and replay private evidence

Python 3.10+, standard library only:

```sh
# Use a new directory to preserve previous observations.
python3 collect_sellers.py --base https://router.example/v1 \
  --out-dir seller-runs/current --workers 2 --timeout 45

python3 seller_audit.py seller-runs/current/rows.jsonl \
  --catalog seller-runs/current/catalog.json --refs references.json \
  --out seller-runs/current/assessment.json --markdown seller-runs/current/assessment.md

python3 -m unittest discover
```

The collector resolves exact model IDs or explicit aliases, never sibling models by substring. It preserves distinct provider/service offers from the same peer. Its defaults cover Opus 5.5, Muse Wonder, Kimi K3.1 Pro, GPT 6 Astra, GLM 5.3 and GPT 6.1 Terra; `--models` selects other explicit catalog entries.

Each offer receives two fresh tasks: nonce echo and arithmetic. There are at most two offers in flight by default, no automatic retries, and a 1024-token output budget per request. Unfinished responses are inconclusive rather than incorrect answers. Use `--key-env VARIABLE_NAME` for authentication; the credential is not stored in request headers or collection logs.

For a separate full fingerprint:

```sh
python3 fpx.py probe seller-runs/one.jsonl session-1 claimed-model \
  --base https://router.example/v1 --pin PEER_ID
```

Pinned requests require an identical `x-antseed-peer-id` response header. Missing or mismatched attribution excludes the response from identification evidence. This verifies the router's attribution, not the provider's upstream model weights. Tokenizer cells, reasoning, stack, behaviour and follow-on sampling need their own attribution; a single successful echo is insufficient.

## Interpret results separately

- **Exact-model unknown:** independent, exact-model calibrated evidence is unavailable or inconclusive. Successful readiness checks do not change this.
- **Usable in this test:** both attributable readiness tasks passed. This supports trying comparable work, not trusting the advertised model name.
- **Avoid for now:** repeated attributable requests met the documented operational failure criterion. This concerns observed service reliability, not fraud or future behaviour.
- **Operational unknown:** policy/authentication blocks, rate limits, missing attribution, unfinished responses or insufficient observations prevent a conclusion.

Do not flatten per-model observations into a blanket judgment about a seller. Keep repeated and contradictory observations rather than selecting the best-looking run.

Tokenizer deltas are differences, not raw counts: a synthetic example is 20 raw tokens minus a 30-token baseline, yielding -10. Neither raw count is negative. Variable hidden prompts, adapters or metering conventions can invalidate a constant-offset comparison. Usage fields alone do not establish fraud or billed amounts. Missing reasoning can reflect settings, filtering or reporting rather than model substitution.

## Exact-model reference requirements

Use fresh generation access to the exact advertised model through an independently trusted route, record provider/settings, and evaluate held-out repeats against plausible alternatives. A first-party API is preferred; a trusted broker route must pin the intended provider and disable fallback. A successful `/models` listing does not establish generation access. Never enroll the seller under assessment as its own reference.

The included database does not contain exact references for every default target, and documents unresolved sibling collisions. Sibling references must not substitute for missing exact references. The offline seller report currently has no independently calibrated positive-trust mode: tokenizer compatibility stays compatibility, and exact-model confidence stays unknown. Existing `identify` and `verify-ref` outputs are classifier results, not authenticated seller verdicts; their calibration does not automatically transfer between gateways.

`verify-ref` returns `INDETERMINATE` when the claimed reference remains compatible but the classifier cannot name an exact model; abstention is not a mismatch. An absent claimed reference returns `insufficient_reference`. `MISMATCH` describes disagreement with the enrolled reference, not proof of substitution. The family verifier reports indistinguishable GPT Astra/Sol twins as `CONSISTENT_WITH_GROUP`, never exact-model confirmation.

Even well-calibrated black-box agreement is conditional evidence, not cryptographic attestation of weights. A binary honesty list is unjustified when competing models remain indistinguishable.

## Publication boundary

Raw runs under `seller-runs/` are ignored by Git. They can contain seller identifiers, request IDs, policy details, error bodies and reasoning artifacts. Keep real catalogs, captures, assessments and seller recommendations private and outside published commits. Ignoring a directory is only a guardrail: inspect staged changes before publishing. Publish only generic implementation, synthetic fixtures/tests and generic documentation.

API field semantics: [OpenAI Chat Completions reference](https://platform.openai.com/docs/api-reference/chat/create). Peer pinning follows the existing AntSeed integration in `fp2.py`; router attribution is not upstream attestation.
