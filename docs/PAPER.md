# Which Model Is Behind This API? Integrating and Evaluating Black-Box LLM Fingerprints

**KrE80r** · <https://kre80r.com> · companion blog post: <https://kre80r.com/is-it-really-opus/>

*Technical report. Measurements: 2026-10-11 through 2026-10-15. Code: `fpx.py`, `fp2.py`. Trial IDs (G21, R002, …) and paths under `trials/` and `ledger/` refer to the author's trial ledger and raw captures, which are not part of the public repository. Report licensed CC BY 4.0, code MIT; please cite as given in `CITATION.cff`.*

## Abstract
LLM API labels may be inaccurate, and stealth endpoints may omit them. This report integrates vendor-documented API gates, cache and replay primitives, tokenizer and serving-stack signatures, and content-based behaviour tests. We report four contributions: (a) a new application of the CacheTracer/CacheProbe cache-reach primitive, using cross-label cache transfer as a substitution test that distinguishes deployments, not weights; (b) a hash-frozen, pre-registered 40-model × 2-session blind evaluation with zero wrong names; (c) measured negative results, including failure modes for timing and token-ID signals; and (d) an extension of Panfilov et al.'s Table 1 to GPT-6.x, where 6.x targets dropped 5.6 traces. All `identify` results are from OpenRouter with pinned providers; transfer to other gateways or resellers is untested.

With the shipped verifier, Claude OpenRouter scores 20/20. Reseller R non-suspect labels score 4/12 exact and 8/12 at the correct group; its suspect label was labelled Opus 5.5 in 4/4 sessions, a verdict rather than an accuracy score. GPT scores 3/18 exact plus 15/18 correct group (6 twin, 9 three-model), with 0/18 excluding the truth. In the 40-model test, 32/40 models are named in both sessions (65/80 rows) with zero wrong names. Excluding `err`, `image` and `effort_none` gives 29/40 (58/80), 0 wrong; MiniMax M2.5, MiniMax M2.7 and GLM-5.3 FlashX are no longer named (see `trials/G21/ablate_gateway.py`). Certification on and off produce identical results. All identify results use OpenRouter with pinned providers; transfer to other gateways or resellers is untested.

## 1. Problem and threat model
- **Verify:** an endpoint E claims model M. Is it M?
- **Identify:** E serves an unknown model (a stealth model). Which lab made it, and ideally which model is it?

All `identify` results reported here are from OpenRouter with pinned providers; transfer to other gateways or resellers is untested. OpenRouter and a peer-to-peer reseller marketplace are test sources, not runtime dependencies of the identifier.

Adversaries, in increasing strength:

| Class | Behaviour | What it can forge |
|---|---|---|
| Honest reseller | forwards requests unchanged | nothing |
| Lazy substituter | rewrites `model` to a cheaper sibling | the label only |
| Adaptive substituter | also rewrites error texts, strips parameters, zeroes usage fields, throttles speed | API surface, accounting, speed (downward only) |
| Stealth host | the lab serves its own unreleased model under an alias | name only; the stack is genuine |

We assume the auditor pays for normal API access and has trusted reference paths: first-party providers where available, or a fixed common provider when the lab has no usable first-party route. The main target surface is an OpenAI-compatible Chat Completions endpoint; unsupported optional fields and missing usage cause abstention, and native Anthropic/Gemini schemas still need adapters. OpenRouter and a peer-to-peer reseller marketplace are test sources, not runtime dependencies; all reported `identify` results use OpenRouter-pinned providers. We never forge or decrypt signatures. Probes stay at concurrency ≤6 and only hit endpoints we pay for.

## 2. Related work and novelty
Most of this system applies published, community or vendor-documented methods; the differences below are narrow.

- **Text-behaviour fingerprints:** LLMmap (USENIX Sec '25), IRIS (arXiv:2607.20860), *Are You Getting What You Pay For?* (arXiv:2504.04715), AgentProv (arXiv:2609.00052), and BazaarLink's resale study use content and behaviour probes. The specific One-Token JSD test used here did not separate Opus 5.5 and Sonnet 5.5 (0.195); that is a result for this pair and test, not a general limit on text fingerprints.
- **One-Token JSD** (arXiv:2607.10252) provides the 10-prompt battery used in §3.5. BazaarLink LLMprobe-engine V3H samples answer distributions to separate siblings; our addition is its use as a gated, per-family stage in this pipeline.
- **MET**, *Model Equality Testing: Which Model Is This API Serving?* (arXiv:2410.20247), frames substitution as a two-sample test on output distributions; our stage 2 is a cheaper, per-family distribution check in the same spirit, and our title question is theirs.
- **RUT**, *Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test* (arXiv:2506.06975), detects substitution from outputs alone; it does not use cache or accounting fields.
- **Token Counts Are Not Model Lineage** (arXiv:2608.29930) shows token accounting does not establish lineage; our M001 result agrees, and our token-count layer is bounded accordingly.
- **KBF** (arXiv:2605.29524) uses a model's knowledge boundary as a fingerprint; our behaviour battery overlaps (hard questions, newest-iOS knowledge) and is integration, not a new method.
- **Shadow APIs**, *Real Money, Fake Models* (arXiv:2603.01919), measures deceptive model claims by real resellers; we contribute a cache-based substitution test, not a market measurement.
- **GateScope**, *Behavioral Consistency and Transparency Analysis on LLM API Gateways* (arXiv:2604.21083), detects model downgrading or switching and billing errors; our proxy checks (§3.1) are a small instance of that.
- **AdaptPrint** (arXiv:2608.22213) does response-adaptive open-set fingerprinting with rejection; our abstention is not new.
- **ModelPrint** (<https://github.com/unclecode/modelprint>) uses infrastructure probes. This report integrates stack-signature features rather than introducing them.
- **Veridrop** (<https://github.com/canarybyte/veridrop>) and [CheckFakeAPI](https://www.checkfakeapi.net/) use API response structure and identifiers; our stack signature overlaps with these signals.
- **Timing.** *LLMs Have Rhythm* (arXiv:2502.20589) classifies inter-token times. The speed-era API was withdrawn from the shipped tool and is reported only as history; *Time Will Tell* (arXiv:2412.15431) and *The Early Bird* (arXiv:2409.20002) study timing side channels for prompt content, not model identity.
- **Prompt-cache auditing.** Gu et al., *Auditing Prompt Caching in LLM APIs* (arXiv:2502.07776), infer model architecture from cache hits as well as studying cross-user sharing. CacheProbe (arXiv:2605.30613) detects cache sharing from metadata. CacheTracer (Ji et al., arXiv:2608.20732) uses a flood-then-prove cache-reach primitive across reseller endpoints for one fixed model. We apply this primitive across labels on one endpoint as a substitution test. It distinguishes deployments, not weights, and can miss substitutions when labels use separate upstream cache accounts.
- **Tokenizer and serving-stack fingerprints.** LM-Fingerprint (<https://gitlab.com/WattoCyber/lm-fingerprint>) documents token accounting, validation errors, role/limit behaviour, JSON/SSE shape and headers. iSimplifyMe/tokenizer-fingerprint is a community tokenizer probe. We integrate these bounded signals; TokenPrint (arXiv:2608.08139) instead compares top-k projections of late hidden states and requires white-box access. Finlayson et al. (arXiv:2403.09539) and *Evaluating and Recalibrating Probabilistic Classifiers from One-Bit Threshold Queries* (arXiv:2609.05125) use API `logit_bias` for token information. ModelPrint and Veridrop/CheckFakeAPI also provide stack-signature prior art; our implementation is integration, not a new signature family.
- **Reasoning replay.** `pangolinsec/thinking-replay` (<https://github.com/pangolinsec/thinking-replay>) measures whether OpenAI reasoning items are kept via `input_tokens` deltas. Panfilov et al., *Stealing Reasoning Traces* (arXiv:2608.09867), study replay; `aceaura/claude-relay-verifier` redirects to <https://github.com/lizongti/claude-relay-verifier>. Anthropic documents the preserved-thinking matrix and account binding. Our contribution is the GPT-6.x extension to Panfilov's Table 1, not replay itself.
- **Under-trained tokens (UTF).** These are behavioural; we do not use them. N001 is a narrow application of tokenizer and logit-bias primitives. Its broad provider-independence hypothesis failed (§5.5).

Contribution status:
- **C1 component.** Vendor-documented capability gates, applied and checked across proxies; not a new gate matrix.
- **C2.** New application of CacheTracer/CacheProbe cache reach to cross-label substitution, separating deployments rather than weights.
- **C3.** Replay and format-forensics methods are prior art; the GPT-6.x extension is an observation.
- **C4 component.** A billed-accounting habit, not a weights-level signal. The decode-speed estimator was withdrawn.
- **Evaluation.** Hash-frozen, pre-registered 40-model × 2-session blind evaluation with zero wrong names.
- **Negative results.** We report measured failures and limits rather than treating unsuccessful signals as contributions.

## 3. Method
### 3.1 Capability gates and error oracles (C1)
The provider's API front-end checks parameters against the model it routes to. The gate matrix is vendor-documented, not a discovery here. Anthropic's [thinking troubleshooting](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting) and [fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode) document the Claude gates; OpenAI's [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) documents the GPT `none` restrictions. We apply these probes and check them across proxy paths.

- **Claude**, on both OpenRouter and the native path:
  - `thinking.type=between_tools` is accepted only by Sonnet 5.5.
  - `thinking.type=disabled` is rejected by Opus 5.5 and Sonnet 5.5 with different native error text, and accepted by Opus 5, Sonnet 5 and Haiku 5.5.
  - `speed:"fast"` is echoed as `usage.speed=fast` only for Opus models.
  - Adaptive thinking adds +2 input tokens on the 5.5 generation and 0 on generation 5. This same-path difference is measured here.
  - Five references give pairwise-distinct vectors; 0 of 90 cells changed across repeats (T004).
- **GPT:** `reasoning.effort=none` is rejected only by gpt-6.1-sol and gpt-6-astra (G001), consistent with the OpenAI documentation.
- **Proxy check (T006, G001):** the peer-to-peer marketplace buyer proxy passes bytes unchanged. OpenRouter drops unsupported request parameters as documented; its `max_tokens` clamp and Responses-stream re-synthesis are undocumented. The documented gates survived both tested proxies.

### 3.2 Cross-label cache-transfer oracle (C2)
A provider cache exposes shared deployment state; it does not prove identical weights. The test applies the CacheTracer/CacheProbe cache-reach primitive across labels:
1. Send a unique ~2.2k-token prefix under the **claimed** label.
2. Re-send it under the same label as a positive control. It must report `cached_tokens>0`.
3. Send it under a **suspect** label on the same endpoint.

Verdicts (each repeated twice with a fresh prefix):
- **SHARED_DEPLOYMENT:** the suspect label reports cached tokens (any count above zero) in every repetition.
- **SEPARATE_DEPLOYMENTS:** the suspect label reports zero in every repetition.
- **NO_CACHE_SIGNAL:** a positive control misses, so the endpoint hides or disables caching and the oracle abstains.
- **INCONSISTENT:** the suspect label hits in some repetitions but not others.

The test uses 6 requests and reads accounting fields (`fpx.py cache-oracle`). Labels routed through separate upstream accounts can produce a false negative. Because SHARED only requires a positive count, an endpoint that adds its own cached text (for example a fixed system prompt) to every request can report cached tokens on a fresh prompt and produce a false SHARED; a SHARED verdict should be checked by sending a fresh, never-seen prompt under the suspect label first.

### 3.3 Reasoning-artifact forensics and replay (C3)
- **Format forensics.** OpenRouter's reasoning-token documentation lists `reasoning_details[].format` values. Community analysis identifies OpenAI encrypted reasoning as Fernet; we inspect only the public header and never decrypt it.
- **Replay.** Replay is established by `pangolinsec/thinking-replay` and Panfilov et al. (arXiv:2608.09867). Anthropic's [preserved-thinking documentation](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking) publishes the readability matrix. The observations below re-measure that behaviour and report an additional GPT result.
  - **Claude (T002):** Opus 5.5 keeps Sonnet 5.5's block; Sonnet 5.5 drops Opus 5.5's. The tested target columns are distinct. Anthropic documents binding for Sonnet/Haiku 5.5; we also observed cross-account 400s for Opus 5 blocks.
  - **GPT (G003):** blocks were kept only when both models were from the same generation (6.x or 5.6), identically on two paths. This extends Panfilov et al.'s Table 1 to GPT-6.x. Unlike the 5.6 targets, the 6.x targets dropped 5.6 traces; this reverse direction is the new observation.

### 3.4 Billed-accounting habit and withdrawn decode speed (C4)
The shipped tool retains one **billed-accounting habit**: reasoning tokens billed on a fixed prompt (numbers 1–150 as words, low effort, k=3). This is a gateway/provider accounting observation, not a weights-level property. A003 removed text-phase decode rate because it tracked deployment, load and time rather than model identity, could be faked downward, and caused wrong verifier labels. Historical speed-era results are not current tool results.

Without speed, the label layer plus habit leaves some siblings as a group (GPT Luna / 5.6 Terra / 5.6 Sol; Opus 5 / Sonnet 5 on native Anthropic paths that reject `speed:"fast"`). `verify` reports `CONSISTENT_WITH_GROUP`; replay and cache tests are separate tools, not layers that `verify` combines. Re-scored recorded data: 0 of 54 verify labels exclude the true model.

### 3.5 Identification of unknown models
`fpx.py identify` compares an endpoint with saved references (`references.json`, 40 models, 12 companies (13 lab slugs: meta and meta-llama are both Meta)). It does not call the reference models again. All reported identify results use OpenRouter with pinned providers; transfer to other gateways or resellers is untested. Stage 1 sends 42 requests:
- 8 tokenizer probes (prompt-token deltas against "ok");
- a **serving-stack signature** using template token deltas, tool-call ID grammar, upstream error text, finish reason, `system_fingerprint` format and two status-only option checks. These are established signature features (LM-Fingerprint, ModelPrint, Veridrop/CheckFakeAPI), integrated here; model names are stripped from errors;
- reasoning-artifact format and whether reasoning comes back as a separate part;
- a **temperature-0 behaviour battery** of 26 short items, including exact echo, formats, refusals, capability questions, hard questions, newest-iOS knowledge, and per-question reasoning shape/length.

**Stage 2 (one-word distributions).** The 10 prompts are One Token Is Enough's battery (arXiv:2607.10252). BazaarLink LLMprobe-engine V3H is prior art for temperature-1 answer distributions, sibling separation, abstention and never crossing families. Here the tool asks each prompt 10 times and compares answer mixes with saved mixes (28 models); a mix counts only when closer to one sibling by at least `DIST_MARGIN=0.10`. On calibration comparisons, 276 comparisons are leave-one-set-out within the reference sets, not fresh held-out data: 228 credits, 48 below margin, 0 false credits.

**Naming rule.** A model is named only when the winner leads every sibling in at least two independent evidence blocks. Single-change certification is a redundant safety check: it withholds a name if a single evidence item changes the result. On G21, certification on and off are identical (32/40, 65/80, 0 wrong; see `trials/G21/ablate_gateway.py`). Otherwise the tool returns the lab or `unknown`; open-set rejection and abstention are established methods.

### 3.6 Optional sparse token-ID namespace check
`fpx.py token-id` sends a frozen sparse set of token IDs through `logit_bias:+100`, plus a `token_id=32, logit_bias:-100` negative control, in two repeats. A route is eligible only when the forced token is visible and stable, the forbidden-token response differs, status acceptance is stable, and both the acceptance vector and visible token echoes exactly match one frozen entry. Collision, instability, ignored bias or missing rows returns `ABSTAIN`; a stable unmatched vector returns `INCOMPATIBLE_WITH_ENROLLED`.

This is deliberately not contribution C5. It measures the token-ID mapping exposed by an API adapter. It cannot establish weights, provider, lineage, family or exact model, and is correlated with the prompt-token layer rather than an independent vote.

### 3.7 Layer agreement
`fpx.py verify` reports a label only when the API-gate layer and billed-accounting habit agree. `fpx.py audit` is more conservative: at least three non-contradictory evidence groups produce `CONSISTENT_WITH`; two contradictions produce `CONTRADICTED`; every other combination is `INDETERMINATE`. These are scoped observations, not proof of immutable weights. `cache-oracle` reports shared or separate cache domains rather than model identity. S001 was pre-registered but has no reported evaluation.

## 4. Experimental protocol
- Pre-registration applies only to trials with an entry in `ledger/trials.jsonl`; G16/G18/G20/G21 are the strongest examples because their freeze files predate the probes. G001–G006, T006, the A/R analyses, and G19a/b were not pre-registered. M007 and G20a entries have no threshold. S001 was pre-registered but was never reported.
- Main T/G/M held-out sessions were collected at least 5 minutes later. N001 was a shorter stability check: session 2 began about 35 seconds after session 1 ended, and the 480 calls spanned 4 minutes 21 seconds.
- The G16, G18, G20 and G21 attempts create optional-stopping risk across four tries; the sequence was G16 FAIL, G18 PASS later invalidated by requested-label echo (A005), G20 FAIL and G21 PASS.
- Collection used scripted subagents under a fixed contract: write only to `trials/T###/`, make no parameter changes, and report anomalies verbatim.
- Historical jitter analyses re-scored recorded streams with injected delays and coalescing. Speed-era jitter findings are not results for the shipped tool.
- **Paths:** OpenRouter pinned to first party; a peer-to-peer reseller marketplace through Reseller R; and a local deliberately dishonest test proxy (`trials/G011/swapproxy.py`).

## 5. Results
### 5.1 Claude exact-model (T007)
The shipped verifier scores Claude on OpenRouter **20/20**. Reseller R's non-suspect labels score **4/12 exact and 8/12 at the correct group**. Its suspect label was labelled Opus 5.5 in **4/4** sessions; this is the verdict, not an accuracy score. The prior 19/20 and 16/16 figures came from the withdrawn speed-based version and are historical, not current results. Claude verification uses 7 requests.

### 5.2 GPT exact-model (G009)
The shipped verifier scores **3/18 exact plus 15/18 correct group** (6 twin-class results and 9 three-model group results), with **0/18 excluding the truth**. Historical speed-era 17/18 results are withdrawn and not produced by the shipped tool. GPT-6 Astra and GPT-6.1 Sol remain indistinguishable in these probes; cache results distinguish deployments, not weights. A native-API `minimal` probe may distinguish them: OpenAI documents different support, while both accepted `eff_minimal` through OpenRouter in G11.

### 5.3 Swap detection with the cache oracle (G010, G011, G013)
Real endpoint results include separate cache deployments on tested OpenRouter labels and cache abstention on Reseller R's GPT path. G010 found Reseller R's Claude Opus and Sonnet labels in separate cache deployments: controls reported 3963–3965 cached tokens, while the cross-label probe missed.

| Simulated-reseller verdict | Result |
|---|---|
| Correct verdicts | 10/10 (5/5 swaps caught, 5/5 honest cleared) |
| Honest runs | 16, with 0 false positives |

The local swap proxy is a positive control, not evidence that every reseller uses a shared cache account.

### 5.4 Lab identification of unknown models (M001–M007)
M001 showed token-count instability across routes and tokenizer generations. M003's 30 probes were in-sample for selection of the open-set rule, and the classifier at that time still used decode speed. The apparently fresh M004/M005 probes are reported separately: 10 usable probes, 0 wrong, giving a one-sided 95% upper error bound of 25.9%.

Later negatives matter: M006 LOMO was 83.9%, below the 90% threshold; M006/M007 LOLO produced two wrong family assignments (Muse Glimmer → Llama). The historical M003–M005 aggregate is not a fresh holdout or a shipped-tool result.

The early M001 failure and the later LOMO/LOLO results show that tokenizer and serving-stack matches are bounded signals, not guarantees of lab or model identity.

### 5.5 Sparse token-ID namespace oracle (N001)
N001 selected 15 token IDs around seven public tokenizer-size regions. Both sessions covered all 15 routes; after session 1, the registry was frozen from the seven designated enrollment routes before session 2 began. Session 2 began about 35 seconds after session 1 ended; the complete 480-request run spanned 4 minutes 21 seconds. Summed response billing was **$0.001540174** (≈$0.0015); the key-usage counter initially lagged by about 8%.

The concrete signal is an API-visible sparse token map, not TLS:

| Token ID | GPT-3.5 / OpenAI (unshipped cl100k observation) | GPT-4o-mini / OpenAI (o200k entry) |
|---:|---|---|
| 32 | `200`, echo `A` | `200`, echo `A` |
| 100276 | `400` | `200`, echo `gard` |
| 154880 | `400` | `200`, echo `Zi` |
| 200018 | `400` | `400` |

These are selected cells, not a claim that the highest accepted ID is vocabulary size; providers can reserve, mask or remap IDs. The cl100k row is not safe to name: a smaller vocabulary can also accept only ID 32 and decode it as `A`.

| N001 outcome | Result |
|---|---:|
| Intended namespaces / panel routes | 7 / 15 |
| Eligible primary-analysis routes | 4 / 15 |
| Corrected nameable-route matches | 1 same-route o200k retest + 2 cross-route o200k holdouts |
| Shipped nameable namespaces | 1 (`openai-o200k`) |
| Cross-route o200k holdouts | 2 / 2 matched |
| Provider controls | OpenAI/Azure `PASS`; Llama `ABSTAIN` |

Treating the two o200k holdouts as independent gives a 77.6% one-sided 95% upper error bound, too weak for an error-rate claim. Eleven of fifteen routes were ineligible. The test is an optional compatibility oracle, not a general family classifier.

Transport capture did not produce another identity layer. TLS terminates at the client-to-gateway CDN; eBPF or `SSLKEYLOGFILE` on our host cannot observe the gateway-to-provider hop.

### 5.6 Case study: Reseller R
| Label | Assessment | Evidence |
|---|---|---|
| claude-opus-5-5 | consistent with Opus 5.5 | Opus-specific gate errors; upstream error name; replay asymmetry; billed-accounting habit. |
| gpt-6.1-sol, gpt-6-astra | consistent with the labels; pair not separable | one-token JSD and reasoning-replay generation match |
| gpt-6-sol | consistent with | JSD 0.077 vs next 0.156 |
| gpt-6-luna | consistent with | JSD 0.085 vs next 0.102 |
| gpt-5.6-sol | consistent with | JSD 0.090 vs next 0.145 |
| gpt-5.6-terra | consistent with | JSD 0.060 vs next 0.360 |

Reseller R's Claude path discards client system prompts, confounding the original JSD comparison. Its Codex path caps decode and reports no prompt caching; neither signal establishes GPT identity there.

### 5.7 Identify on 40 models: G16–G21
The pre-registered threshold was **at least 32/40 models named in both sessions with 0 wrong names**. G16/G18/G20/G21 have freeze files predating the probes. Earlier exploratory and post-hoc analyses are not described as pre-registered.

| Test | Result |
|---|---|
| G16 | 25/40 in both sessions; 51/80 rows; 0 wrong; FAIL |
| G18 | 34/40 in both sessions; 70/80 rows; 0 wrong; PASS, invalidated by requested-label echo (A005) |
| G19a/b | exploratory hardening/reversion; not pre-registered |
| G20 | 21/40 in both sessions; 44/80 rows; 0 wrong; FAIL |
| G21 | 32/40 in both sessions; 65/80 rows; 0 wrong; nominal PASS |

G21's natural result names 32/40 with 65/80 rows and 0 wrong. Dropping `err`, `image` and `effort_none` from both references and probes gives **29/40 (58/80), 0 wrong**; MiniMax M2.5, MiniMax M2.7 and GLM-5.3 FlashX are no longer named. The ablation script is `trials/G21/ablate_gateway.py`. Certification on and off produce an identical natural result.

G21 was the fourth pre-registered attempt at the same threshold (G16, G18, G20, G21), with the tool changed between attempts. This is optional stopping: the single nominal pass at exactly the threshold should be read with that in mind. An exploratory re-score of all 320 G16–G21 blind rows with the current tool gave 0 wrong names; this was not a fresh pre-registered holdout.

Among 33 models named at least once, 0 wrong gives an 8.7% one-sided 95% upper bound. This applies only to enrolled models on cooperative OpenRouter routes pinned to first-party providers.

### 5.8 Main negative results
| Negative result | Quantified observation |
|---|---|
| L001 speed | GPT verify gave 4/18 wrong labels at 0.8× and 6/18 at 0.6–0.7×; native-path Reseller R Sonnet 5 was labelled Opus 5 in all 4/4 sessions at 0.9× slowdown or slower. |
| M001 token-count lineage | Default routing yielded 90% lab and 55% LOMO; host template variance and tokenizer drift made counts insufficient for lineage. |
| Knowledge-question drift | Temperature-0 GPT knowledge answers wobble across sessions; only `k_ios` was stable and separating in G15. |
| Hard questions | G13 found 0 answer differences: siblings answered every hard question correctly. |
| N001 token namespace | Only 4/15 routes were eligible; 11/15 were ineligible, and 2 cross-route holdouts are too few for a useful error bound (77.6%). |

### 5.9 Field note: marketplace sellers (R002, R003)
An exploratory, non-pre-registered probe of cache and billing fields on a peer-to-peer marketplace; it is not an evaluation of `identify`. We sent ordinary paid requests to 8 anonymised sellers listing both Claude Opus 5.5 and Sonnet 5.5; 3 responded. Requests used the sellers' OpenAI chat-completions interface, and all figures are the sellers' own reported usage fields. Data and summaries: `trials/R002/`, `trials/R003/`.

**Seller-injected preambles confound the cache oracle.** On seller S2 the oracle returned `SHARED_DEPLOYMENT` for both Opus 5.5 vs Sonnet 5.5 and GPT-6.1 Sol vs GPT-6 Luna. A negative-control check with fresh, never-seen prompts showed why: they already carried cached tokens (GPT-6 Luna: a one-line prompt billed as 9,061 prompt tokens, 8,665 cached). S2 prepends its own fixed text to every request, and that text stays cached. Under the §3.2 rule this yields SHARED regardless of substitution, so these verdicts are not counted as evidence. Subscription resellers must send the official client's instructions and many relays add a fixed system prompt, so this confound is likely common; the fresh-prompt check in §3.2 is required before trusting a SHARED verdict on third-party sellers.

**Billed token counts reveal a tokenizer mismatch.** The same long text was billed as:

| Route and label | Prompt tokens |
|---|---:|
| OpenRouter, Claude Sonnet 5.5 | 3,938 |
| S2, `claude-opus-5.5` | 3,946 / 3,947 |
| S2, `claude-sonnet-5.5` | 2,333 / 2,335 |
| OpenRouter, GPT-6 Luna | 2,281 |

S2's Opus label matches Claude; its Sonnet label is 41% lower and matches a GPT-family tokenizer. Either a non-Claude model serves that label or its billing is falsified. Token counts can be forged, so this does not name the model, but comparing billed prompt tokens for identical text under two labels is a cheap substitution check that works even where the cache signal is confounded.

## 6. Limitations and evasion
- **The gates read the routing label.** A substituter that also strips parameters and forges errors can defeat C1.
- **The billed-accounting habit is not weights-level.** A gateway can alter or emulate usage accounting. `verify` does not run replay; no evidence supports claiming a gate/replay disagreement was tested there.
- **Provider-configurable reasoning items.** The `rt_*` and `think_*` observations depend on provider settings. Enabling reasoning for Sonnet 5.5 made both G21 sessions read as Opus 5.5 in all eight transformed variants; use `verify` and its gates for this pair.
- **Timing is not identity.** Decode speed was removed because it tracks host, load and time and can be throttled. No shipped decision uses timing.
- **The cache oracle needs cache accounting.** Zeroing `cached_tokens` blinds it; separate upstream cache accounts can also hide substitution. The timing twin (G012) failed because its saving was smaller than queueing noise.
- **The cache oracle distinguishes shared deployment state, not shared weights.**
- **Unseen-model identification** requires a sibling model of that lab in the database. Tokenizer drift across generations breaks it.
- **The token-ID oracle is optional and narrow.** Eleven of fifteen routes were ineligible; a gateway can sanitize or emulate `logit_bias`.
- **TLS/eBPF sees the wrong boundary.** Client-side capture can fingerprint the CDN/gateway but cannot prove model weights.
- **Coverage** is limited to enrolled models and tested paths; unenrolled labs return `unknown`. Native non-OpenAI schemas still need adapters.
- **Identify is fingerprinting, not authentication.** A red team showed rewritten metadata or multiple simultaneous changes can fool the tool. Hardening reduced coverage and was reverted.
- **OpenRouter scope.** All reported `identify` results are on OpenRouter with pinned providers; transfer to other gateways or resellers is untested.
- **References drift.** Providers change software, billing and templates; references require periodic re-measurement.
- **Budget:** about $30 total API spend. Sample sizes are small, so uncertainty remains.

## 7. Ethics
- We paid for every request, kept concurrency ≤6, never forged, decrypted or tampered with signatures or payments, and read only public token headers.
- Swap tests used our own local proxy. The marketplace field note (§5.9) used ordinary paid requests to anonymised sellers; seller names are kept out of the repository.
- Reseller R's resale paths may breach provider terms. We report them, not exploit them.

## 8. Reproducibility
The following commands run the current tools; they do not reproduce the historical speed-era numbers or every figure in §5:
```
python3 fpx.py verify <path> opus-5.5                    # Claude/GPT verify; GPT uses 4 requests
python3 fpx.py cache-oracle <path> opus-5.5 sonnet-5.5   # 6 requests
python3 fpx.py identify references.json <model-id> --base https://target.example/v1 --key-env SUSPECT_KEY   # 42 requests (+100 stage 2)
python3 fpx.py build references.json trials/G9/raw/ref.jsonl trials/G14/raw/behav.jsonl trials/G15/raw/behav.jsonl trials/G17/raw/pilot.jsonl trials/G17/raw/gpt.jsonl trials/G17/raw/more.jsonl
python3 fpx.py token-id token_id_references.json <model-id> --base https://target.example/v1 --key-env SUSPECT_KEY
python3 fp2.py score trials/T007/raw/or.jsonl            # current output: 20/20; no jitter replay
python3 fp2.py score trials/G009/raw/or.jsonl            # shipped-tool score, not historical speed-era matrix
python3 fpx.py evaluate references.json trials/M003/raw/heldout.jsonl trials/M004/raw/heldout.jsonl trials/M005/raw/heldout.jsonl
python3 trials/G21/ablate_gateway.py
```
The current `evaluate` command prints lab 100%, exact 100% precision at 55% coverage, LOMO 31/38 (81.6%), and LOLO 14 correct / 26 unknown / 0 wrong; it does not reproduce historical §5.4 claims. `docs/make_figures.py` targets the removed speed API and does not run against current `fp2.py`. The public `fpx.py` differs from the hash in `trials/G21/freeze.sha256` by one usage-docstring line (reseller name removed, ledger entry S-SCRUB) and a guard for error bodies returned as objects; `references.json` is byte-identical, and the ablation script reproduces the G21 numbers.

**Spend:** about $30 total. Stage 1 median cost was $0.013, maximum $0.12; average including stage 2 was about $0.04.

Dead ends with causes are listed in `ledger/INDEX.md` and `ledger/ideas.md`. They include prompt-cache minimum length, SSE bytes-per-event, prefill slope, logprobs, GPT tokenizer/server defaults, the cache timing twin, and broad token-ID namespace generalization.
