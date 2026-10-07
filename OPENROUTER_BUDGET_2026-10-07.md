# What $200 of OpenRouter credit buys — 2026-10-07

Planning note, not a result. Prices come from OpenRouter's live model list
(`https://openrouter.ai/api/v1/models`, fetched 2026-10-07). Token counts come
from our own runs in `results.jsonl`. Capability scores are mostly
vendor-reported and depend on the harness, so treat them as a filter, not a
ranking.

## Terms

- **Paired instance**: one SWE-bench Lite instance run through both arms.
  Arm A is the reactive fix agent with no BRT. Arm B is the test-first
  deployable pipeline (`07`): BRT phase, then fix phase. This is the unit of
  the RQ1 sweep and what `11_paired_sweep.py` / `16_nightly_batch.py` run.
- **Run**: in this note, a run means one paired instance. One Arm B run alone
  costs about 55% of a paired instance.
- **Sweep**: the full frozen RQ1 sample run once per arm. That is the base 55
  plus the 90-instance extension, so 145 paired instances.
- **Cache hit**: the agent re-sends its whole transcript every turn, so most
  of each request's input is a prefix the provider has already seen. Providers
  that cache bill that prefix at the lower "cache read" price.

## 1. What a run uses (measured)

From the 74 paired instances in `rq1-sweep`, `rq1-armA` and `rq1-ext`
(nemotron-3-super, free tier, no caching, 30-iteration cap):

| per paired instance | input tokens | output tokens |
|---|---|---|
| mean   | 532,320 | 90,438 |
| median | 499,628 | 75,217 |
| p90    | 854,492 | 156,788 |
| max    | 1,183,620 | 263,705 |

Split by arm (means): Arm B is about 123k + 203k input (BRT + fix) and 47k
output; Arm A is about 241k input and 42k output. A paired instance takes
about 48–79 model calls and 16–28 minutes of wall time.

Cost per paired instance:

    cost = (input_tokens × input_price + output_tokens × output_price) / 1,000,000

With caching, assuming 75% of input is read from cache:

    cost = (0.25 × in × in_price + 0.75 × in × cache_price + out × out_price) / 1e6

The 75% figure is conservative. Sonnet with explicit caching reached 90%.

**Caveat:** these are nemotron's token counts. nemotron writes long reasoning,
so its output is high. Sonnet used far fewer tokens on the same pipeline
(about 75k cache reads + 9k writes + 3k output per phase). Other models
will differ in both directions, so measure 5 instances per model before
scaling up (section 5).

## 2. Candidates and cost per paired instance

All of these support tool calling on OpenRouter. The table is sorted by cost
without caching. "Runs/$200" means paired instances per $200 at the mean.

| model | $/M in / out | $/M cache read | mean, no cache | mean, 75% cache | p90, no cache | runs/$200 no cache | runs/$200 cached | SWE-bench Verified |
|---|---|---|---|---|---|---|---|---|
| openai/gpt-oss-120b | 0.037 / 0.17 | — | $0.035 | $0.035 | $0.058 | 5,702 | 5,702 | ~62% (vendor) |
| nvidia/nemotron-3-super-120b-a12b (paid) | 0.08 / 0.45 | — | $0.083 | $0.083 | $0.139 | 2,401 | 2,401 | ~60% (NVIDIA, OpenHands) |
| anthropic/claude-haiku-5.5 | 0.10 / 0.50 | 0.01 | $0.098 | $0.063 | $0.164 | 2,031 | 3,199 | n/a (listed today; verify the price) |
| z-ai/glm-5.3-flash | 0.15 / 0.50 | 0.03 | $0.125 | $0.077 | $0.207 | 1,599 | 2,592 | n/a |
| deepseek/deepseek-v4-flash | 0.03 / 1.28 | 0.03 | $0.132 | $0.132 | $0.226 | 1,518 | 1,518 | 79% (vendor) |
| qwen/qwen3-coder-next | 0.12 / 0.80 | 0.07 | $0.136 | $0.116 | $0.228 | 1,468 | 1,720 | n/a |
| **deepseek/deepseek-v4-pro** | 0.209 / 0.418 | 0.017 | $0.149 | $0.072 | $0.244 | 1,341 | 2,762 | ~80% (vendor, "Max" setting) |
| deepseek/deepseek-v3.2 | 0.28 / 0.42 | 0.028 | $0.187 | $0.086 | $0.305 | 1,069 | 2,314 | ~73% (vendor) |
| minimax/minimax-m2.7 | 0.21 / 0.84 | 0.042 | $0.188 | $0.121 | $0.311 | 1,065 | 1,657 | n/a |
| **qwen/qwen3-coder (480B)** | 0.30 / 1.00 | 0.10 | $0.250 | $0.170 | $0.413 | 799 | 1,174 | 66.5% (independent) |
| minimax/minimax-m3 | 0.30 / 1.20 | 0.06 | $0.268 | $0.172 | $0.444 | 745 | 1,160 | 80.5% (vendor) |
| google/gemini-3.1-flash-lite | 0.25 / 1.50 | 0.025 | $0.269 | $0.179 | $0.449 | 744 | 1,117 | n/a |
| mistralai/devstral-2512 | 0.40 / 2.00 | 0.04 | $0.394 | $0.250 | $0.655 | 507 | 799 | n/a |
| **moonshotai/kimi-k2.6** | 0.465 / 2.45 | 0.098 | $0.469 | $0.323 | $0.781 | 426 | 619 | 80.2% (vendor) |
| openai/gpt-5.4-mini | 0.75 / 4.50 | 0.075 | $0.806 | $0.537 | $1.346 | 248 | 372 | n/a |
| anthropic/claude-sonnet-5.5 (reference only) | 2.00 / 10.00 | 0.20 | $1.969 | $1.250 | $3.277 | 101 | 159 | frontier |

Left out on purpose: the sub-30B models (qwen3-30b-a3b, gemma, ministral,
nemotron-nano and similar). Our 7B–14B local runs already showed that this
size cannot sustain the loop. Also left out: `~…-latest` aliases (the model can
change under a frozen sample), `:batch` variants (not interactive), and
anything above about $1/M input.

## 3. The main finding: money is not the limit, time is

A full 145-instance paired sweep costs:

| model | sweep, mean | sweep, ×1.5 safety margin |
|---|---|---|
| nemotron-3-super (paid) | $12 | $18 |
| deepseek-v4-pro | $22 (≈$10 cached) | $33 |
| qwen3-coder 480B | $36 | $54 |
| minimax-m3 | $39 | $58 |
| kimi-k2.6 | $68 | $102 |

So $200 covers 3–5 full sweeps on mid-tier models. What binds is wall time
and Docker:

- A paired instance takes about 16–28 minutes, so 145 of them take roughly
  40–65 hours per model if run one at a time.
- Each instance needs its prebuilt image pulled, gold-checked and deleted
  (`16_nightly_batch.py`), and disk only holds a few images at once.
- With the deadline around 2026-10-20 (13 days), that is realistically
  **2–3 full-sample model sweeps**, or more models on a smaller subset.

Paying removes the free-tier daily request cap (1,000/day, about 15–25
paired instances) and the upstream throttling we hit on `:free`. Those were
the actual bottlenecks.

## 4. Suggested plans for $200

**Plan 1 — same model, paid (cheapest, cleanest).** Run
`nemotron-3-super-120b-a12b` (paid, not `:free`) on the remaining extension
instances. It is the same model as the paper's main arm, so results pool
directly, and it ends the rate-limit waits. About $10 for the rest of the
sample, leaving about $190.

**Plan 2 — one stronger open-weight model on the full sample (RQ1 robustness
+ RQ3).** Add `deepseek-v4-pro` (or `qwen3-coder` 480B as a second choice) on
all 145 paired instances. That answers whether the RQ1 null holds with a
stronger model. About $22–35.

**Plan 3 — RQ3 capability ladder on a fixed subset.** Run 4–5 open-weight
models on the same 30–40 paired instances to report where each trajectory
breaks (submits? reproduces? resolves?):
gpt-oss-120b → nemotron-3-super → qwen3-coder → deepseek-v4-pro → kimi-k2.6.
40 instances × 5 models ≈ $40–45 mean, about $70 at p90.

**Recommended combination:** Plan 1 + Plan 2 + Plan 3, about $80–110
expected, with $90+ held back for re-runs, the determinism 3× check, and
models that turn out more verbose than nemotron.

## 5. Before spending anything (code and account)

1. **Pricing table.** `src/tdbf/pricing.py` only knows `:free` and Anthropic
   models. Every paid OpenRouter model would be logged at $0, the same bug as
   the 21 fable runs. Add each model's rates before its first run.
2. **Record what OpenRouter actually billed.** `OpenRouterClient` does not
   read `usage.prompt_tokens_details.cached_tokens` or `usage.cost`. Capture
   both, so cost comes from the bill and caching is visible, not estimated.
3. **Pin the provider.** Cheap listings are served by several providers, some
   quantized. Set `provider.order`/`quantizations` (or `allow_fallbacks:
   false`) so one model id means one deployment for the whole sample. We
   already record `served_by`. Keep doing it.
4. **Hard spend limit.** Put a credit limit on the API key in the OpenRouter
   dashboard (for example $60 per model key) so a runaway loop cannot drain
   the $200.
5. **5-instance probe per model.** Run `07` on 5 dev-set instances first and
   read the real tokens per paired instance and the submit rate. Drop any
   model that cannot reach `submit_*` (as qwen 7B–14B could not) before it
   enters the sample.
6. **Tag runs.** Use a new `--run-group` per model so `arm_of()` keeps them
   apart from the free nemotron runs.

## 6. Open questions for you

- Closed models (Haiku 5.5, Gemini Flash-Lite, GPT-5.4-mini) are cheap enough,
  but RQ3 is about *open-weight* models. Include one closed model as a
  reference point, or stay open-weight only?
- Haiku 5.5's listed $0.10/$0.50 is one tenth of Haiku 4.5's price and was
  published today. Confirm it before planning around it.
- Full-sample sweep with one stronger model (Plan 2) vs. more models on a
  subset (Plan 3): with about 13 days left there is time for roughly one of
  them at full size.

## Sources

- OpenRouter models API (prices, cache prices, tool support): https://openrouter.ai/api/v1/models
- OpenRouter prompt caching docs (`cached_tokens`, `cache_discount`): https://openrouter.ai/docs/features/prompt-caching
- Nemotron 3 Super SWE-bench Verified (60.47% OpenHands): https://docs.nvidia.com/nemotron/0.1.0/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.html
- Open-weight SWE-bench Verified roundup (DeepSeek V4 Pro, MiniMax M3, Kimi K2.6, Qwen3-Coder): https://www.morphllm.com/best-open-source-coding-model-2026 and https://www.morphllm.com/best-ai-model-for-coding
- SWE-bench Verified methodology: https://www.vals.ai/benchmarks/swebench
