"""Model pricing — the single source of truth for what a run cost.

Imported by `agent/pipeline.py` (costing a run as it finishes) and by
`metrics.py` (costing stored runs at report time), so the two cannot drift.

Rates are Anthropic first-party $/1M tokens. A model missing from the table
costs `None` — treated as free, which is right for local Ollama models and
wrong for everything else: 21 stored `claude-fable-5` runs were recorded at
$0 because the table only knew opus/sonnet/haiku. Add every hosted model you
actually run.
"""

from __future__ import annotations

# $/1M tokens (input, output). Matched by substring, IN ORDER, so more specific
# keys must come first: Sonnet 5 is $2/$10 while Sonnet 4.6 is $3/$15, and a
# bare "sonnet" key would silently price both at the 4.6 rate.
PRICING: dict[str, tuple[float, float]] = {
    # OpenRouter free models: an explicit $0.00, never None. They are hosted, so
    # "unpriced" would be the wrong record, and it must precede every other key
    # because a free id can contain any vendor or family name.
    ":free": (0.0, 0.0),
    "fable": (10.0, 50.0),
    "sonnet-5": (2.0, 10.0),
    "sonnet": (3.0, 15.0),   # 4.6 and earlier
    "opus": (5.0, 25.0),
    "haiku": (1.0, 5.0),
}

# Prompt-caching multipliers applied to the INPUT rate. Writing to the cache
# costs a premium; reading from it is the saving that makes caching worthwhile.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def rates_for(model: str) -> tuple[float, float] | None:
    """(input, output) $/1M for a model id, or None if it isn't priced."""
    for key, rates in PRICING.items():
        if key in (model or ""):
            return rates
    return None


def cost_usd(
    model: str,
    in_tok: int,
    out_tok: int,
    *,
    cache_write_tok: int = 0,
    cache_read_tok: int = 0,
) -> float | None:
    """USD cost for a run, or None for local/unknown (free) models.

    `in_tok` is the UNCACHED input count. With prompt caching on, the provider
    reports cache writes and reads as separate usage fields; passing them here
    keeps the total honest instead of billing everything at the full input rate
    (or, worse, appearing to cost almost nothing because the cached tokens were
    never counted at all).
    """
    rates = rates_for(model)
    if rates is None:
        return None
    pin, pout = rates
    total = (
        in_tok / 1e6 * pin
        + out_tok / 1e6 * pout
        + cache_write_tok / 1e6 * pin * CACHE_WRITE_MULTIPLIER
        + cache_read_tok / 1e6 * pin * CACHE_READ_MULTIPLIER
    )
    return round(total, 4)
