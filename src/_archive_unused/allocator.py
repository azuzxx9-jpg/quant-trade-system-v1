from typing import Dict, List, Tuple


def allocate_capital(
    ranked_symbols: List[Tuple[str, float]],
    sleeve: str,
    cfg: Dict,
) -> Dict[str, float]:
    """
    Allocate sleeve-level capital across the top ranked symbols.

    Returns a dict of:
        {symbol: weight_within_total_portfolio}

    Example:
        sleeve_weights = {"trend_long": 0.7, "breakout_long": 0.3}
        top_n for trend_long = 5

        -> each selected trend symbol gets 0.7 / 5 = 0.14
    """

    sleeve_weights = cfg.get("sleeve_weights", {})
    sleeve_weight = float(sleeve_weights.get(sleeve, 0.0))

    if sleeve_weight <= 0:
        return {}

    top_n_key = f"symbol_rank_top_n_{sleeve}"
    top_n = int(cfg.get(top_n_key, 0))

    if top_n <= 0:
        return {}

    selected = ranked_symbols[:top_n]
    if not selected:
        return {}

    equal_weight = sleeve_weight / len(selected)
    return {symbol: equal_weight for symbol, _score in selected}
