import sys

import numpy as np
import polars as pl


BATCH_SIZE = 2_000_000
SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS

AggPart = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def masked_reduceat(
    values: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    scratch: np.ndarray,
) -> np.ndarray:
    scratch.fill(0.0)
    np.copyto(scratch, values, where=mask)
    return np.add.reduceat(scratch, starts)


def aggregate_bucket(
    bucket: np.ndarray,
    amount: np.ndarray,
    quantity: np.ndarray,
    is_sell: np.ndarray,
) -> AggPart:
    if len(bucket) == 0:
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty, empty, empty

    starts = np.concatenate(([0], np.flatnonzero(bucket[1:] != bucket[:-1]) + 1))
    keys = bucket[starts]
    buy_mask = ~is_sell
    scratch = np.empty_like(amount, dtype=np.float64)

    buy_amount = masked_reduceat(amount, buy_mask, starts, scratch)
    buy_qty = masked_reduceat(quantity, buy_mask, starts, scratch)
    sell_amount = masked_reduceat(amount, is_sell, starts, scratch)
    sell_qty = masked_reduceat(quantity, is_sell, starts, scratch)

    return keys, buy_amount, buy_qty, sell_amount, sell_qty


def combine_parts(parts: list[AggPart]) -> AggPart:
    if len(parts) == 1:
        return parts[0]

    keys = np.concatenate([part[0] for part in parts])
    buy_amount = np.concatenate([part[1] for part in parts])
    buy_qty = np.concatenate([part[2] for part in parts])
    sell_amount = np.concatenate([part[3] for part in parts])
    sell_qty = np.concatenate([part[4] for part in parts])

    starts = np.concatenate(([0], np.flatnonzero(keys[1:] != keys[:-1]) + 1))
    return (
        keys[starts],
        np.add.reduceat(buy_amount, starts),
        np.add.reduceat(buy_qty, starts),
        np.add.reduceat(sell_amount, starts),
        np.add.reduceat(sell_qty, starts),
    )


def forward_fill(values: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    if not np.any(valid):
        return values

    idx = np.where(valid, np.arange(len(values)), -1)
    np.maximum.accumulate(idx, out=idx)

    result = values.copy()
    filled = idx >= 0
    result[filled] = values[idx[filled]]
    return result


def kahan_sum(values: np.ndarray) -> float:
    total = 0.0
    compensation = 0.0

    for value in values:
        y = float(value) - compensation
        t = total + y
        compensation = (t - total) - y
        total = t

    return total


def calc_result(part: AggPart) -> float:
    keys, buy_amount, buy_qty, sell_amount, sell_qty = part

    if len(keys) == 0:
        return 0.0

    buy_vwap = np.full(len(keys), np.nan, dtype=np.float64)
    sell_vwap = np.full(len(keys), np.nan, dtype=np.float64)

    buy_known = buy_qty > 0
    sell_known = sell_qty > 0

    buy_vwap[buy_known] = buy_amount[buy_known] / buy_qty[buy_known]
    sell_vwap[sell_known] = sell_amount[sell_known] / sell_qty[sell_known]

    buy_vwap = forward_fill(buy_vwap)
    sell_vwap = forward_fill(sell_vwap)

    valid = ~np.isnan(buy_vwap) & ~np.isnan(sell_vwap)
    if not np.any(valid):
        return 0.0

    diff_abs = np.abs(buy_vwap[valid] - sell_vwap[valid])
    return kahan_sum(diff_abs)


def calc(path: str) -> tuple[float, float]:
    second_parts: list[AggPart] = []
    minute_parts: list[AggPart] = []

    batches = (
        pl.scan_parquet(path)
        .select(["timestamp", "price", "quantity", "is_buyer_maker"])
        .collect_batches(chunk_size=BATCH_SIZE, engine="streaming")
    )

    for batch in batches:
        timestamp = batch["timestamp"].dt.epoch(time_unit="ns").to_numpy()
        price = batch["price"].to_numpy()
        quantity = batch["quantity"].to_numpy()
        is_sell = batch["is_buyer_maker"].to_numpy()
        amount = price * quantity

        second_part = aggregate_bucket(timestamp // SECOND_NS, amount, quantity, is_sell)
        minute_part = aggregate_bucket(timestamp // MINUTE_NS, amount, quantity, is_sell)

        second_parts.append(second_part)
        minute_parts.append(minute_part)

    return calc_result(combine_parts(second_parts)), calc_result(combine_parts(minute_parts))


def main() -> None:
    path = sys.argv[1]
    vwap_s, vwap_m = calc(path)
    print(f"VWAP_s={vwap_s:.6f}")
    print(f"VWAP_m={vwap_m:.6f}")


if __name__ == "__main__":
    main()
