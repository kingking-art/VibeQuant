"""Thin adapter over akquant.

This is the ONLY module in VibeQuant that imports akquant. Everything
above it speaks in TaskSpec / signal functions / plain dataclasses, so
akquant stays untouched and swappable.

The adapter:
  1. wraps a pure-Python signal function into a generic akquant Strategy,
  2. calls akquant.run_backtest with cost/risk settings from the DSL,
  3. flattens the result into an engine-agnostic BacktestOutput.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from html import escape as esc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import akquant as aq

from ..dsl import TaskSpec
from ..strategies import SignalFn, build_signal, warmup_bars


@dataclass
class BacktestOutput:
    """Engine-agnostic backtest result consumed by report/risk/memory."""

    metrics: Dict[str, Optional[float]]
    equity_curve: pd.Series  # index: datetime, values: total equity
    trades: pd.DataFrame
    num_trades: int
    initial_cash: float
    engine: str = "akquant"
    engine_version: str = ""
    raw: Any = field(default=None, repr=False)  # akquant BacktestResult
    # factor_rotation only: per-rebalance scores (date -> {symbol: score}) and
    # the top_k used. None for non-rotation strategies. Reports use these to
    # render "which symbols were picked on which day" (akquant's native
    # report does not include this section).
    rotation_scores: Optional[Dict[str, Dict[str, float]]] = None
    rotation_top_k: Optional[int] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "initial_cash": self.initial_cash,
            "num_trades": self.num_trades,
            "metrics": self.metrics,
        }


class _SignalStrategy(aq.Strategy):
    """Generic bridge: rolling closes -> signal fn -> target-percent orders."""

    def __init__(
        self,
        signal_fn: SignalFn = None,  # type: ignore[assignment]
        max_position_pct: float = 0.95,
        history_cap: int = 512,
    ) -> None:
        super().__init__()
        self._signal_fn = signal_fn
        self._max_position_pct = max_position_pct
        self._history_cap = history_cap
        self._closes: Dict[str, List[float]] = {}

    def on_bar(self, bar: Any) -> None:  # noqa: D102
        symbol = bar.symbol
        closes = self._closes.setdefault(symbol, [])
        closes.append(float(bar.close))
        if len(closes) > self._history_cap:
            del closes[: len(closes) - self._history_cap]

        position = float(self.get_position(symbol))
        target = self._signal_fn(closes, position)
        if target is None:
            return
        target = max(0.0, min(float(target), 1.0)) * self._max_position_pct
        if target == 0.0 and position <= 0:
            return
        self.order_target_percent(target_percent=target, symbol=symbol)


class _RotationStrategy(aq.Strategy):
    """Cross-sectional rotation: hold the top-K symbols by factor score.

    Scores are precomputed from the same bars the engine replays (factor
    value at date t uses data up to t's close; orders fill at the next
    open, so there is no lookahead). Rebalances every `rebalance_days`
    trading days.
    """

    def __init__(
        self,
        scores: Dict[str, Dict[str, float]] = None,  # date -> {symbol: score}
        top_k: int = 5,
        rebalance_days: int = 5,
        max_position_pct: float = 0.95,
    ) -> None:
        super().__init__()
        self._scores = scores or {}
        self._top_k = max(1, int(top_k))
        self._rebalance_days = max(1, int(rebalance_days))
        self._max_position_pct = max_position_pct
        self._day_count = -1
        self._pending_target: Optional[Dict[str, float]] = None
        self._held: set = set()

    def on_daily_rebalance(self, trading_date, timestamp) -> None:  # noqa: D102
        # phase 2: entries queued at the previous session — exit proceeds
        # have settled by now, so buys cannot be rejected for cash
        if self._pending_target is not None:
            self.order_target_weights(
                target_weights=self._pending_target,
                liquidate_unmentioned=False,
                rebalance_tolerance=0.01,
            )
            self._held = set(self._pending_target)
            self._pending_target = None

        self._day_count += 1
        if self._day_count % self._rebalance_days:
            return
        key = str(trading_date)[:10]
        day_scores = self._scores.get(key) or {}
        if len(day_scores) < self._top_k:
            return  # warmup: not enough valid scores yet
        ranked = sorted(day_scores, key=day_scores.get, reverse=True)
        weight = self._max_position_pct / self._top_k
        target = {s: weight for s in ranked[: self._top_k]}

        # phase 1: exit names leaving the portfolio today; enter tomorrow
        # (two-phase rebalance, as live rotation desks do)
        for symbol in self._held - set(target):
            if float(self.get_position(symbol)) > 0:
                self.close_position(symbol)
        self._pending_target = target


def _rotation_scores(
    expressions: List[str], data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    """Precompute per-date combined factor scores (z-scored mean)."""
    from . import akquant_factor

    panel = akquant_factor.compute_factors(expressions, data)
    names = [
        akquant_factor.split_named_expression(raw, i)[0]
        for i, raw in enumerate(expressions)
    ]
    for name in names:  # per-date z-score, then average across factors
        grouped = panel.groupby("date")[name]
        std = grouped.transform("std").replace(0.0, pd.NA)
        panel[name] = (panel[name] - grouped.transform("mean")) / std
    panel["_score"] = panel[names].mean(axis=1)

    scores: Dict[str, Dict[str, float]] = {}
    for (date, symbol), value in panel.set_index(["date", "symbol"])["_score"].items():
        if pd.isna(value):
            continue
        scores.setdefault(str(date)[:10], {})[symbol] = float(value)
    return scores


ROTATION_DEFAULTS = {
    "expressions": ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"],
    "top_k": 5,
    "rebalance_days": 5,
}


def _metric(metrics: Any, name: str) -> Optional[float]:
    try:
        value = getattr(metrics, name)
    except AttributeError:
        return None
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def run_backtest(spec: TaskSpec, data: Dict[str, pd.DataFrame]) -> BacktestOutput:
    """Run one backtest for a TaskSpec over pre-loaded per-symbol bars."""
    symbols = list(data.keys())

    rotation_scores: Optional[Dict[str, Dict[str, float]]] = None
    rotation_top_k: Optional[int] = None
    if spec.strategy.name == "factor_rotation":
        params = {**ROTATION_DEFAULTS, **spec.strategy.params}
        rotation_scores = _rotation_scores(
            [str(e) for e in params["expressions"]], data
        )
        rotation_top_k = int(params["top_k"])
        strategy: Any = _RotationStrategy(
            scores=rotation_scores,
            top_k=rotation_top_k,
            rebalance_days=int(params["rebalance_days"]),
            max_position_pct=spec.risk.max_position_pct,
        )
    else:
        signal_fn = build_signal(spec.strategy.name, spec.strategy.params)
        warmup = warmup_bars(spec.strategy.name, spec.strategy.params)
        strategy = _SignalStrategy(
            signal_fn=signal_fn,
            max_position_pct=spec.risk.max_position_pct / max(len(symbols), 1),
            history_cap=max(warmup + 64, 256),
        )

    kwargs: Dict[str, Any] = dict(
        data=data if len(symbols) > 1 else next(iter(data.values())),
        strategy=strategy,
        symbols=symbols if len(symbols) > 1 else symbols[0],
        initial_cash=spec.execution.initial_cash,
        commission_rate=spec.execution.commission_rate,
        stamp_tax_rate=spec.execution.stamp_tax_rate,
        slippage={
            "type": "percent",
            "value": spec.execution.slippage_bps / 10_000.0,
        },
        t_plus_one=spec.execution.t_plus_one,
        show_progress=False,
    )
    if spec.risk.max_order_value:
        kwargs["strategy_max_order_value"] = {
            "default": float(spec.risk.max_order_value)
        }

    result = aq.run_backtest(**kwargs)

    metrics = result.metrics
    trades = result.trades_df
    num_trades = 0 if trades is None or trades.empty else int(len(trades))

    flat = {
        "total_return_pct": _metric(metrics, "total_return_pct"),
        "annualized_return": _metric(metrics, "annualized_return"),
        "sharpe_ratio": _metric(metrics, "sharpe_ratio"),
        "sortino_ratio": _metric(metrics, "sortino_ratio"),
        "max_drawdown_pct": _metric(metrics, "max_drawdown_pct"),
        "win_rate": _metric(metrics, "win_rate"),
    }

    return BacktestOutput(
        metrics=flat,
        equity_curve=result.equity_curve,
        trades=trades if trades is not None else pd.DataFrame(),
        num_trades=num_trades,
        initial_cash=spec.execution.initial_cash,
        engine_version=getattr(aq, "__version__", ""),
        raw=result,
        rotation_scores=rotation_scores,
        rotation_top_k=rotation_top_k,
    )


def write_html_report(
    output: BacktestOutput,
    path: str,
    title: str = "VibeQuant Strategy Report",
    market_data: Optional[Dict[str, pd.DataFrame]] = None,
    benchmark: Optional[pd.Series] = None,
) -> Optional[str]:
    """Render akquant's native HTML report (plotly). Best-effort.

    With a benchmark return series the report adds the benchmark block
    (excess return, alpha/beta, information ratio, tracking error).

    For factor_rotation runs we additionally inject a "rotation timeline"
    section (heatmap of top-k picks per rebalance) before </body>, since
    akquant's native report does not surface the per-rebalance selection.
    """
    try:
        output.raw.report(
            title=title,
            filename=path,
            show=False,
            market_data=market_data,
            benchmark=benchmark,
        )
    except Exception:
        return None
    section = _build_rotation_html_section(output.rotation_scores, output.rotation_top_k)
    if section:
        try:
            html = Path(path).read_text(encoding="utf-8")
            marker = "</body>"
            if marker in html:
                Path(path).write_text(
                    html.replace(marker, section + marker, 1),
                    encoding="utf-8",
                )
        except Exception:
            pass  # ponytail: native report is still readable without the section
    return path


def _build_rotation_html_section(
    scores: Optional[Dict[str, Dict[str, float]]],
    top_k: Optional[int],
) -> str:
    """Build a Plotly heatmap + recent-rebalances table for factor_rotation.

    Returns "" if there is nothing meaningful to plot (non-rotation run, or
    fewer than `top_k` valid scores on every rebalance date). The heatmap
    uses the same plotly CDN already loaded by the akquant native report.
    """
    if not scores or not top_k:
        return ""
    rows: List[Tuple[str, List[str]]] = []
    all_symbols: set[str] = set()
    for date in sorted(scores):
        sym_scores = scores[date]
        if len(sym_scores) < top_k:
            continue
        ranked = sorted(sym_scores, key=sym_scores.get, reverse=True)[:top_k]
        rows.append((date, ranked))
        all_symbols.update(ranked)
    if not rows:
        return ""
    symbols_sorted = sorted(all_symbols)
    z = [
        [1 if s in picks else 0 for s in symbols_sorted]
        for _, picks in rows
    ]
    dates = [d for d, _ in rows]
    heatmap_data = {
        "data": [
            {
                "type": "heatmap",
                "x": symbols_sorted,
                "y": dates,
                "z": z,
                "colorscale": [[0, "#f0f0f0"], [1, "#2c3e50"]],
                "showscale": False,
                "hovertemplate": "%{y} · %{x}<extra>picked</extra>",
            }
        ],
        "layout": {
            "title": "Rotation timeline (top-{} per rebalance)".format(top_k),
            "xaxis": {"title": "symbol", "tickangle": -45},
            "yaxis": {
                "title": "rebalance date",
                "autorange": "reversed",
                "type": "category",
            },
            "height": max(360, 24 * len(rows) + 120),
            "margin": {"l": 110, "r": 20, "t": 60, "b": 100},
        },
    }
    heatmap_args = json.dumps([heatmap_data["data"]])
    heatmap_layout = json.dumps(heatmap_data["layout"])
    plotly_config = json.dumps({"displaylogo": False})
    recent = list(reversed(rows))[:20]
    table_rows = "".join(
        "<tr><td style='padding:6px 8px;border-bottom:1px solid #eee'>"
        "{d}</td><td style='padding:6px 8px;border-bottom:1px solid #eee'>"
        "{picks}</td></tr>".format(
            d=esc(d),
            picks=", ".join(esc(s) for s in picks),
        )
        for d, picks in recent
    )
    return (
        "<section style='max-width:1200px;margin:30px auto;"
        "font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif'>"
        "<h2 style='color:#2c3e50;border-bottom:2px solid #3498db;"
        "padding-bottom:8px'>轮动明细 (Rotation Timeline)</h2>"
        "<div id='vq-rotation-heatmap'></div>"
        "<script>Plotly.newPlot("
        "'vq-rotation-heatmap', {args}, {layout}, {config});</script>"
        "<h3 style='margin-top:30px;color:#2c3e50'>"
        "最近 20 次调仓 (Last 20 rebalances)</h3>"
        "<table style='border-collapse:collapse;width:100%;font-size:14px'>"
        "<thead><tr style='background:#2c3e50;color:#fff'>"
        "<th style='text-align:left;padding:8px'>日期</th>"
        "<th style='text-align:left;padding:8px'>持仓 (top-{k})</th>"
        "</tr></thead><tbody>{rows}</tbody></table>"
        "</section>"
    ).format(
        args=heatmap_args,
        layout=heatmap_layout,
        config=plotly_config,
        k=top_k,
        rows=table_rows,
    )
