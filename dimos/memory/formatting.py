# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rich text rendering for memory types and streams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable

_console = Console(force_terminal=True, highlight=False)


def render_text(text: Text) -> str:
    """Render rich Text to a terminal string with ANSI codes."""
    with _console.capture() as cap:
        _console.print(text, end="", soft_wrap=True)
    return cap.get()


# ── Filter rendering ────────────────────────────────────────────────


def _after_rich(f: Any) -> Text:
    t = Text()
    t.append("after", style="cyan")
    t.append(f"(t={f.t})")
    return t


def _before_rich(f: Any) -> Text:
    t = Text()
    t.append("before", style="cyan")
    t.append(f"(t={f.t})")
    return t


def _time_range_rich(f: Any) -> Text:
    t = Text()
    t.append("time_range", style="cyan")
    t.append(f"({f.t1}, {f.t2})")
    return t


def _at_rich(f: Any) -> Text:
    t = Text()
    t.append("at", style="cyan")
    t.append(f"(t={f.t}, tol={f.tolerance})")
    return t


def _near_rich(f: Any) -> Text:
    t = Text()
    t.append("near", style="cyan")
    t.append(f"(radius={f.radius})")
    return t


def _tags_rich(f: Any) -> Text:
    t = Text()
    t.append("tags", style="cyan")
    pairs = ", ".join(f"{k}={v!r}" for k, v in f.tags)
    t.append(f"({pairs})")
    return t


def _embedding_search_rich(f: Any) -> Text:
    t = Text()
    t.append("search_embedding", style="cyan")
    t.append("(")
    if f.label:
        t.append(repr(f.label), style="green")
        t.append(", ")
    t.append(f"k={f.k}")
    t.append(")")
    return t


def _text_search_rich(f: Any) -> Text:
    t = Text()
    t.append("text", style="cyan")
    t.append(f"({f.text!r})")
    return t


def _lineage_rich(f: Any) -> Text:
    t = Text()
    t.append("lineage", style="cyan")
    hops = " -> ".join(f.hops) if f.hops else "direct"
    t.append(f"({f.source_table} -> {hops})")
    return t


_FILTER_DISPATCH: dict[type, Callable[..., Text]] | None = None


def _get_dispatch() -> dict[type, Callable[..., Text]]:
    global _FILTER_DISPATCH
    if _FILTER_DISPATCH is not None:
        return _FILTER_DISPATCH
    from dimos.memory.type import (
        AfterFilter,
        AtFilter,
        BeforeFilter,
        EmbeddingSearchFilter,
        LineageFilter,
        NearFilter,
        TagsFilter,
        TextSearchFilter,
        TimeRangeFilter,
    )

    _FILTER_DISPATCH = {
        AfterFilter: _after_rich,
        BeforeFilter: _before_rich,
        TimeRangeFilter: _time_range_rich,
        AtFilter: _at_rich,
        NearFilter: _near_rich,
        TagsFilter: _tags_rich,
        EmbeddingSearchFilter: _embedding_search_rich,
        TextSearchFilter: _text_search_rich,
        LineageFilter: _lineage_rich,
    }
    return _FILTER_DISPATCH


def filter_rich(f: Any) -> Text:
    """Render a Filter to rich Text."""
    dispatch = _get_dispatch()
    renderer = dispatch.get(type(f))
    if renderer is None:
        return Text(str(f))
    return renderer(f)


def query_rich(q: Any) -> Text:
    """Render a StreamQuery to rich Text."""
    t = Text()
    pipe = Text(" | ", style="dim")
    parts: list[Text] = [filter_rich(f) for f in q.filters]
    if q.order_field:
        p = Text()
        p.append("order", style="cyan")
        direction = "desc" if q.order_desc else "asc"
        p.append(f"({q.order_field}, {direction})")
        parts.append(p)
    if q.limit_val is not None:
        p = Text()
        p.append("limit", style="cyan")
        p.append(f"({q.limit_val})")
        parts.append(p)
    if q.offset_val is not None:
        p = Text()
        p.append("offset", style="cyan")
        p.append(f"({q.offset_val})")
        parts.append(p)
    for i, part in enumerate(parts):
        if i > 0:
            t.append_text(pipe)
        t.append_text(part)
    return t


# ── Stream rendering ────────────────────────────────────────────────


def rich_text(obj: Any) -> Text:
    """Render a Stream, TransformStream, ObservationSet, or StreamQuery to rich Text.

    Uses duck-typing on attributes — no dispatch table needed.
    """
    # TransformStream: has _source and _transformer
    if hasattr(obj, "_transformer"):
        xf = obj._transformer
        t = Text()
        t.append("TransformStream", style="bold cyan")
        t.append("[", style="dim")
        t.append(xf.output_type.__name__ if xf.output_type else "?", style="yellow")
        t.append("]", style="dim")
        t.append("(", style="dim")
        t.append_text(rich_text(obj._source))
        t.append(" -> ", style="dim")
        t.append(type(xf).__name__, style="magenta")
        if obj._live:
            t.append(", ", style="dim")
            t.append("live=True", style="yellow")
        if obj._backfill_only:
            t.append(", ", style="dim")
            t.append("backfill_only=True", style="yellow")
        t.append(")", style="dim")
        qt = query_rich(obj._query)
        if qt.plain:
            t.append(" | ", style="dim")
            t.append_text(qt)
        return t

    # ObservationSet: has _observations list
    if hasattr(obj, "_observations"):
        type_name = obj._payload_type.__name__ if obj._payload_type else "?"
        t = Text()
        t.append("ObservationSet", style="bold cyan")
        t.append("[", style="dim")
        t.append(type_name, style="yellow")
        t.append("]", style="dim")
        t.append("(", style="dim")
        t.append(f"{len(obj._observations)} items", style="green")
        t.append(")", style="dim")
        return t

    # StreamQuery
    if hasattr(obj, "filters"):
        return query_rich(obj)

    # Stream (and subclasses like EmbeddingStream, TextStream)
    cls_name = type(obj).__name__
    type_name = obj._payload_type.__name__ if obj._payload_type else "?"
    name = obj._backend.stream_name if obj._backend else "unbound"
    t = Text()
    t.append(cls_name, style="bold cyan")
    t.append("[", style="dim")
    t.append(type_name, style="yellow")
    t.append("]", style="dim")
    t.append("(", style="dim")
    t.append(f'"{name}"', style="green")
    t.append(")", style="dim")
    qt = query_rich(obj._query)
    if qt.plain:
        t.append(" | ", style="dim")
        t.append_text(qt)
    return t
