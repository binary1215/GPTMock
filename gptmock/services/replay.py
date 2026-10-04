from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from gptmock.core.constants import SSE_OUTPUT_ITEM_DONE, SSE_RESPONSE_COMPLETED, SSE_RESPONSE_INCOMPLETE


@dataclass
class ReasoningReplay:
    """Keep opaque reasoning items, preferring the terminal response snapshot."""

    _done: list[tuple[int | None, dict[str, Any]]] = field(default_factory=list)
    _terminal: list[dict[str, Any]] | None = None

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind in (SSE_RESPONSE_COMPLETED, SSE_RESPONSE_INCOMPLETE):
            response = event.get("response")
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list):
                self._terminal = [
                    deepcopy(item) for item in output
                    if isinstance(item, dict) and item.get("type") == "reasoning"
                ]
            return
        if kind != SSE_OUTPUT_ITEM_DONE:
            return
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            return
        index = event.get("output_index")
        index = index if isinstance(index, int) and not isinstance(index, bool) else None
        item_id = item.get("id")
        for position, (previous_index, previous) in enumerate(self._done):
            same_index = index is not None and index == previous_index
            same_id = isinstance(item_id, str) and bool(item_id) and item_id == previous.get("id")
            same_anonymous = index is None and previous_index is None and item == previous
            if same_index or same_id or same_anonymous:
                self._done[position] = (index if index is not None else previous_index, deepcopy(item))
                return
        self._done.append((index, deepcopy(item)))

    def items(self) -> list[dict[str, Any]]:
        if self._terminal is not None:
            return deepcopy(self._terminal)
        ordered = sorted(
            enumerate(self._done),
            key=lambda entry: (
                entry[1][0] is None,
                entry[1][0] if entry[1][0] is not None else entry[0],
            ),
        )
        return [deepcopy(item) for _, (_, item) in ordered]
