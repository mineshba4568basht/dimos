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

"""In-memory fan-out notifier (same-process, thread-safe)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from reactivex.disposable import Disposable

from dimos.memory2.registry import qual
from dimos.memory2.type.backend import Notifier

if TYPE_CHECKING:
    from reactivex.abc import DisposableBase

    from dimos.memory2.buffer import BackpressureBuffer
    from dimos.memory2.type.observation import Observation

T = TypeVar("T")


class SubjectNotifier(Notifier[T], Generic[T]):
    """In-memory fan-out notifier for same-process live notification.

    Thread-safe.  ``notify()`` copies the subscriber list under the lock,
    then iterates outside the lock to avoid deadlocks with slow consumers.
    """

    def __init__(self) -> None:
        self._subscribers: list[BackpressureBuffer[Observation[T]]] = []
        self._lock = threading.Lock()

    def subscribe(self, buf: BackpressureBuffer[Observation[T]]) -> DisposableBase:
        with self._lock:
            self._subscribers.append(buf)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(buf)
                except ValueError:
                    pass

        return Disposable(action=_unsubscribe)

    def notify(self, obs: Observation[T]) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for buf in subs:
            buf.put(obs)

    # ── Serialization ─────────────────────────────────────────────

    def serialize(self) -> dict[str, Any]:
        return {"class": qual(type(self)), "config": {}}

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SubjectNotifier[Any]:
        return cls()
