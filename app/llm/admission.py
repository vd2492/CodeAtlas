"""Bounded, process-local admission control for expensive LLM pipelines.

CodeAtlas intentionally runs as one application process today.  A small FIFO
queue prevents a burst of long-running LLM calls from consuming every server
thread while leaving authentication, admin, and health-check traffic responsive.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


MAX_CONCURRENT_REQUESTS = max(
    1, int(os.environ.get("CODEATLAS_MAX_CONCURRENT_LLM_REQUESTS", "12"))
)
MAX_QUEUED_REQUESTS = max(
    0, int(os.environ.get("CODEATLAS_MAX_QUEUED_LLM_REQUESTS", "20"))
)
QUEUE_TIMEOUT_SECONDS = max(
    0.0, float(os.environ.get("CODEATLAS_LLM_QUEUE_TIMEOUT_SECONDS", "15"))
)


class LLMCapacityError(RuntimeError):
    """Raised when the bounded queue is full or a queued request times out."""


@dataclass(frozen=True)
class AdmissionSnapshot:
    active: int
    queued: int
    max_active: int
    max_queued: int


class LLMAdmissionController:
    """FIFO admission controller with a bounded queue and guaranteed release."""

    def __init__(
        self,
        max_active: int = MAX_CONCURRENT_REQUESTS,
        max_queued: int = MAX_QUEUED_REQUESTS,
        queue_timeout_seconds: float = QUEUE_TIMEOUT_SECONDS,
    ) -> None:
        self.max_active = max(1, int(max_active))
        self.max_queued = max(0, int(max_queued))
        self.queue_timeout_seconds = max(0.0, float(queue_timeout_seconds))
        self._active = 0
        self._waiters: deque[object] = deque()
        self._condition = threading.Condition()

    def snapshot(self) -> AdmissionSnapshot:
        with self._condition:
            return AdmissionSnapshot(
                active=self._active,
                queued=len(self._waiters),
                max_active=self.max_active,
                max_queued=self.max_queued,
            )

    def acquire(self) -> float:
        """Acquire an execution slot and return the number of seconds queued."""
        started_at = time.monotonic()
        with self._condition:
            if self._active < self.max_active and not self._waiters:
                self._active += 1
                return 0.0

            if len(self._waiters) >= self.max_queued:
                raise LLMCapacityError(
                    "The answer service is at capacity. Please retry shortly."
                )

            waiter = object()
            self._waiters.append(waiter)
            deadline = started_at + self.queue_timeout_seconds

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                    raise LLMCapacityError(
                        "The answer service is busy. Please retry shortly."
                    )

                if self._waiters[0] is waiter and self._active < self.max_active:
                    self._waiters.popleft()
                    self._active += 1
                    self._condition.notify_all()
                    return time.monotonic() - started_at

                self._condition.wait(timeout=remaining)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("LLM admission slot released without acquisition")
            self._active -= 1
            self._condition.notify_all()

    @contextmanager
    def slot(self) -> Iterator[float]:
        queued_seconds = self.acquire()
        try:
            yield queued_seconds
        finally:
            self.release()


llm_admission = LLMAdmissionController()
