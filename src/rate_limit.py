"""A sliding-window request limit, per client and for everyone together."""

import time
from collections import deque


class RateLimiter:
    """At most `per_client` requests per client and `total` overall in any `window` seconds.

    A limit of 0 is no limit. State lives in memory, so it suits one server process.
    """

    def __init__(self, per_client: int, total: int, window: float = 60, clock=time.monotonic):
        self.per_client, self.total, self.window, self.clock = per_client, total, window, clock
        self.clients = {}      # client -> deque of the times of its accepted requests
        self.all = deque()     # times of every accepted request

    def allow(self, client: str) -> bool:
        now = self.clock()
        self._forget_before(now - self.window)
        mine = self.clients.get(client, ())
        if (self.per_client and len(mine) >= self.per_client) or \
           (self.total and len(self.all) >= self.total):
            return False
        self.clients.setdefault(client, deque()).append(now)
        self.all.append(now)
        return True

    def _forget_before(self, cutoff: float):
        while self.all and self.all[0] <= cutoff:
            self.all.popleft()
        for client in [c for c, times in self.clients.items() if times[-1] <= cutoff]:
            del self.clients[client]
        for times in self.clients.values():
            while times[0] <= cutoff:
                times.popleft()
