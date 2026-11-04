"""Contexto de conversa em memoria, isolado por usuario e com expiracao."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


class SessionCapacityError(RuntimeError):
    pass


@dataclass
class Conversation:
    context: dict = field(default_factory=dict)
    updated_at: float = field(default_factory=monotonic)
    lock: Lock = field(default_factory=Lock)
    active: int = 0


class SessionStore:
    def __init__(self, ttl=1800, capacity=1000, clock=monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self._sessions = {}
        self._lock = Lock()

    @contextmanager
    def conversation(self, session_id: str, user_id: int | None):
        # Um visitante nunca acessa o contexto de um usuario autenticado.
        key = (user_id, session_id)
        with self._lock:
            now = self.clock()
            expired = [k for k, s in self._sessions.items()
                       if not s.active and now - s.updated_at >= self.ttl]
            for old_key in expired:
                del self._sessions[old_key]
            if key not in self._sessions:
                if len(self._sessions) >= self.capacity:
                    available = [(k, s) for k, s in self._sessions.items() if not s.active]
                    if not available:
                        raise SessionCapacityError("conversation capacity reached")
                    del self._sessions[min(available, key=lambda entry: entry[1].updated_at)[0]]
                self._sessions[key] = Conversation(updated_at=now)
            session = self._sessions[key]
            session.active += 1
        try:
            with session.lock:
                yield session
        finally:
            with self._lock:
                session.updated_at = self.clock()
                session.active -= 1
