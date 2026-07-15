from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from .errors import WorkerError


@dataclass(frozen=True, slots=True)
class ExposurePolicy:
    host: str = "127.0.0.1"
    bearer_token: str | None = None
    tls_terminated: bool = False
    trusted_reverse_proxy: bool = False

    def validate(self) -> None:
        try: loopback = ipaddress.ip_address(self.host).is_loopback
        except ValueError: loopback = self.host.lower() == "localhost"
        if not loopback and not (self.bearer_token and (self.tls_terminated or self.trusted_reverse_proxy)):
            raise WorkerError("worker_not_ready", "non-loopback exposure requires bearer auth and TLS/proxy trust")

    def authorized(self, header: str | None) -> bool:
        return self.bearer_token is None or header == f"Bearer {self.bearer_token}"
