import json
import logging
from datetime import datetime
from time import perf_counter
from typing import Any


logger = logging.getLogger("decisionlens.telemetry")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def now_ms() -> float:
    return perf_counter() * 1000


def elapsed_ms(start_ms: float) -> float:
    return round(perf_counter() * 1000 - start_ms, 2)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def emit_latency(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "type": "latency",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    payload.update({key: _normalize(value) for key, value in fields.items()})
    logger.info(json.dumps(payload, sort_keys=True))
