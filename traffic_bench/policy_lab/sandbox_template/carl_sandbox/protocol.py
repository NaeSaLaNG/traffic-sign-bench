"""Wire format between the judge and the sandboxed policy.

A frame is a 4-byte big-endian header length, a JSON header, then an optional
raw byte payload whose size the header gives in ``payload_bytes``. Only JSON
crosses from the policy to the judge, so the judge never unpickles anything a
policy produced.
"""
from __future__ import annotations

import json
import socket
import struct

MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 24


def send_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_bytes"] = len(payload)
    raw = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(raw)) + raw)
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return bytes(buf)


def recv_frame(sock: socket.socket, max_payload: int = MAX_PAYLOAD_BYTES) -> tuple[dict, bytes]:
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    if n > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {n} bytes")
    header = json.loads(_recv_exact(sock, n).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header must be a JSON object")
    size = int(header.get("payload_bytes", 0) or 0)
    if size < 0 or size > max_payload:
        raise ValueError(f"payload size {size} out of bounds (max {max_payload})")
    payload = _recv_exact(sock, size) if size else b""
    return header, payload
