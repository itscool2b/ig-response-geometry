"""Inject an anonymous review link only into explicitly private build outputs.

The canonical shared source contains MARKER, never a live anonymous route.
Public builds remove the marker hyperlink. Route files stay outside publication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

MARKER = "https://review.invalid/archive"


def load_review_url(path):
    if path is None:
        return None
    data = Path(path).read_bytes()
    text = data.decode("utf-8-sig").strip()
    if text.startswith("{"):
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {"review_archive_url"}:
            raise ValueError("Review route JSON must contain only review_archive_url")
        url = value["review_archive_url"]
    else:
        url = text
    if not isinstance(url, str):
        raise ValueError("Review route must be a string")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.port not in (None, 443)
            or not re.fullmatch(r"https://[A-Za-z0-9.-]+(?::443)?/[A-Za-z0-9/_~.-]+", url)
            or url == MARKER):
        raise ValueError("Review route must be a plain HTTPS URL without credentials, query or fragment")
    return {"url": url, "source_sha256": hashlib.sha256(data).hexdigest(),
            "url_sha256": hashlib.sha256(url.encode()).hexdigest()}


def review_binding(review):
    return None if review is None else {key: review[key] for key in ("source_sha256", "url_sha256")}


def render_review_route(text, review=None, *, anonymous):
    count = text.count(MARKER)
    if not count:
        if review is not None and anonymous:
            raise ValueError("Anonymous source is missing the review route marker")
        return text
    pattern = re.compile(r"\\href\{" + re.escape(MARKER) + r"\}\{([^{}]+)\}")
    matches = list(pattern.finditer(text))
    if count != 1 or len(matches) != 1:
        raise ValueError("Expected one review route marker in a simple hyperlink")
    if anonymous and review is not None:
        return text.replace(MARKER, review["url"])
    return pattern.sub(lambda match: match.group(1), text)
