import hashlib

from scripts.build_paper import checkout_digests


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def test_lf_blob_accepts_its_uniform_crlf_checkout():
    blob = b"alpha\nbeta\n"
    assert checkout_digests(blob) == {digest(blob), digest(b"alpha\r\nbeta\r\n")}


def test_crlf_blob_accepts_its_uniform_lf_checkout():
    blob = b"alpha\r\nbeta\r\n"
    assert checkout_digests(blob) == {digest(blob), digest(b"alpha\nbeta\n")}


def test_mixed_or_bare_carriage_returns_must_match_exactly():
    for blob in (b"alpha\r\nbeta\n", b"alpha\rbeta\n", b"alpha\r"):
        assert checkout_digests(blob) == {digest(blob)}


def test_content_changes_are_never_accepted():
    recorded = digest(b"alpha\r\nbeta\r\n")
    assert recorded in checkout_digests(b"alpha\nbeta\n")
    assert recorded not in checkout_digests(b"alpha\nbeta \n")
    assert recorded not in checkout_digests(b"alpha\nbeta\n\n")


def test_binary_without_newlines_has_one_identity():
    blob = bytes(range(256)).replace(b"\n", b"").replace(b"\r", b"")
    assert checkout_digests(blob) == {digest(blob)}
