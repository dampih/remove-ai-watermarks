"""Guards for the generated official C2PA soft-binding registry snapshot."""

from __future__ import annotations

import re

import pytest

import scripts.sync_c2pa_soft_bindings as sync
from remove_ai_watermarks._internal._generated_c2pa_soft_bindings import (
    C2PA_SOFT_BINDING_ROWS,
    C2PA_SOFT_BINDING_SOURCE_LICENSE,
    C2PA_SOFT_BINDING_SOURCE_REVISION,
    C2PA_SOFT_BINDING_SOURCE_URL,
)
from remove_ai_watermarks._internal.constants import C2PA_SOFT_BINDING_REGISTRY, C2PA_SOFT_BINDINGS
from scripts.sync_c2pa_soft_bindings import parse_registry, render_module


def _entry(*, identifier: int = 1, algorithm: str = "com.example.mark.1") -> dict[str, object]:
    return {
        "identifier": identifier,
        "alg": algorithm,
        "type": "watermark",
        "encodedMediaTypes": ["text/plain"],
        "entryMetadata": {
            "description": "Example watermark. Longer implementation detail.",
            "dateEntered": "2026-01-01T00:00:00Z",
            "contact": "registry@example.com",
            "informationalUrl": "https://example.com/mark",
        },
    }


def test_packaged_registry_covers_the_current_official_baseline():
    algorithms = [entry.algorithm for entry in C2PA_SOFT_BINDING_REGISTRY]
    identifiers = [entry.identifier for entry in C2PA_SOFT_BINDING_REGISTRY]

    assert len(C2PA_SOFT_BINDING_REGISTRY) >= 53
    assert identifiers == sorted(identifiers)
    assert len(algorithms) == len(set(algorithms))
    assert set(C2PA_SOFT_BINDINGS) == {algorithm.encode() for algorithm in algorithms}
    assert sum(entry.date_entered.startswith("2026-") for entry in C2PA_SOFT_BINDING_REGISTRY) >= 28
    assert {entry.kind for entry in C2PA_SOFT_BINDING_REGISTRY} == {"watermark", "fingerprint"}
    assert "com.adobe.hiermark.A" in algorithms
    assert "com.adobe.flowmark.A" in algorithms

    assert C2PA_SOFT_BINDING_ROWS
    assert C2PA_SOFT_BINDING_SOURCE_URL.endswith("/softbinding-algorithm-list.json")
    assert re.fullmatch(r"[0-9a-f]{40}", C2PA_SOFT_BINDING_SOURCE_REVISION)
    assert C2PA_SOFT_BINDING_SOURCE_LICENSE == "CC BY 4.0"


def test_sync_accepts_encoded_media_and_renders_attribution():
    rows = parse_registry([_entry()])

    assert rows[0].encoded_media_types == ("text/plain",)
    assert rows[0].decoded_media_types == ()
    assert rows[0].display_label == "Example watermark"
    rendered = render_module(rows, source="fixture.json", revision="fixture-revision")
    assert "Source license: CC BY 4.0" in rendered
    assert "Changes: validated, ordered" in rendered


def test_sync_rejects_duplicate_algorithms():
    with pytest.raises(ValueError, match="algorithms must be unique"):
        parse_registry([_entry(identifier=1), _entry(identifier=2)])


def test_official_source_is_fetched_through_its_resolved_revision(monkeypatch: pytest.MonkeyPatch):
    revision = "a" * 40
    requested: list[str] = []
    monkeypatch.setattr(sync, "_latest_revision", lambda: revision)
    monkeypatch.setattr(sync, "_read_bytes", lambda source: requested.append(source) or b"[]")

    payload, resolved = sync._source_snapshot(sync.DEFAULT_SOURCE_URL, None)

    assert payload == b"[]"
    assert resolved == revision
    assert requested == [sync.PINNED_SOURCE_URL.format(revision=revision)]
