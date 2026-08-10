"""Smoke tests for the remaining importable packages."""

from __future__ import annotations


def test_package_imports() -> None:
    import temporal_manifolds  # noqa: F401
    from temporal_manifolds import activations, dataset, viz  # noqa: F401
