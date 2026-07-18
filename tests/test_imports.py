"""Smoke test: every new package is importable."""

from __future__ import annotations


def test_new_packages_import() -> None:
    from src import analysis, capture, chat_markup, conversation, core, datasets, engine  # noqa: F401
    from src.capture.boundaries import find_boundaries  # noqa: F401
    from src.capture.store import ResponseStore, run_fingerprint  # noqa: F401
    from src.chat_markup.registry import MARKUP_REGISTRY, detect_markup  # noqa: F401
    from src.conversation.driver import ConversationDriver, ProtocolConfig  # noqa: F401
    from src.datasets.generator import build_dataset  # noqa: F401
    from src.engine.base import Engine  # noqa: F401


def test_legacy_package_imports() -> None:
    import temporal_manifolds  # noqa: F401
