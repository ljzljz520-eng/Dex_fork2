from __future__ import annotations

import json
import os
import sqlite3
import unicodedata
from pathlib import Path

import pytest
import yaml

from core import entity_engine
from core.entity_engine import index as entity_index
from core.entity_engine.contract import render_company_page, render_person_page
from core.entity_engine.write import fingerprint_page, upsert_frontmatter


@pytest.fixture(autouse=True)
def _clear_reconcile_cache() -> None:
    entity_index.clear_reconcile_cache()
    yield
    entity_index.clear_reconcile_cache()


@pytest.fixture
def entity_vault(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "root": tmp_path,
        "people": tmp_path / "05-Areas" / "People",
        "companies": tmp_path / "05-Areas" / "Companies",
        "people_export": tmp_path / "System" / "People_Index.json",
        "company_export": tmp_path / "System" / "Company_Index.json",
    }
    (paths["people"] / "External").mkdir(parents=True)
    paths["companies"].mkdir(parents=True)
    return paths


def _kwargs(vault: dict[str, Path]) -> dict[str, Path]:
    return {
        "people_dir": vault["people"],
        "companies_dir": vault["companies"],
        "people_index_path": vault["people_export"],
        "company_index_path": vault["company_export"],
    }


def _verify_kwargs(vault: dict[str, Path]) -> dict[str, Path]:
    return {
        "people_index_path": vault["people_export"],
        "company_index_path": vault["company_export"],
    }


class _StatWithMtime:
    """stat_result-like object overriding only nanosecond mtime."""

    def __init__(self, real: os.stat_result, mtime_ns: int) -> None:
        self._real = real
        self.st_mtime_ns = mtime_ns

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def _stat_with_mtime_ns(result: os.stat_result, mtime_ns: int) -> _StatWithMtime:
    return _StatWithMtime(result, mtime_ns)


def _write_person(vault: dict[str, Path], name: str = "Alice Smith") -> Path:
    path = vault["people"] / "External" / f"{name.replace(' ', '_')}.md"
    path.write_text(
        render_person_page(
            name,
            role="VP Product",
            company="Acme",
            emails=["fixture-alice@example.com"],
            aliases=["Al"],
        ),
        encoding="utf-8",
    )
    return path


def _write_company(vault: dict[str, Path], name: str = "Acme") -> Path:
    path = vault["companies"] / f"{name.replace(' ', '_')}.md"
    path.write_text(
        render_company_page(
            name,
            domains=["acme.test"],
            website="https://acme.test",
            status="Customer",
        ),
        encoding="utf-8",
    )
    return path


def test_build_from_frontmatter_creates_schema_nodes_keys_and_json_exports(
    entity_vault: dict[str, Path],
) -> None:
    _write_person(entity_vault)
    _write_company(entity_vault)

    result = entity_index.build_from_vault(
        entity_vault["root"],
        **_kwargs(entity_vault),
    )

    assert result == {"added": 2, "changed": 0, "removed": 0}
    db_path = entity_index.database_path(entity_vault["root"])
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "source_files",
            "nodes",
            "node_keys",
            "edges",
            "touches",
            "meta",
        } <= tables
        assert connection.execute("SELECT COUNT(*) FROM nodes").fetchone() == (2,)
        assert connection.execute(
            "SELECT kind, value FROM node_keys ORDER BY kind, value"
        ).fetchall() == [
            ("alias", "al"),
            ("domain", "acme.test"),
            ("email", "fixture-alice@example.com"),
            ("name", "acme"),
            ("name", "alice smith"),
            ("stem", "acme"),
            ("stem", "alice smith"),
        ]
        assert connection.execute("SELECT COUNT(*) FROM edges").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM touches").fetchone() == (0,)

    people_export = json.loads(entity_vault["people_export"].read_text())
    company_export = json.loads(entity_vault["company_export"].read_text())
    assert people_export["people"][0]["name"] == "Alice Smith"
    assert company_export["companies"][0]["name"] == "Acme"


def test_path_set_reconcile_removes_deleted_person_and_old_company_name(
    entity_vault: dict[str, Path],
) -> None:
    person = _write_person(entity_vault)
    company = _write_company(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))

    person.unlink()
    renamed = company.with_name("Acme_Corporation.md")
    company.rename(renamed)
    result = entity_index.reconcile(entity_vault["root"], **_kwargs(entity_vault))

    assert result == {"added": 1, "changed": 0, "removed": 2}
    assert entity_index.lookup_person(
        entity_vault["root"], "Alice", **_kwargs(entity_vault)
    )["matches"] == []
    company_match = entity_index.find_company_by_domain(
        entity_vault["root"], "mail.acme.test", **_kwargs(entity_vault)
    )
    assert company_match is not None
    assert company_match["path"].endswith("Acme_Corporation.md")
    with sqlite3.connect(entity_index.database_path(entity_vault["root"])) as connection:
        paths = {
            row[0] for row in connection.execute("SELECT path FROM source_files")
        }
    assert paths == {"05-Areas/Companies/Acme_Corporation.md"}


def test_foreign_keys_are_enabled_and_source_delete_cascades(
    entity_vault: dict[str, Path],
) -> None:
    person = _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    source_path = person.relative_to(entity_vault["root"]).as_posix()
    db_path = entity_index.database_path(entity_vault["root"])

    with entity_index.connect(db_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        connection.execute(
            """
            INSERT INTO edges(src_id, edge_type, dst_id, dst_ref, source_path)
            VALUES (?, 'works_at', NULL, 'Acme', ?)
            """,
            (source_path, source_path),
        )
        connection.execute(
            """
            INSERT INTO touches(
                node_id, ts, touch_type, direction, source, nature, source_path
            ) VALUES (?, '2026-07-20T10:00:00', 'meeting', 'none',
                      'meeting-1', 'Planning', ?)
            """,
            (source_path, source_path),
        )

    person.unlink()
    entity_index.reconcile(
        entity_vault["root"], force=True, **_kwargs(entity_vault)
    )

    with entity_index.connect(db_path) as connection:
        for table in ("source_files", "nodes", "node_keys", "edges", "touches"):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed table names
            ).fetchone() == (0,)


def test_frontmatter_touches_project_idempotently_and_cascade_on_delete(
    entity_vault: dict[str, Path],
) -> None:
    person = _write_person(entity_vault)
    original = person.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(original.split("---", 2)[1])
    frontmatter["touches"] = [
        {
            "ts": "2026-07-20",
            "type": "meeting",
            "direction": "none",
            "source": {"id": "meeting-1", "title": "Planning"},
            "nature": "Reviewed the roadmap.",
        },
        {
            "ts": "2026-07-21",
            "type": "mention",
            "source": "meeting-2",
        },
        {"ts": "2026-07-22", "source": {"id": "missing-type"}},
        {
            "ts": "2026-07-23",
            "type": "meeting",
            "direction": {"bad": "shape"},
            "source": {"id": "malformed-direction"},
        },
        {
            "ts": "2026-07-24",
            "type": "mention",
            "source": {"id": "malformed-nature"},
            "nature": ["bad", "shape"],
        },
    ]
    body = original.split("---", 2)[2]
    person.write_text(
        (
            f"---\n{yaml.safe_dump(frontmatter, sort_keys=False).rstrip()}\n---{body}"
        ).replace("ts: '2026-07-20'", "ts: 2026-07-20"),
        encoding="utf-8",
    )
    source_path = person.relative_to(entity_vault["root"]).as_posix()

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    entity_index.reconcile(
        entity_vault["root"], force=True, **_kwargs(entity_vault)
    )

    db_path = entity_index.database_path(entity_vault["root"])
    with entity_index.connect(db_path) as connection:
        assert connection.execute(
            """
            SELECT node_id, ts, touch_type, direction, source, nature, source_path
            FROM touches ORDER BY ts
            """
        ).fetchall() == [
            (
                source_path,
                "2026-07-20",
                "meeting",
                "none",
                "meeting-1",
                "Reviewed the roadmap.",
                source_path,
            ),
            (
                source_path,
                "2026-07-21",
                "mention",
                None,
                "meeting-2",
                None,
                source_path,
            ),
        ]

    person.unlink()
    entity_index.reconcile(
        entity_vault["root"], force=True, **_kwargs(entity_vault)
    )
    with entity_index.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM touches").fetchone() == (0,)


def test_schema_version_drift_rebuilds_touches_from_frontmatter(
    entity_vault: dict[str, Path],
) -> None:
    person = _write_person(entity_vault)
    original = person.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(original.split("---", 2)[1])
    frontmatter["touches"] = [
        {
            "ts": "2026-07-20",
            "type": "meeting",
            "direction": "none",
            "source": {"id": "meeting-1", "title": "Planning"},
        }
    ]
    body = original.split("---", 2)[2]
    person.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False).rstrip()}\n---{body}",
        encoding="utf-8",
    )

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    db_path = entity_index.database_path(entity_vault["root"])
    with entity_index.connect(db_path) as connection:
        connection.execute("DELETE FROM touches")
        connection.execute(
            "UPDATE meta SET value = '0' WHERE key = 'schema_version'"
        )

    result = entity_index.reconcile(
        entity_vault["root"],
        debounce_seconds=60,
        **_kwargs(entity_vault),
    )

    assert result == {"added": 1, "changed": 0, "removed": 0}
    with entity_index.connect(db_path) as connection:
        assert connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone() == (entity_index.SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT ts, source FROM touches"
        ).fetchall() == [("2026-07-20", "meeting-1")]


def test_frontmatter_relationships_project_edges_and_power_neighbors(
    entity_vault: dict[str, Path],
) -> None:
    alice = _write_person(entity_vault)
    bob = _write_person(entity_vault, "Bob Jones")
    company = _write_company(entity_vault)
    bob_original = bob.read_text(encoding="utf-8")
    bob_frontmatter = yaml.safe_load(bob_original.split("---", 2)[1])
    bob_frontmatter["emails"] = ["bob@example.com"]
    bob.write_text(
        (
            f"---\n{yaml.safe_dump(bob_frontmatter, sort_keys=False).rstrip()}"
            f"\n---{bob_original.split('---', 2)[2]}"
        ),
        encoding="utf-8",
    )
    original = alice.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(original.split("---", 2)[1])
    frontmatter["relationships"] = [
        {
            "type": "works_at",
            "target": "[[Acme]]",
            "status": "suggested",
            "source": {"kind": "domain-match", "id": "acme.test"},
            "date": "2026-07-23",
        },
        {
            "type": "reports_to",
            "target": "bob@example.com",
            "status": "suggested",
            "source": {"kind": "meeting", "id": "meeting-1"},
            "date": "2026-07-22",
        },
        {
            "type": "related_to",
            "target": "[[Unresolved Person]]",
            "status": "suggested",
            "source": {"kind": "co-attendance", "id": "meeting-2"},
            "date": "2026-07-21",
        },
    ]
    frontmatter["relationships"].append(
        dict(frontmatter["relationships"][-1])
    )
    body = original.split("---", 2)[2]
    alice.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False).rstrip()}\n---{body}",
        encoding="utf-8",
    )
    alice_id = alice.relative_to(entity_vault["root"]).as_posix()
    bob_id = bob.relative_to(entity_vault["root"]).as_posix()
    company_id = company.relative_to(entity_vault["root"]).as_posix()

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))

    with entity_index.connect(
        entity_index.database_path(entity_vault["root"])
    ) as connection:
        assert connection.execute(
            """
            SELECT src_id, edge_type, dst_id, dst_ref, source_path
            FROM edges
            ORDER BY edge_type
            """
        ).fetchall() == [
            (
                alice_id,
                "related_to",
                None,
                "[[Unresolved Person]]",
                alice_id,
            ),
            (
                alice_id,
                "reports_to",
                bob_id,
                "bob@example.com",
                alice_id,
            ),
            (
                alice_id,
                "works_at",
                company_id,
                "[[Acme]]",
                alice_id,
            ),
        ]

    assert entity_index.neighbors(
        entity_vault["root"], alice_id, **_kwargs(entity_vault)
    ) == [
        {
            "other": "[[Unresolved Person]]",
            "edge_type": "related_to",
            "direction": "out",
            "label": "related_to",
        },
        {
            "other": bob_id,
            "edge_type": "reports_to",
            "direction": "out",
            "label": "reports_to",
        },
        {
            "other": company_id,
            "edge_type": "works_at",
            "direction": "out",
            "label": "works_at",
        },
    ]
    assert entity_index.neighbors(
        entity_vault["root"], bob_id, **_kwargs(entity_vault)
    ) == [
        {
            "other": alice_id,
            "edge_type": "reports_to",
            "direction": "in",
            "label": "manages",
        }
    ]
    assert entity_index.neighbors(
        entity_vault["root"], company_id, **_kwargs(entity_vault)
    ) == [
        {
            "other": alice_id,
            "edge_type": "works_at",
            "direction": "in",
            "label": "employs",
        }
    ]


def test_nfd_person_nfc_relationship_confirm_resync_and_index_resolve_once(
    entity_vault: dict[str, Path],
) -> None:
    nfd_name = unicodedata.normalize("NFD", "José Álvarez")
    nfc_name = unicodedata.normalize("NFC", nfd_name)
    target = _write_person(entity_vault, nfd_name)
    source = _write_person(entity_vault, "Alice Smith")
    suggestion = {
        "type": "related_to",
        "target": f"[[{nfd_name}]]",
        "status": "suggested",
        "source": {"kind": "co-attendance", "id": "meeting-1"},
        "date": "2026-07-23",
    }
    assert upsert_frontmatter(source, {"relationships": [suggestion]})
    assert upsert_frontmatter(
        source,
        {
            "relationships": [
                {
                    **suggestion,
                    "target": f"[[{nfc_name}]]",
                    "source": {
                        "kind": "co-attendance",
                        "id": "meeting-2",
                    },
                }
            ]
        },
    )
    confirmed = entity_engine.mutate_relationships(
        source,
        fingerprint_page(source),
        {
            "kind": "confirm_relationship",
            "edge_key": f"related_to::[[{nfc_name.lower()}]]",
        },
    )
    assert confirmed.status == "updated"
    resynced = entity_engine.mutate_relationships(
        source,
        fingerprint_page(source),
        {
            "kind": "relationship",
            "relationships": [
                {
                    "type": "related_to",
                    "target_ref": f"[[{nfd_name}]]",
                    "source": {
                        "kind": "co-attendance",
                        "id": "meeting-3",
                        "date": "2026-07-24",
                    },
                    "confidence": "suggested",
                }
            ],
        },
    )
    assert resynced.status == "noop"

    entity_index.build_from_vault(
        entity_vault["root"],
        **_kwargs(entity_vault),
    )
    target_id = target.relative_to(entity_vault["root"]).as_posix()
    source_id = source.relative_to(entity_vault["root"]).as_posix()
    with entity_index.connect(
        entity_index.database_path(entity_vault["root"])
    ) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM nodes WHERE type = 'person'"
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT src_id, dst_id FROM edges WHERE edge_type = 'related_to'"
        ).fetchall() == [(source_id, target_id)]
        assert connection.execute(
            "SELECT COUNT(*) FROM node_keys WHERE kind = 'name' AND value = ?",
            (nfc_name.casefold(),),
        ).fetchone() == (1,)
    looked_up = entity_index.lookup_person(
        entity_vault["root"],
        nfc_name,
        **_kwargs(entity_vault),
    )
    assert looked_up["total_matches"] == 1
    assert looked_up["matches"][0]["path"] == target_id


def test_neighbors_derives_inverse_without_storing_a_second_edge(
    entity_vault: dict[str, Path],
) -> None:
    alice = _write_person(entity_vault)
    bob = _write_person(entity_vault, "Bob Jones")
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    alice_id = alice.relative_to(entity_vault["root"]).as_posix()
    bob_id = bob.relative_to(entity_vault["root"]).as_posix()

    with entity_index.connect(
        entity_index.database_path(entity_vault["root"])
    ) as connection:
        connection.execute(
            """
            INSERT INTO edges(src_id, edge_type, dst_id, dst_ref, source_path)
            VALUES (?, 'works_at', ?, NULL, ?)
            """,
            (alice_id, bob_id, alice_id),
        )

    assert entity_index.neighbors(
        entity_vault["root"], alice_id, **_kwargs(entity_vault)
    ) == [
        {
            "other": bob_id,
            "edge_type": "works_at",
            "direction": "out",
            "label": "works_at",
        }
    ]
    assert entity_index.neighbors(
        entity_vault["root"], bob_id, **_kwargs(entity_vault)
    ) == [
        {
            "other": alice_id,
            "edge_type": "works_at",
            "direction": "in",
            "label": "employs",
        }
    ]
    with entity_index.connect(
        entity_index.database_path(entity_vault["root"])
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM edges").fetchone() == (1,)


def test_delete_disposable_index_then_lookup_rebuilds_identical_results(
    entity_vault: dict[str, Path],
) -> None:
    _write_person(entity_vault)
    before = entity_index.lookup_person(
        entity_vault["root"], "fixture-alice@example.com", **_kwargs(entity_vault)
    )
    entity_index.remove_database(entity_index.database_path(entity_vault["root"]))

    after = entity_index.lookup_person(
        entity_vault["root"], "fixture-alice@example.com", **_kwargs(entity_vault)
    )

    assert after["matches"] == before["matches"]
    assert entity_index.database_path(entity_vault["root"]).exists()


def test_reconcile_debounces_unchanged_tree_but_not_a_deletion(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    person = _write_person(entity_vault)
    publications = 0
    original_dump = entity_index.dump_json_views

    def counted_dump(*args, **kwargs):
        nonlocal publications
        publications += 1
        return original_dump(*args, **kwargs)

    monkeypatch.setattr(entity_index, "dump_json_views", counted_dump)

    # The second, unchanged reconcile still content-fingerprints the pages
    # (that check is what makes silent equal-length swaps detectable) but must
    # skip projection writes and export publication entirely.
    first = entity_index.reconcile(entity_vault["root"], **_kwargs(entity_vault))
    second = entity_index.reconcile(entity_vault["root"], **_kwargs(entity_vault))
    assert first == {"added": 1, "changed": 0, "removed": 0}
    assert second == {"added": 0, "changed": 0, "removed": 0}
    assert publications == 1

    person.unlink()
    third = entity_index.reconcile(entity_vault["root"], **_kwargs(entity_vault))
    assert third == {"added": 0, "changed": 0, "removed": 1}
    assert publications == 2


def test_busy_database_error_does_not_remove_the_index(
    entity_vault: dict[str, Path],
) -> None:
    person = _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    db_path = entity_index.database_path(entity_vault["root"])
    inode = db_path.stat().st_ino
    person.write_text(
        render_person_page("Alice Smith", role="Chief Product Officer"),
        encoding="utf-8",
    )

    blocker = sqlite3.connect(db_path, timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            entity_index.reconcile(
                entity_vault["root"], force=True, **_kwargs(entity_vault)
            )
    finally:
        blocker.rollback()
        blocker.close()

    assert db_path.exists()
    assert db_path.stat().st_ino == inode
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM nodes").fetchone() == (1,)


def test_connect_allows_five_seconds_for_busy_writes(
    entity_vault: dict[str, Path],
) -> None:
    with entity_index.connect(
        entity_index.database_path(entity_vault["root"])
    ) as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)


def test_lookup_person_uses_last_good_json_while_database_is_busy(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    exported = json.loads(entity_vault["people_export"].read_text())
    db_path = entity_index.database_path(entity_vault["root"])

    original_connect = entity_index.connect

    def connect_with_short_test_wait(path: str | Path) -> sqlite3.Connection:
        connection = original_connect(path)
        connection.execute("PRAGMA busy_timeout = 1")
        return connection

    blocker = sqlite3.connect(db_path, timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(entity_index, "connect", connect_with_short_test_wait)
    try:
        result = entity_index.lookup_person(
            entity_vault["root"],
            "Alice Smith",
            force=True,
            **_kwargs(entity_vault),
        )
    finally:
        blocker.rollback()
        blocker.close()

    assert result["matches"][0]["name"] == "Alice Smith"
    assert result["index_age"] == exported["built_at"]


def test_reconcile_reads_and_parses_sources_before_write_transaction(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_person(entity_vault)
    captured_connection: sqlite3.Connection | None = None
    original_connect = entity_index.connect
    original_parse_content = entity_index.parse_entity_page_content
    original_read_bytes = Path.read_bytes

    def capture_connection(path: str | Path) -> sqlite3.Connection:
        nonlocal captured_connection
        captured_connection = original_connect(path)
        return captured_connection

    def parse_outside_transaction(
        page_path: str | Path,
        text: str,
    ) -> dict[str, object]:
        assert captured_connection is not None
        assert captured_connection.in_transaction is False
        return original_parse_content(page_path, text)

    def read_outside_transaction(path: Path) -> bytes:
        if path.suffix == ".md":
            assert captured_connection is not None
            assert captured_connection.in_transaction is False
        return original_read_bytes(path)

    monkeypatch.setattr(entity_index, "connect", capture_connection)
    monkeypatch.setattr(
        entity_index,
        "parse_entity_page_content",
        parse_outside_transaction,
    )
    monkeypatch.setattr(Path, "read_bytes", read_outside_transaction)

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))


def test_corrupt_database_and_sidecars_are_removed_then_rebuilt(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_person(entity_vault)
    db_path = entity_index.database_path(entity_vault["root"])
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"not a sqlite database")
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    wal_path.write_bytes(b"stale wal")
    shm_path.write_bytes(b"stale shm")
    removed_together = False
    original_remove = entity_index.remove_database

    def checked_remove(path: str | Path) -> None:
        nonlocal removed_together
        original_remove(path)
        removed_together = not any(
            candidate.exists() for candidate in (db_path, wal_path, shm_path)
        )

    monkeypatch.setattr(entity_index, "remove_database", checked_remove)

    result = entity_index.lookup_person(
        entity_vault["root"], "Alice", **_kwargs(entity_vault)
    )

    assert result["matches"][0]["name"] == "Alice Smith"
    assert removed_together is True
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)


def test_failed_quick_check_rebuilds_from_frontmatter(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    original_connect = entity_index.connect
    calls = 0

    def fail_quick_check_once(path: str | Path) -> sqlite3.Connection:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise entity_index._FailedQuickCheck("simulated quick_check failure")
        return original_connect(path)

    monkeypatch.setattr(entity_index, "connect", fail_quick_check_once)

    result = entity_index.reconcile(
        entity_vault["root"], force=True, **_kwargs(entity_vault)
    )

    assert result == {"added": 1, "changed": 0, "removed": 0}
    assert calls == 2


def test_quarantined_page_is_findable_without_trusting_frontmatter(
    entity_vault: dict[str, Path],
) -> None:
    broken = entity_vault["people"] / "External" / "Broken_Profile.md"
    broken.write_text(
        "---\n"
        "type: person\n"
        "name: [not closed\n"
        "role: CEO\n"
        "company: Unsafe Co\n"
        "emails: [unsafe@example.com]\n"
        "aliases: [Bad Data]\n",
        encoding="utf-8",
    )

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))

    with entity_index.connect(
        entity_index.database_path(entity_vault["root"])
    ) as connection:
        assert connection.execute(
            "SELECT quarantined FROM source_files"
        ).fetchone() == (1,)
        node = connection.execute(
            """
            SELECT name, type, role, company, status, last_interaction, fields_json
            FROM nodes
            """
        ).fetchone()
        assert node is not None
        assert node[:6] == (
            "Broken Profile",
            "person",
            None,
            None,
            "quarantined",
            None,
        )
        compatibility = json.loads(node[6])["_compat"]
        assert compatibility == {
            "aliases": [],
            "company": None,
            "email": None,
            "emails": [],
            "first_name": "broken",
            "last_interaction": None,
            "name": "Broken Profile",
            "path": "05-Areas/People/External/Broken_Profile.md",
            "role": None,
            "status": "quarantined",
            "tags": [],
            "type": "external",
        }
        assert connection.execute("SELECT COUNT(*) FROM node_keys").fetchone() == (0,)

    result = entity_index.lookup_person(
        entity_vault["root"], "Broken Profile", **_kwargs(entity_vault)
    )
    assert result["matches"][0]["name"] == "Broken Profile"
    assert result["matches"][0]["status"] == "quarantined"
    assert result["matches"][0]["emails"] == []


def test_people_scan_skips_nested_archive_and_stray_root_pages(
    entity_vault: dict[str, Path],
) -> None:
    included = entity_vault["people"] / "Internal" / "Included.md"
    archived = entity_vault["people"] / "Internal" / "_archive" / "Archived.md"
    stray = entity_vault["people"] / "Z.md"
    nested_company = entity_vault["companies"] / "Archive" / "Nested_Co.md"
    included.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    nested_company.parent.mkdir(parents=True)
    included.write_text(render_person_page("Included"), encoding="utf-8")
    archived.write_text(render_person_page("Archived"), encoding="utf-8")
    stray.write_text(render_person_page("Z"), encoding="utf-8")
    nested_company.write_text(
        render_company_page("Nested Co", domains=["nested.example.com"]),
        encoding="utf-8",
    )

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))

    people = entity_index.people_index_data(
        entity_vault["root"], **_kwargs(entity_vault)
    )
    assert [person["name"] for person in people["people"]] == ["Included"]
    company = entity_index.find_company_by_domain(
        entity_vault["root"], "mail.nested.example.com", **_kwargs(entity_vault)
    )
    assert company is not None
    assert company["name"] == "Nested Co"


def test_default_scan_respects_folder_path_remapping(tmp_path: Path) -> None:
    folder_map = tmp_path / "System" / "folder-paths.yaml"
    folder_map.parent.mkdir(parents=True)
    folder_map.write_text(
        'people_internal: "Relationships/Team"\n'
        'people_external: "Relationships/Contacts"\n'
        'companies: "Relationships/Accounts"\n',
        encoding="utf-8",
    )
    person = tmp_path / "Relationships" / "Contacts" / "Ada_Lovelace.md"
    company = tmp_path / "Relationships" / "Accounts" / "Analytical_Engines.md"
    person.parent.mkdir(parents=True)
    company.parent.mkdir(parents=True)
    person.write_text(
        render_person_page("Ada Lovelace", emails=["fixture-ada@example.org"]),
        encoding="utf-8",
    )
    company.write_text(
        render_company_page("Analytical Engines", domains=["engines.test"]),
        encoding="utf-8",
    )

    entity_index.build_from_vault(tmp_path)

    assert entity_index.lookup_person(tmp_path, "Ada")["matches"][0]["path"] == (
        "Relationships/Contacts/Ada_Lovelace.md"
    )
    assert entity_index.find_company_by_domain(
        tmp_path, "mail.engines.test"
    )["path"] == "Relationships/Accounts/Analytical_Engines.md"


def _generation_triplet(vault: dict[str, Path]) -> tuple[str, str, str]:
    db_path = entity_index.database_path(vault["root"])
    with entity_index.connect(db_path) as connection:
        db_generation = connection.execute(
            "SELECT value FROM meta WHERE key = 'generation_id'"
        ).fetchone()[0]
    people_generation = json.loads(vault["people_export"].read_text())["generation_id"]
    company_generation = json.loads(
        vault["company_export"].read_text()
    )["generation_id"]
    return db_generation, people_generation, company_generation


def test_equal_length_same_mtime_swap_reaches_nodes_edges_and_exports(
    entity_vault: dict[str, Path],
) -> None:
    alice = _write_person(entity_vault)
    _write_company(entity_vault)
    original = alice.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(original.split("---", 2)[1])
    frontmatter["relationships"] = [
        {
            "type": "works_at",
            "target": "[[Acme]]",
            "status": "suggested",
            "source": {"kind": "domain-match", "id": "acme.test"},
            "date": "2026-07-23",
        }
    ]
    body = original.split("---", 2)[2]
    alice.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False).rstrip()}\n---{body}",
        encoding="utf-8",
    )
    alice_id = alice.relative_to(entity_vault["root"]).as_posix()
    company_id = (
        entity_vault["companies"] / "Acme.md"
    ).relative_to(entity_vault["root"]).as_posix()

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    db_path = entity_index.database_path(entity_vault["root"])
    with entity_index.connect(db_path) as connection:
        assert connection.execute(
            "SELECT dst_id, dst_ref FROM edges WHERE src_id = ?",
            (alice_id,),
        ).fetchall() == [(company_id, "[[Acme]]")]

    # Equal-length byte swaps (role value and the edge target) with size and
    # mtime_ns deliberately restored to their pre-swap values.
    before = alice.stat()
    swapped = (
        alice.read_bytes()
        .replace(b"VP Product", b"VP Prodcct", 1)
        .replace(b"[[Acme]]", b"[[Bcme]]", 1)
    )
    assert len(swapped) == before.st_size
    alice.write_bytes(swapped)
    os.utime(alice, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (alice.stat().st_size, alice.stat().st_mtime_ns) == (
        before.st_size,
        before.st_mtime_ns,
    )

    result = entity_index.reconcile(entity_vault["root"], **_kwargs(entity_vault))

    assert result == {"added": 0, "changed": 1, "removed": 0}
    with entity_index.connect(db_path) as connection:
        assert connection.execute(
            "SELECT role FROM nodes WHERE id = ?",
            (alice_id,),
        ).fetchone() == ("VP Prodcct",)
        assert connection.execute(
            "SELECT dst_id, dst_ref FROM edges WHERE src_id = ?",
            (alice_id,),
        ).fetchall() == [(None, "[[Bcme]]")]

    people_export = json.loads(entity_vault["people_export"].read_text())
    assert people_export["people"][0]["role"] == "VP Prodcct"
    db_generation, people_generation, company_generation = _generation_triplet(
        entity_vault
    )
    assert db_generation == people_generation == company_generation
    assert entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )
    match = entity_index.lookup_person(
        entity_vault["root"], "Alice Smith", **_kwargs(entity_vault)
    )
    assert match["matches"][0]["role"] == "VP Prodcct"


def test_stable_read_retries_after_an_in_flight_change_then_projects(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    person = _write_person(entity_vault)
    real_stat = Path.stat
    tick = {"n": 0}

    def flaky_stat(self: Path):
        result = real_stat(self)
        if self == person and tick["n"] < 3:
            tick["n"] += 1
            return _stat_with_mtime_ns(result, result.st_mtime_ns + tick["n"])
        return result

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "stat", flaky_stat)
    monkeypatch.setattr(entity_index.time, "sleep", sleeps.append)

    result = entity_index.build_from_vault(
        entity_vault["root"], **_kwargs(entity_vault)
    )

    assert result == {"added": 1, "changed": 0, "removed": 0}
    assert sleeps == [entity_index.STABLE_READ_BACKOFF_SECONDS]
    db_path = entity_index.database_path(entity_vault["root"])
    with entity_index.connect(db_path) as connection:
        assert connection.execute(
            "SELECT quarantined FROM source_files"
        ).fetchone() == (0,)
        assert connection.execute("SELECT role FROM nodes").fetchone() == (
            "VP Product",
        )


def test_persistently_changing_page_is_quarantined_without_torn_content(
    entity_vault: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    person = _write_person(entity_vault)
    real_stat = Path.stat
    tick = {"n": 0}

    def always_changing_stat(self: Path):
        result = real_stat(self)
        if self == person:
            tick["n"] += 1
            return _stat_with_mtime_ns(result, result.st_mtime_ns + tick["n"])
        return result

    monkeypatch.setattr(Path, "stat", always_changing_stat)
    monkeypatch.setattr(entity_index.time, "sleep", lambda _seconds: None)

    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))

    db_path = entity_index.database_path(entity_vault["root"])
    with entity_index.connect(db_path) as connection:
        assert connection.execute(
            "SELECT quarantined FROM source_files"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT status, role, company FROM nodes"
        ).fetchone() == ("quarantined", None, None)
        assert connection.execute("SELECT COUNT(*) FROM node_keys").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM edges").fetchone() == (0,)
    people_export = json.loads(entity_vault["people_export"].read_text())
    entry = people_export["people"][0]
    assert entry["status"] == "quarantined"
    assert entry["emails"] == []
    assert entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )

    match = entity_index.lookup_person(
        entity_vault["root"], "Alice Smith", **_kwargs(entity_vault)
    )
    assert match["matches"][0]["status"] == "quarantined"
    assert match["matches"][0]["emails"] == []


def test_generation_split_is_detected_and_republishes_even_without_content_change(
    entity_vault: dict[str, Path],
) -> None:
    _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    db_path = entity_index.database_path(entity_vault["root"])
    previous_generation, _, _ = _generation_triplet(entity_vault)
    assert entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )

    # Simulate a crash after SQLite committed a new generation but before either
    # JSON export was renamed into place.
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE meta SET value = 'sqlite-ahead' WHERE key = 'generation_id'"
        )
        connection.commit()
    assert not entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )

    # Content is untouched: healing must not depend on a content change or on
    # force=True.
    entity_index.reconcile(entity_vault["root"], **_kwargs(entity_vault))

    db_generation, people_generation, company_generation = _generation_triplet(
        entity_vault
    )
    assert db_generation == people_generation == company_generation
    assert db_generation not in {previous_generation, "sqlite-ahead"}
    assert entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )


def test_split_generation_is_healed_on_the_reader_path(
    entity_vault: dict[str, Path],
) -> None:
    _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))

    # Simulate a crash after one export was renamed but before the other.
    people_payload = json.loads(entity_vault["people_export"].read_text())
    people_payload["generation_id"] = "json-ahead"
    entity_vault["people_export"].write_text(json.dumps(people_payload))
    assert not entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )

    view = entity_index.people_index_data(
        entity_vault["root"], **_kwargs(entity_vault)
    )

    assert entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )
    disk_generation = json.loads(
        entity_vault["people_export"].read_text()
    )["generation_id"]
    assert view["generation_id"] == disk_generation
    assert disk_generation != "json-ahead"


def test_verify_generation_rejects_exports_without_marker(
    entity_vault: dict[str, Path],
) -> None:
    _write_person(entity_vault)
    entity_index.build_from_vault(entity_vault["root"], **_kwargs(entity_vault))
    legacy = {
        key: value
        for key, value in json.loads(
            entity_vault["company_export"].read_text()
        ).items()
        if key != "generation_id"
    }
    entity_vault["company_export"].write_text(json.dumps(legacy))

    assert not entity_index.verify_generation(
        entity_vault["root"], **_verify_kwargs(entity_vault)
    )
