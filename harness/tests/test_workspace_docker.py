"""Unit tests for Docker /testbed archive extraction (no Docker required)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from comptroller_swebench.workspace_docker import extract_testbed_archive


def _tar_chunks(buf: io.BytesIO) -> list[bytes]:
    data = buf.getvalue()
    mid = len(data) // 2 or 1
    return [data[:mid], data[mid:]]


def test_extract_testbed_archive_single_top_level_dir(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        raw = b"hello"
        info = tarfile.TarInfo(name="testbed/a.txt")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    dest = tmp_path / "out"
    extract_testbed_archive(_tar_chunks(buf), dest)
    assert (dest / "a.txt").read_bytes() == b"hello"


def test_extract_testbed_archive_flat_members(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, raw in [("x.txt", b"aa"), ("y.txt", b"bb")]:
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    dest = tmp_path / "flat"
    extract_testbed_archive(iter([buf.getvalue()]), dest)
    assert (dest / "x.txt").read_bytes() == b"aa"
    assert (dest / "y.txt").read_bytes() == b"bb"
