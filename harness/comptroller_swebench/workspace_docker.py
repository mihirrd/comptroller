"""Materialize SWE-bench instance ``/testbed`` from Docker for host-side Comptroller runs."""

from __future__ import annotations

import io
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Matches swebench.harness.constants.DOCKER_WORKDIR
DOCKER_TESTBED = "/testbed"

# Align with swebench.harness.run_evaluation / build_env_images (default make_test_spec arch).
# On Apple Silicon, Docker Desktop typically runs these linux/x86_64 images via emulation.
SWEBENCH_IMAGE_ARCH = "x86_64"


def _harness_root() -> Path:
    """Directory containing ``harness/pyproject.toml`` (for stable swebench relative ``logs/`` paths)."""
    return Path(__file__).resolve().parent.parent


def _import_swebench_harness():
    try:
        from swebench.harness.constants import DOCKER_WORKDIR
        from swebench.harness.docker_build import (
            build_container,
            build_env_images,
            close_logger,
            setup_logger,
        )
        from swebench.harness.docker_utils import cleanup_container
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as e:
        raise ImportError(
            "Docker materialization requires the `swebench` package. "
            "Install SWE-bench from Git (see harness/README.md), e.g. "
            '`uv pip install "git+https://github.com/princeton-nlp/SWE-bench.git@<ref>"`.'
        ) from e
    return (
        DOCKER_WORKDIR,
        build_container,
        build_env_images,
        close_logger,
        setup_logger,
        cleanup_container,
        make_test_spec,
    )


@contextmanager
def _temp_extract_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="sweb_extract_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def extract_testbed_archive(stream: Any, dest: Path) -> None:
    """Unpack ``docker get_archive`` stream for ``/testbed`` into ``dest`` (repository root files)."""
    dest.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    for chunk in stream:
        buf.write(chunk)
    buf.seek(0)
    with _temp_extract_dir() as td_path:
        with tarfile.open(fileobj=buf, mode="r:*") as tar:
            tar.extractall(td_path, filter="data")
        entries = list(td_path.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            top = entries[0]
            for child in top.iterdir():
                shutil.move(str(child), dest / child.name)
        else:
            for child in entries:
                shutil.move(str(child), dest / child.name)


def materialize_testbed_to_host(
    instance: dict[str, Any],
    dest: Path,
    *,
    run_id: str | None = None,
    force_rebuild_image: bool = False,
    overwrite: bool = False,
) -> Path:
    """Start instance container, copy ``/testbed`` to ``dest``, stop container.

    Requires Docker daemon, the ``docker`` Python SDK, and the ``swebench`` package (SWE-bench).
    SWE-bench writes under ``harness/logs/`` relative to process cwd; this function temporarily
    changes cwd to the harness package root so those paths stay inside this repo.

    Returns
    -------
    Path
        ``dest`` (resolved), the repository root for ``AgentTurnInput.workspace_root``.
    """
    import docker

    dest = dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Destination {dest} is not empty. Pass overwrite=True or use --overwrite-materialized."
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    (
        docker_workdir,
        build_container,
        build_env_images,
        close_logger,
        setup_logger,
        cleanup_container,
        make_test_spec,
    ) = _import_swebench_harness()

    if docker_workdir != DOCKER_TESTBED:
        logger.warning(
            "swebench DOCKER_WORKDIR is %r (expected %r); using get_archive(%r)",
            docker_workdir,
            DOCKER_TESTBED,
            docker_workdir,
        )

    rid = run_id or f"cgmat-{uuid.uuid4().hex[:12]}"
    iid = instance["instance_id"]
    harness_dir = _harness_root()
    log_dir = harness_dir / "logs" / "materialize" / rid / iid
    log_dir.mkdir(parents=True, exist_ok=True)
    sw_logger = setup_logger(iid, log_dir / "materialize.log")

    container: Any = None
    cwd_before = Path.cwd()
    client: Any = None
    try:
        client = docker.from_env()
        os.chdir(harness_dir)
        # run_evaluation builds env/base images before instance images; mirror that here.
        # Tags must not be None: get_test_specs_from_dataset passes these as
        # make_test_spec(..., base_image_tag, env_image_tag) positionally.
        _env_ok, env_failed = build_env_images(
            client,
            [instance],
            force_rebuild=force_rebuild_image,
            max_workers=1,
            namespace=None,
            instance_image_tag="latest",
            env_image_tag="latest",
        )
        if env_failed:
            env_logs = harness_dir / "logs" / "build_images" / "env"
            raise RuntimeError(
                "SWE-bench did not finish building the environment Docker image(s). "
                f"See logs under {env_logs} (each image has a build_image.log). "
                "If Docker reported exit code 137 during setup_env.sh, the build was usually killed for "
                "lack of memory (OOM): raise Docker Desktop memory (Settings → Resources), quit other "
                "heavy apps, and retry. On Apple Silicon, linux/x86_64 image builds are emulated and need "
                "more RAM than native arm64."
            )
        spec = make_test_spec(instance, arch=SWEBENCH_IMAGE_ARCH)
        container = build_container(
            spec,
            client,
            rid,
            sw_logger,
            False,
            force_rebuild_image,
        )
        container.start()
        sw_logger.info("Container started: %s", container.id)
        stream, _stat = container.get_archive(docker_workdir)
        extract_testbed_archive(stream, dest)
        sw_logger.info("Exported %s to %s", docker_workdir, dest)
    finally:
        try:
            if client is not None and container is not None:
                cleanup_container(client, container, sw_logger)
        except Exception as e:
            sw_logger.error("cleanup_container: %s", e)
        close_logger(sw_logger)
        os.chdir(cwd_before)

    return dest
