"""Fetching the AlphaFold2 parameter files.

Replaces a previous implementation that shelled out to ``mkdir``/``wget``/``tar``
through ``subprocess`` with ``shell=True`` and an interpolated URL, and which
discarded every command's exit status -- so a failed or partial download looked
exactly like a successful one, and only surfaced later as an opaque error from
the model loader.
"""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import urllib.error
import urllib.request

from aftoolkit.constants import MODEL_WEIGHTS_DIR, OF_WEIGHTS, SOURCE_OF_WEIGHTS_URL

__all__ = ["WeightsUnavailable", "ensure_weights"]

logger = logging.getLogger(__name__)


class WeightsUnavailable(RuntimeError):
    """The AlphaFold parameters are missing and could not be downloaded."""


def ensure_weights(weights_dir: str = MODEL_WEIGHTS_DIR,
                   filename: str = OF_WEIGHTS,
                   url: str = SOURCE_OF_WEIGHTS_URL,
                   download: bool = True) -> str:
    """Return the path to the AlphaFold parameter file, downloading it if needed.

    Args:
        weights_dir: directory the parameters live in.
        filename: the parameter file expected inside it.
        url: tar archive to fetch when the file is absent.
        download: when False, a missing file raises instead of being fetched.

    Returns:
        Absolute path to the parameter file.

    Raises:
        WeightsUnavailable: if the parameters are absent and cannot be obtained.
    """
    weights_path = os.path.join(weights_dir, filename)
    if os.path.exists(weights_path):
        return weights_path

    if not download:
        raise WeightsUnavailable(
            f"AlphaFold parameters not found at {weights_path} and downloading "
            f"is disabled. Fetch them from {url}."
        )

    logger.info("AlphaFold parameters not found; downloading into %s", weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    _download_and_extract(url, weights_dir)

    if not os.path.exists(weights_path):
        raise WeightsUnavailable(
            f"Downloaded and extracted {url}, but {filename} is still missing "
            f"from {weights_dir}."
        )
    logger.info("AlphaFold parameters ready at %s", weights_path)
    return weights_path


def _download_and_extract(url: str, destination: str) -> None:
    """Download a tar archive and extract it, failing loudly on any error."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "alphafold_params.tar")
        try:
            urllib.request.urlretrieve(url, archive)
        except (urllib.error.URLError, OSError) as exc:
            raise WeightsUnavailable(
                f"Could not download AlphaFold parameters from {url}: {exc}"
            ) from exc

        try:
            with tarfile.open(archive) as tar:
                _extract_safely(tar, destination)
        except (tarfile.TarError, OSError) as exc:
            raise WeightsUnavailable(
                f"Could not extract the AlphaFold parameter archive from {url}: {exc}"
            ) from exc


def _extract_safely(tar: tarfile.TarFile, destination: str) -> None:
    """Extract ``tar``, refusing members that would escape ``destination``.

    A tar entry may name an absolute path or contain ``..``; extracting one
    blindly writes outside the target directory.
    """
    destination = os.path.realpath(destination)
    for member in tar.getmembers():
        target = os.path.realpath(os.path.join(destination, member.name))
        if target != destination and not target.startswith(destination + os.sep):
            raise WeightsUnavailable(
                f"Refusing to extract {member.name!r}: it would write outside "
                f"{destination}."
            )
    # `filter="data"` additionally strips special files and unsafe permissions.
    try:
        tar.extractall(destination, filter="data")
    except TypeError:  # Python < 3.12 has no extraction filters.
        tar.extractall(destination)
