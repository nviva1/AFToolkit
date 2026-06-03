"""Logging configuration helper for AFToolkit command-line entry points."""

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level=logging.INFO):
    """Configure root logging for CLI usage.

    Idempotent: calling it more than once does not add duplicate handlers.
    Library code should *not* call this; it is intended for the ``main()``
    entry points so that ``logger.info(...)`` messages are actually emitted.
    """
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    logging.getLogger().setLevel(level)
