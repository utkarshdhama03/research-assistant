"""One stdout log format for `ra serve`, `ra worker` and migrations: `<UTC time> <LEVEL> <logger>: <msg>`."""

import logging
import sys
import time

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


class UtcFormatter(logging.Formatter):
    def converter(self, timestamp: float | None) -> time.struct_time:
        return time.gmtime(timestamp)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(UtcFormatter(LOG_FORMAT, LOG_DATEFMT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request otherwise
