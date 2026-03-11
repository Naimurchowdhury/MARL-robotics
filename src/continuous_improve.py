from __future__ import annotations

import time

from improve_training import main


INTERVAL_SECONDS = 600


if __name__ == "__main__":
    while True:
        main()
        time.sleep(INTERVAL_SECONDS)
