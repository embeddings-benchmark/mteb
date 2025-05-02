from __future__ import annotations

import multiprocessing
import time

import pytest

from mteb.leaderboard.app import get_leaderboard_app


def run_leaderboard_app():
    """Function to launch the leaderboard app."""
    app = get_leaderboard_app()
    app.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)


@pytest.mark.timeout(300)
def test_leaderboard_app_does_not_crash():
    """Test to ensure the leaderboard app does not crash within the first 5 minutes."""
    process = multiprocessing.Process(target=run_leaderboard_app)
    process.start()

    try:
        for _ in range(300):
            if not process.is_alive():
                pytest.fail("Leaderboard app crashed during the test.")
            time.sleep(1)
    finally:
        process.terminate()
        process.join()
