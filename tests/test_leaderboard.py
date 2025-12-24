import logging
import multiprocessing
import os
import time
import warnings

import pytest
import requests

TIMEOUT = 300


def run_leaderboard_app():
    """Function to launch the leaderboard app."""
    # Set up logging for subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - SUBPROCESS_PID:%(process)d - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress the same WARNING messages that the main app suppresses
    logging.getLogger("mteb.results.task_result").setLevel(logging.ERROR)
    logging.getLogger("mteb.models.model_meta").setLevel(logging.ERROR)
    logging.getLogger("mteb.results.benchmark_results").setLevel(logging.ERROR)

    # Filter out the "Couldn't get scores" warnings
    warnings.filterwarnings("ignore", message="Couldn't get scores for .* due to .*")

    try:
        logger.info(
            f"Subprocess {os.getpid()} starting leaderboard app initialization..."
        )

        from mteb.leaderboard.app import get_leaderboard_app

        logger.info("Calling get_leaderboard_app()...")

        app = get_leaderboard_app()
        logger.info("get_leaderboard_app() completed successfully")

        logger.info("Launching Gradio app...")
        # Remove prevent_thread_lock=True so Gradio blocks and keeps process alive
        app.launch(server_name="0.0.0.0", server_port=7860)
        logger.info(
            "Gradio app launched successfully (this shouldn't print if blocking works)"
        )

    except Exception as e:
        logger.error(f"SUBPROCESS CRASHED: {type(e).__name__}: {e}")
        logger.error("Exception details:", exc_info=True)
        # Ensure logs are flushed before the process dies
        for handler in logging.root.handlers:
            handler.flush()
        raise


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.leaderboard_stability
def test_leaderboard_app_does_not_crash():
    """Test to ensure the leaderboard app does not crash within the first 5 minutes."""
    # Set up logging for the main test process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - TEST_PID:%(process)d - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WARNING messages in the main test process too
    logging.getLogger("mteb.results.task_result").setLevel(logging.ERROR)
    logging.getLogger("mteb.models.model_meta").setLevel(logging.ERROR)
    logging.getLogger("mteb.results.benchmark_results").setLevel(logging.ERROR)

    # Filter out the "Couldn't get scores" warnings in main process too
    warnings.filterwarnings("ignore", message="Couldn't get scores for .* due to .*")

    logger.info(f"Starting leaderboard stability test (PID: {os.getpid()})")
    logger.info(f"Test will run for {TIMEOUT} seconds")

    # Create process
    logger.info("Creating subprocess...")
    process = multiprocessing.Process(target=run_leaderboard_app)

    try:
        logger.info("Starting subprocess...")
        process.start()
        logger.info(f"Subprocess started with PID: {process.pid}")

        # Monitor the process and wait for HTTP 200 OK
        app_ready = False
        app_ready_time = None

        for i in range(TIMEOUT):
            try:
                is_alive = process.is_alive()

                if not is_alive:
                    # Process died - get more information
                    exit_code = process.exitcode
                    logger.error(f"PROCESS DIED at second {i}!")
                    logger.error(f"Subprocess exit code: {exit_code}")
                    logger.error(f"Process PID was: {process.pid}")

                    # Try to get any remaining information
                    if hasattr(process, "_stderr") and process._stderr:
                        logger.error(f"Subprocess stderr: {process._stderr}")
                    if hasattr(process, "_stdout") and process._stdout:
                        logger.error(f"Subprocess stdout: {process._stdout}")

                    pytest.fail(
                        f"Leaderboard app crashed after {i} seconds with exit code {exit_code}"
                    )

                # Check if HTTP server is ready (only start checking after 30 seconds to avoid spam)
                if not app_ready and i >= 30:
                    try:
                        response = requests.head("http://localhost:7860", timeout=2)
                        if response.status_code == 200:
                            app_ready = True
                            app_ready_time = i
                            logger.info(
                                f"🎉 HTTP 200 OK received at second {i}! App is fully ready."
                            )
                    except (
                        requests.exceptions.RequestException,
                        requests.exceptions.Timeout,
                    ):
                        # Server not ready yet, keep waiting
                        pass

                # If app is ready and we've waited 10 more seconds, test passes!
                if app_ready and (i - app_ready_time) >= 10:
                    logger.info(
                        f"✅ Test completed successfully! App ready at {app_ready_time}s, waited 10 more seconds."
                    )
                    logger.info(
                        f"Total test time: {i} seconds (much faster than {TIMEOUT}s)"
                    )
                    break

                # Log progress every 30 seconds
                if i % 30 == 0 and i > 0:
                    if app_ready:
                        logger.info(
                            f"App ready, waiting {10 - (i - app_ready_time)} more seconds..."
                        )
                    else:
                        logger.info(
                            f"Process alive after {i}s, checking for HTTP 200..."
                        )

                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"Exception while checking process status: {type(e).__name__}: {e}"
                )
                logger.error("Exception details:", exc_info=True)
                # Continue checking unless it's a critical error
                if "process" in str(e).lower():
                    pytest.fail(f"Critical error monitoring subprocess: {e}")

        else:
            # Only reach here if we hit the timeout without app being ready + 10 seconds
            if app_ready:
                logger.info(
                    f"✅ Test completed at timeout - app was ready at {app_ready_time}s"
                )
            else:
                logger.warning(
                    f"⚠️  Test completed but never got HTTP 200 OK (process alive for {TIMEOUT}s)"
                )
                logger.info(
                    "This might indicate the app started but isn't serving HTTP properly"
                )

    except Exception as e:
        logger.error(f"Exception in test: {type(e).__name__}: {e}")
        logger.error("Test exception details:", exc_info=True)
        raise

    finally:
        logger.info("Cleaning up subprocess...")
        try:
            if process.is_alive():
                logger.info(f"Terminating process {process.pid}...")
                process.terminate()

                # Give it a moment to terminate gracefully
                logger.info("Waiting for process to terminate...")
                process.join(timeout=5)

                if process.is_alive():
                    logger.warning(
                        f"Process {process.pid} didn't terminate gracefully, killing..."
                    )
                    process.kill()
                    process.join()

                logger.info(
                    f"Process cleanup complete. Final exit code: {process.exitcode}"
                )
            else:
                logger.info("Process was already dead, no cleanup needed")

        except Exception as e:
            logger.error(f"Exception during cleanup: {type(e).__name__}: {e}")
            logger.error("Cleanup exception details:", exc_info=True)
