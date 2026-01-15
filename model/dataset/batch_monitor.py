"""Monitor batch job progress and handle downloads."""

import time

from absl import logging
from tqdm import tqdm

try:
    from .doubleword_client import DoublewordClient
except ImportError:
    from doubleword_client import DoublewordClient


class BatchMonitor:
    """Monitors batch jobs and downloads results when complete."""

    def __init__(self, client: DoublewordClient):
        """
        Initialize batch monitor.

        Args:
            client: DoublewordClient instance
        """
        self.client = client

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 60,
        timeout: int | None = None,
        show_progress: bool = True,
    ) -> str:
        """
        Poll batch until complete, then download results.

        Args:
            batch_id: Batch job ID to monitor
            poll_interval: Seconds between status checks (default: 60)
            timeout: Maximum seconds to wait (None for no timeout)
            show_progress: Whether to show progress bar

        Returns:
            Content of results file as string

        Raises:
            RuntimeError: If batch fails, expires, or is cancelled
            TimeoutError: If timeout is reached
        """
        start_time = time.time()
        checks = 0

        if show_progress:
            pbar = tqdm(desc="Waiting for batch", unit="check")
        else:
            pbar = None

        try:
            while True:
                status = self.client.get_batch_status(batch_id)
                checks += 1

                # Update progress bar
                if pbar:
                    pbar.set_postfix(
                        {
                            "status": status["status"],
                            "progress": f"{status['completed']}/{status['total']}",
                            "failed": status["failed"],
                        }
                    )
                    pbar.update(1)
                else:
                    logging.info(
                        f"Batch {batch_id}: {status['status']} - "
                        f"{status['completed']}/{status['total']} completed, "
                        f"{status['failed']} failed"
                    )

                # Check completion status
                if status["status"] == "completed":
                    logging.info(f"Batch {batch_id} completed successfully")
                    if pbar:
                        pbar.close()
                    return self.client.download_results(status["output_file_id"])

                elif status["status"] in ["failed", "expired", "cancelled"]:
                    error_msg = f"Batch {batch_id} {status['status']}"
                    if pbar:
                        pbar.close()
                    raise RuntimeError(error_msg)

                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    if pbar:
                        pbar.close()
                    raise TimeoutError(
                        f"Batch {batch_id} timed out after {timeout}s "
                        f"(status: {status['status']})"
                    )

                # Wait before next check
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            if pbar:
                pbar.close()
            logging.warning(f"Monitoring interrupted for batch {batch_id}")
            raise

    def check_status(self, batch_id: str) -> dict:
        """
        Check batch status once without waiting.

        Args:
            batch_id: Batch job ID

        Returns:
            Status dictionary
        """
        return self.client.get_batch_status(batch_id)
