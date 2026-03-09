import functools
import time
import logging
import torch # type: ignore

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def setup_logging(log_file, verbosity: int = 1) -> None:
    """Configure both file and stdout logging, creating the directory if needed."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(
            "Execution of %s took %s.",
            func.__name__,
            time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        )
        return result

    return wrapper

def log_gpu_memory_info():
    num_gpus = torch.cuda.device_count()
    logging.info(f"Found {num_gpus} GPU(s):")

    for i in range(num_gpus):
        # Get device properties
        props = torch.cuda.get_device_properties(i)

        # Get memory info (free, total)
        free_mem, total_mem = torch.cuda.mem_get_info(i)

        logging.info(f"\nGPU {i} ({props.name}):")
        logging.info(f"  Total memory: {total_mem / 1e9:.2f} GiB")
        logging.info(f"  Free memory:  {free_mem / 1e9:.2f} GiB")
        logging.info(f"  Used memory:  {(total_mem - free_mem) / 1e9:.2f} GiB")
        logging.info(f"  Utilization:  {(total_mem - free_mem)/total_mem * 100:.1f}%")
