from pathlib import Path
import argparse
import re
import sys


def main(args):
    path = Path(args.log_dir)

    watt_hours = []
    gpu_seconds = []

    if args.prefix:
        prefixes = [p.strip() for p in args.prefix.split(",")]
    else:
        prefixes = []

    for file in path.iterdir():
        if file.name.endswith("out"):
            # If a prefix is specified, skip files that don't start with the prefix
            if args.prefix and not any(
                file.name.startswith(prefix) for prefix in prefixes
            ):
                continue
            with file.open("r") as f:
                file_lines = f.readlines()
                for line in file_lines:
                    # TOTAL: 13324.34 Wh
                    if line.startswith("TOTAL:"):
                        watt_hours.append(float(line.split(" ")[1]))
                    if line.startswith("GPU "):
                        # GPU 0: 3388.46 Wh, avg power: 348.57 W (34996.26 s)
                        # Extract the time in seconds using regex
                        match = re.search(r"\((\d+\.?\d*) s\)", line)
                        if match:
                            gpu_seconds.append(float(match.group(1)))

    if not gpu_seconds:
        print("No GPU time data found in log files.", flush=True, file=sys.stdout)
    if not watt_hours:
        print("No watt-hour data found in log files.", flush=True, file=sys.stdout)

    # Convert GPU seconds to hours and calculate total watt-hours
    if gpu_seconds:
        gpu_hours = sum(gpu_seconds) / 3600
        print(f"Total GPU hours: {gpu_hours:.2f} h", flush=True, file=sys.stdout)
    if watt_hours:
        total_watt_hours = sum(watt_hours)
        print(
            f"Total watt-hours: {total_watt_hours:.2f} Wh", flush=True, file=sys.stdout
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate resource use from log files."
    )
    parser.add_argument(
        "--log-dir", type=str, default="../logs", help="Directory containing log files."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix of log files to consider. Give multiple prefixes separated by comma, e.g. '1614,1615'."
        "If empty, all log files are considered.",
    )
    args = parser.parse_args()
    main(args)
