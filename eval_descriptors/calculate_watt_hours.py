from pathlib import Path

paths = [Path("..")/"logs",  Path("..")/"logs"/"archive"]

watt_hours = []

for path in paths:
    for file in path.iterdir():
        if file.name.endswith("out"):
            with file.open("r") as f:
                file_lines = f.readlines()
                for line in file_lines:
                    if line.startswith("TOTAL:"):
                        watt_hours.append(float(line.split(" ")[1]))
                        
                        
print(sum(watt_hours), flush=True)