import json
from pathlib import Path
import argparse

def generate_descriptor_ids(input_file, output_file):
    in_f = Path(input_file)
    out_f = Path(output_file)
    
    with in_f.open("r", encoding="utf-8") as f_in, out_f.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            desc_exp = json.loads(line)
            desc_exp["id"] = f"{idx:08d}"
            f_out.write(f"{json.dumps(desc_exp)}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    
    args = parser.parse_args()
    generate_descriptor_ids(args.input, args.output)
