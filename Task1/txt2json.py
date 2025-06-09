import argparse
import json

def convert_tsv_to_json(input_path: str, output_path: str):
    output = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' not in line:
                print(f"Skipping invalid line (no tab found): {line}")
                continue
            id_, instruction = line.split('\t', 1)
            output.append({
                "id": id_,
                "instruction": instruction
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(output)} entries from {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert TSV to JSON for id/instruction format.")
    parser.add_argument("input", help="Path to the input TSV file")
    parser.add_argument("output", help="Path where the output JSON will be written")
    args = parser.parse_args()

    convert_tsv_to_json(args.input, args.output)

if __name__ == "__main__":
    main()
