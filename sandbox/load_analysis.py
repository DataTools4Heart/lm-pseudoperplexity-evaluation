# script to load in json, argparse for arg parsing
#
import argparse
import json

import pandas as pd


def load_json(file_path):
    # append local path
    with open(file_path, "r") as f:
        return json.load(f)


def write_to_parquet(data, file_path):
    df = pd.DataFrame(data)
    print(df.shape)
    df.to_parquet(file_path)


def write_to_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Load JSON file")
    parser.add_argument("--file_path", type=str, help="Path to JSON file")
    args = parser.parse_args()
    data = load_json(args.file_path)
    dictionary_list = [{k: v} for k, v in data.items()]

    write_to_jsonl(dictionary_list, "output.jsonl")


if __name__ == "__main__":
    main()
