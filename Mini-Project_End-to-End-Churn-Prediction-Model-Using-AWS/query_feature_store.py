
import argparse
import json
from pathlib import Path

from src.feature_store_utils import get_online_features, query_latest_records_from_offline_store

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "project_config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-group-name", required=True)
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--athena-output-s3-uri", default=None)
    parser.add_argument("--record-ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main():
    cfg = load_config()
    args = parse_args()

    if args.mode == "offline":
        if not args.athena_output_s3_uri:
            raise ValueError("--athena-output-s3-uri is required for offline querying")
        df = query_latest_records_from_offline_store(
            feature_group_name=args.feature_group_name,
            athena_output_s3_uri=args.athena_output_s3_uri,
            limit=args.limit,
            region_name=cfg["region"],
        )
        print(df.head(args.limit).to_string(index=False))
    else:
        if not args.record_ids:
            raise ValueError("--record-ids is required for online querying")
        df = get_online_features(
            feature_group_name=args.feature_group_name,
            record_ids=args.record_ids,
            region_name=cfg["region"],
        )
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
