from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from strapi_utils import data_dir


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table_name,)).fetchone() is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a joined quote-to-SQL study table")
    parser.add_argument("--database", type=Path, default=data_dir() / "urdaibai.db")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--output", type=Path, default=data_dir() / "sqlite_quote_join.parquet")
    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(f"Missing SQLite database: {args.database}")

    documents = pd.read_parquet(args.documents)
    mask = documents["entity_type"].astype(str).str.contains(r"quote|transcription|transcript", case=False, regex=True)
    quote_docs = documents.loc[mask, ["id", "entity_type", "text"]].copy()

    with sqlite3.connect(args.database) as conn:
        frames: list[pd.DataFrame] = []
        for entity_type, subset in quote_docs.groupby("entity_type", sort=False):
            table_name = str(entity_type).lower()
            if not table_exists(conn, table_name):
                continue
            table_frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            table_frame["id"] = table_frame["id"].astype(str)
            subset = subset.copy()
            subset["id"] = subset["id"].astype(str)
            joined = subset.merge(table_frame, on="id", how="left", suffixes=("_doc", "_sql"))
            joined.insert(0, "source_table", table_name)
            frames.append(joined)

    if not frames:
        raise ValueError("No quote-related tables were found in the SQLite database")

    joined = pd.concat(frames, ignore_index=True, sort=False)
    joined.to_parquet(args.output, index=False)
    print(f"Wrote joined SQLite study table to {args.output}")


if __name__ == "__main__":
    main()