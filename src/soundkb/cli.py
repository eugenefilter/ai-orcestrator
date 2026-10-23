"""Command-line interface for SoundKB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soundkb.analysis.compare import compare_audio_files
from soundkb.analysis.report import write_report


def build_parser() -> argparse.ArgumentParser:
    """Build the SoundKB command parser."""
    parser = argparse.ArgumentParser(prog="soundkb", description="SoundKB audio knowledge base")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    analyze = sub.add_parser("analyze", help="Analyze an audio file (report.json + report.png), no storage")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--out", type=Path, required=True)
    ingest = sub.add_parser("ingest", help="Analyze an audio file and store it in the knowledge base")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--title")
    ingest.add_argument("--tag", action="append", default=[])
    ingest.add_argument("--note", default="")
    ingest.add_argument("--allow-long", action="store_true")
    search = sub.add_parser("search", help="Semantic search")
    search.add_argument("query")
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--kind")
    search.add_argument("--limit", type=int, default=5)
    compare = sub.add_parser("compare", help="Compare a rendered candidate with a reference file")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    return parser


def main() -> None:
    """Run the SoundKB CLI."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    elif args.command == "analyze":
        print("\n".join(write_report(args.path, args.out)["summary"]))
    elif args.command == "compare":
        print(json.dumps(compare_audio_files(args.reference, args.candidate), indent=2))
    else:
        from soundkb.config import load_settings
        from soundkb.kb import KBError, KnowledgeBase

        try:
            kb = KnowledgeBase(load_settings())
            if args.command == "ingest":
                entry = kb.ingest_audio(args.path.expanduser().resolve(), args.title, args.tag, args.note, allow_long=args.allow_long)
                print(f"id: {entry['id']}\n" + "\n".join(entry["analysis"]["summary"]))
            else:
                for hit in kb.search(args.query, args.kind, args.tag, limit=args.limit):
                    print(f"{hit['score']}  {hit['id']}  [{hit['kind']}] {hit['title']}  tags={hit['tags']}")
        except KBError as exc:
            sys.exit(str(exc))
