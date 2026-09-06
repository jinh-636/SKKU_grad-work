import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from aios.storage.policy import SemanticAnalysisError, SemanticAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--file", type=Path, help="UTF-8 text file to analyze")
    args = parser.parse_args()

    try:
        if args.file is not None:
            content = args.file.read_text(encoding="utf-8")
        elif args.text is not None:
            content = args.text
        elif not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            parser.error("provide --text, --file, or text on stdin")

        analyzer = SemanticAnalyzer.from_config()
        profile = analyzer.analyze(content)
    except (SemanticAnalysisError, OSError, UnicodeError, ValueError) as error:
        print(f"Metadata analysis failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(profile), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
