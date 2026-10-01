"""Verify current revision artifacts; historical print-only audit is archived."""
import argparse
import json
from pathlib import Path

from analysis.revision.verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Identified output directory from analysis.revision.analyze")
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
