"""Execute only a separately gated, protocol-bound E05 parameter-null shard."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("metrics","base","protocol","e05-gate","native-registry","prepared","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--lang-dir",required=True)
    parser.add_argument("--checkpoint-path")
    args=parser.parse_args()
    from fp32_probe_cache import process_startup
    startup=process_startup("run")
    import weight_arrangement_control as control
    try:
        control.run(args,startup)
    except Exception as error:
        from experiment_io import canonical_json,file_hash
        path=args.out.with_name(args.out.name+".setup-or-run-failure.json")
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(canonical_json(dict(status="failed",exception_type=type(error).__name__,reason=str(error),
                startup=startup,script_sha256=file_hash(__file__),diagnostics=getattr(error,"diagnostics",{})))+b"\n")
        raise


if __name__=="__main__":
    main()
