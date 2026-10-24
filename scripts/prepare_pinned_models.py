"""Download the pinned model revisions into HF_HOME and check every file hash.

`scripts/prepare_models.py` resolves the hub's current main; this script never
does. Each file is fetched at an exact commit and must match the SHA-256 the
Month 8 runtime manifest recorded (docs/tmlr-readiness/plan/execution/
runtime_model_manifest.json). Runs then work with HF_HUB_OFFLINE=1.

Usage: HF_HOME=DIR python scripts/prepare_pinned_models.py [--with-1b]
"""
import argparse
import hashlib
import json
import sys

PINNED = {
    "robotics-diffusion-transformer/rdt-170m": dict(
        revision="8aa386cac3bbfd9540676c75b3d767cc7f88a10a",
        files={"pytorch_model.bin": "2e87fccee0292b25ee4f714670dd7d8da8876ea092da3744db3292db070b7133"}),
    "google/siglip-so400m-patch14-384": dict(
        revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        files={"config.json": "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b",
               "model.safetensors": "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a",
               "preprocessor_config.json": "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8",
               "special_tokens_map.json": "2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26",
               "spiece.model": "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec",
               "tokenizer.json": "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920",
               "tokenizer_config.json": "d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb"}),
}
ONE_B = {"robotics-diffusion-transformer/maniskill-model": dict(
    revision="9622afab2b7ce2312a6cf1febc526928589b77eb",
    files={"rdt/mp_rank_00_model_states.pt": "3fa89cbf7bbb1b2559b29f9ae153b431541cdaa0b3f1ee44b83b9934599b9fb8"})}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(models):
    from huggingface_hub import hf_hub_download
    report = {}
    for repo, spec in models.items():
        entry = dict(revision=spec["revision"], files={})
        for name, expected in spec["files"].items():
            path = hf_hub_download(repo, name, revision=spec["revision"])
            actual = sha256(path)
            entry["files"][name] = dict(path=path, sha256=actual, matches=actual == expected)
        report[repo] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-1b", action="store_true")
    args = parser.parse_args()
    models = dict(PINNED, **(ONE_B if args.with_1b else {}))
    report = prepare(models)
    print(json.dumps(report, indent=1, sort_keys=True))
    bad = [(repo, name) for repo, entry in report.items() for name, f in entry["files"].items() if not f["matches"]]
    if bad:
        print("HASH MISMATCH:", bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
