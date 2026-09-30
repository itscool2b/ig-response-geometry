"""Download public model artifacts at resolved immutable revisions, with hashes.

These are new runtime inputs, never claimed as recovery of historical weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new model manifest path')
    api = HfApi()
    result = {'purpose': 'new runtime inputs; historical identity unknown', 'models': []}
    specifications = [
        ('robotics-diffusion-transformer/rdt-170m', 'pytorch_model.bin'),
        ('robotics-diffusion-transformer/maniskill-model', 'rdt/mp_rank_00_model_states.pt'),
        ('google/siglip-so400m-patch14-384', None),
        ('google/t5-v1_1-xxl', None),
    ]
    for repository, filename in specifications:
        info = api.model_info(repository)
        if filename:
            path = Path(hf_hub_download(repository, filename, revision=info.sha))
            files = [path]
            root = path.parent
        else:
            names = [x.rfilename for x in info.siblings]
            extension = '*.safetensors' if any(n.endswith('.safetensors') for n in names) else '*.bin'
            root = Path(snapshot_download(repository, revision=info.sha,
                        allow_patterns=['*.json', '*.model', extension]))
            files = sorted(p for p in root.rglob('*') if p.is_file())
        result['models'].append({'repository': repository, 'revision': info.sha,
                                 'local_path': str(path if filename else root),
                                 'files': [{'path': str(p.relative_to(root)), 'sha256': digest(p),
                                            'bytes': p.stat().st_size} for p in files]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix('.partial.json')
        partial.write_text(json.dumps(result, indent=2) + '\n')
        print(repository, info.sha, flush=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
