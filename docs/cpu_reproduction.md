# CPU installation and reproduction

The full local route was installed from scratch into a new virtual environment on Windows x86-64 with Python 3.12.14 on October 1, 2026. It uses the official PyTorch CPU wheel index, then the pinned package snapshot from PyPI. It needs no GPU, model checkpoint or simulator. A new environment is required; these commands are not instructions to modify an existing research environment.

The installation check confirms Python environment isolation, Torch `2.14.1+cpu`, `torch.version.cuda is None`, imports of the analysis/test dependencies, and a successful `pip check`. It does not install Python itself or a TeX distribution. Other operating systems and Python versions were not independently installed in this check.

## Windows PowerShell

Run from the repository root, using Python 3.12.14. If `python` selects another interpreter, replace the first command's executable with the Python 3.12.14 path. The destination `.venv-cpu` must not already exist.

```powershell
python -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip --isolated install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.\.venv-cpu\Scripts\python.exe -m pip --isolated install --index-url https://pypi.org/simple -r requirements-cpu-lock.txt
.\.venv-cpu\Scripts\python.exe -m pip --isolated check
.\.venv-cpu\Scripts\python.exe -c "import sys, torch; assert sys.prefix != sys.base_prefix; assert torch.__version__ == '2.14.1+cpu' and torch.version.cuda is None; print(sys.version, torch.__version__)"
```

The first install explicitly selects the CPU build. The lock's `torch==2.14.1` accepts the installed `2.14.1+cpu` local version; the second command retains it while pinning the remaining dependencies. `--isolated` avoids taking an alternate package index from user pip configuration. Wheels may be reused from pip's download cache; no installed packages or site-packages directory are copied from the earlier environment. The [official PyTorch installation guide](https://pytorch.org/get-started/locally/) describes the platform and CPU-wheel selection.

Use the explicit virtual-environment Python path for the commands below, or activate that environment first. The analysis commands themselves require no network access:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
.\.venv-cpu\Scripts\python.exe -m pytest tests -q
.\.venv-cpu\Scripts\python.exe -m analysis.revision.verify analysis/revision/results/2026-09-30-v2
.\.venv-cpu\Scripts\python.exe -m analysis.revision.analyze --output runs/reproduce-canonical --draws 10000 --seed 0
.\.venv-cpu\Scripts\python.exe -m analysis.revision.verify runs/reproduce-canonical
.\.venv-cpu\Scripts\python.exe -m analysis.paired_rescoring.analyze --output runs/reproduce-paired
.\.venv-cpu\Scripts\python.exe -m analysis.paired_rescoring.influence --output runs/reproduce-influence
.\.venv-cpu\Scripts\python.exe -m analysis.paired_rescoring.influence --verify runs/reproduce-influence
.\.venv-cpu\Scripts\python.exe analysis/numerical_case/2026-10-01-v1/summarize.py --output runs/reproduce-numerical
.\.venv-cpu\Scripts\python.exe -m analysis.revision.nested_grid_aliasing --output runs/reproduce-aliasing.json
.\.venv-cpu\Scripts\python.exe scripts/build_revision_assets.py
.\.venv-cpu\Scripts\python.exe scripts/validate_template_provenance.py
```

Choose fresh output destinations for every reproduction. The frozen source outputs remain unchanged. Asset regeneration writes current display files and their lineage registry; run it in a copied extraction if preserving the unpacked archive byte for byte. PDF compilation is a separate step described in [build_paper.md](../scripts/build_paper.md).

The influence supplement is explicitly exploratory and keeps all original paired results. CPU unit tests include synthetic model substitutes; passing them does not replicate the GPU campaigns, qualify an IG budget, authenticate missing historical contexts or demonstrate useful rankings. If a command fails, retain its output and report the actual failure rather than replacing it with a later successful retry.

## Minimal saved-data route

The canonical and paired analyses and exploratory influence require only the saved-data dependencies in `requirements.txt`. The frozen numerical projection uses the Python standard library. The full route above is required for the Torch aliasing example, complete CPU tests and manuscript asset regeneration. `requirements-cpu-lock.txt` remains a version snapshot; this page supplies the explicit, checked Windows installation procedure around it.

On Linux, virtual-environment Python is normally `.venv-cpu/bin/python`; select the matching CPU wheel through the official index and keep the same pinned dependencies. That path adaptation is guidance, not a claim that this pass independently installed or tested Linux. GPU collection has different dependencies and authorization requirements.
