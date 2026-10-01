# Saved numerical case study

The [versioned bundle](2026-10-01-v1/README.md) reports the sealed-results snapshot labeled October 1, 2026, 03:36 UTC. It retains all 24 planned contexts: 12 complete, three source-unreached and nine available but unsealed. The completed contexts cover six episodes across four 170M tasks. No production numerical setting is approved.

Reproduce the tables and saved-tensor diagnostics from the repository root, with Python 3.12 standard library only:

```text
python analysis/numerical_case/2026-10-01-v1/summarize.py --output runs/numerical-case-reproduction
```

With pytest installed, run the separate verification tests from the repository root:

```text
python -m pytest tests/test_numerical_case.py -q
```

The manifest binds the input projection, reproducer and all derived files. The projection contains the complete recorded roster/checks/metrics and selected authenticated YCB tensor values. It does not contain every raw artifact from the GPU campaign. Reproduction verifies the reported calculations over saved evidence; it does not repeat model execution or the complete original raw-artifact audit.

The retrospective 99-file analysis is separate, under [analysis/revision](../revision/README.md). Its historical inputs and these newly collected numerical contexts are not pooled.
