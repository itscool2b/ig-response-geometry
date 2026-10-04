# Local manuscript builds

The current source uses a 12-point Letter article layout with one-inch margins and Times New Roman. It has NHSJS Standard citations, five figures, two numbered tables, 17 references and two proof appendices. The blinded Standard PDF has 18 pages, the identified reading copy has 19 and the Online Word document has 20. The one-paragraph abstract contains 245 words. Six keywords are retained. See the tracked [final local verification](../docs/manuscript_revision/final_local_verification.md) and [Final-v5 ledger](../docs/manuscript_revision/final_v5_checklist.md) for document and package checks. Formatting and local build success do not imply journal approval or completed submission.

## Compile the canonical source

The portable build entry point is the already assembled `paper/paper.tex`. Later authorized corrections are applied to that source. Use the asset and compilation commands below; replaying the historical approval assembly is not part of a manuscript build or research-package reproduction.

Historical assembly tools, baselines and review records are preserved in the separate local records archive. They are not required for current builds. Restoring an older source would discard later corrections and invalidate current asset lineage and PDF receipts.

## CPU assets and cached Tectonic

Use the [full CPU environment](../docs/cpu_reproduction.md). No model, training or simulator run is needed.

```powershell
python scripts/build_revision_assets.py --reproduce
python scripts/build_paper.py --engine auto --keep-intermediates-dir runs/identified-build
python scripts/build_paper.py --anonymous --engine auto --keep-intermediates-dir runs/blinded-build
```

Use fresh intermediate directories. Asset generation reproduces all 36 retained scientific artifacts when `--reproduce` is supplied. Ordinary refresh verifies the retained reproduction receipt and current inputs. Reporting version `2026-10-03-v2` creates figures, table, prose macros, abstract and lineage without changing frozen protocols or outputs.

`--engine auto` locates an installed Tectonic or the existing Windows cache at `~/.cache/tmlr-tectonic-0.17.0/tectonic.exe`. An absolute executable path is also accepted. Default builds use `--only-cached` and `--untrusted`. Missing public TeX resources cause failure. The explicit `--allow-resource-downloads` option permits initial cache setup using public TeX resources. It does not upload the manuscript. This fontspec source cannot use pdfLaTeX. The older engine route remains available for compatible archived sources.

The builder checks manuscript, display, reporting and frozen-evidence hashes before and after compilation. Stale assets fail instead of producing a misleading PDF. AUX and BBL files can be preserved for Word conversion, with their source and PDF hashes.

## Names and protected paths

An identified build to `paper/paper.pdf` or `paper/paper-revision.pdf` updates both with identical bytes and corresponding logs. A custom output receives only that file. The anonymous default is `paper/paper-anonymous-draft.pdf`. Anonymous builds cannot overwrite identified names and identified builds cannot overwrite the reserved anonymous name. The historical `legacy/pre-response-geometry` archive is protected. Temporary compiler files never replace historical sources.

The final local delivery is under `artifacts/current`; completed checks belong in the tracked [final local verification](../docs/manuscript_revision/final_local_verification.md). The preceding `2026-10-02` and `2026-10-02-humanized` delivery folders are historical. A blinded PDF does not make the complete repository anonymous. Check source packages, metadata, links and figure files separately.

## Online Word export

The Online Word document contains three native table components representing two numbered tables, repeated full citations and literal equation source. The checked 20-page export was regenerated from the same corrected manuscript as the 18-page blinded Standard PDF. Repeated full citations and literal TeX equations affect pagination; both formats are checked against the 20-page template limit. Earlier Word files remain historical.

Use local Pandoc and the bundled document Python with python-docx. The reference document is the official Online template. Source and bibliography must match the compiled intermediate files.

```powershell
python scripts/export_online_docx.py --pandoc PATH_TO_PANDOC --template PATH_TO_ONLINE_TEMPLATE --intermediates runs/identified-build --output "runs/How Scoring Responses Change Attribution Evaluation in a Diffusion Policy.docx"
```

This creates native Word tables, embeds current PNG figures and repeats full bibliography entries at every citation. Reference parity is checked against the compiled bibliography count, currently 17; missing or extra entries fail export. Equations retain literal TeX source as requested for the journal's LaTeX route. Content and export-verification JSON files are also written. Render Word locally and inspect every page. The Windows document renderer uses native LibreOffice and bundled Poppler.

## Source and research packages

The 17-member Final-v5 source archive rebuilt to the same 18-page blinded output in text and pixels. All 57 pages across the blinded PDF, identified PDF and rendered Online Word document passed visual inspection. The current research archive uses an explicit working-copy snapshot. Final archive extraction, manifests and rebuild checks are recorded beside the delivery. The packages in preceding delivery folders remain historical. The unchanged scientific inputs retain their separately verified reproduction evidence.

```powershell
python scripts/package_paper_sources.py --out runs/source-package
python scripts/package_research_supplement.py --working-copy --out runs/research-package
```

Each destination must be new. The source package exports an anonymized manuscript, allowlisted dependencies and required third-party notices. The research package's explicit `--working-copy` option captures exact current bytes, including the already assembled canonical manuscript and reviewed reporting files. Its scope is scientific reproduction and compilation of that source, not historical prose replay. Its manifest hashes every member and the complete snapshot. The recorded Git commit is a base reference, not a claim that uncommitted changes came from that commit. Concurrent changes invalidate packaging. Use `--commit REVISION` only when the committed version is intended.

Both routes preserve raw input and historical-output checks. Extract each archive into a fresh directory, verify its manifest, rebuild the manuscript and reproduce scientific outputs. Review the exact delivered bytes and record hashes. Local completion remains separate from author and journal decisions.

The official TMLR files remain pinned and verified for historical compatibility. Their Apache 2.0 and separate LPPL notices are preserved. The new `nhsjs.bst` is a project-specific numbered bibliography style. See [NOTICE](../NOTICE) and [template provenance](../paper/tmlr-source.json).
