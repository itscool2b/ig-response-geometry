# Local manuscript builds

The R1 source uses Times New Roman 12 point, Letter paper, one-inch margins, single spacing and unnumbered bold headings. Five figures, four tables and 34 references accompany a 221-word abstract and six keywords. Both clean Word manuscript views must render within 20 pages. Formatting success does not establish journal submission or acceptance.

## Rebuild current displays and PDFs

Use the [full CPU environment](../docs/cpu_reproduction.md). These commands need no GPU, model or simulator run:

```powershell
python scripts/build_nhsjs_revision_assets.py
python scripts/build_paper.py --engine auto --keep-intermediates-dir runs/identified-build
python scripts/build_paper.py --anonymous --engine auto --keep-intermediates-dir runs/blinded-build
```

Use fresh intermediate directories. The display builder reads authenticated saved outputs and the curated nonidentifying projection in `analysis/manuscript_revision/nhsjs_r1/inputs`. It records hashes, actual plotted populations and display outputs in `paper/figures_revision/r1_lineage.json`. It leaves earlier figures, source data and frozen analyses unchanged. `scripts/build_revision_assets.py --reproduce` remains the historical October 3 reporting reproduction workflow; it does not produce the R1 displays.

`--engine auto` locates installed Tectonic or the existing Windows cache at `~/.cache/tmlr-tectonic-0.17.0/tectonic.exe`. An explicit executable path is accepted. Default compilation uses cached resources and untrusted mode. `--allow-resource-downloads` allows missing public TeX resources into the cache without uploading the manuscript. The fontspec source requires XeTeX/Tectonic, not pdfLaTeX.

The builder validates source, numerical and display hashes before and after compilation. Preserved AUX and BBL files include hashes tied to the source and PDF. Identified default builds update `paper/paper.pdf` and `paper/paper-revision.pdf` together. Anonymous default builds update `paper/paper-anonymous-draft.pdf`; opposing reserved names and the historical archive are protected.

The anonymous extended mathematical supplement is standalone:

```powershell
tectonic --only-cached --untrusted --reruns 1 --outdir runs/math paper/supplementary_derivations.tex
```

## Word delivery

The revision delivery includes the source-first Word tooling, original templates and immutable comparison baseline. It accepts explicit repository, Pandoc, compiled-source, citation processor, renderer and output paths. Both citation variants derive from the same ordered key groups and repaired NHSJS CSL styles. Equations are native OMML; tables and text remain editable. Generated citations are plain output, not tested live Zotero fields. The Zotero import and styles are retained for handoff.

A pinned Open XML PowerTools wrapper creates real tracked revisions with author `Author`. Its accepted projection must equal the revised anonymous master; its rejected projection must equal the submitted baseline, including effective formatting, mathematics, tables, image bytes and hyperlink targets. Equivalent repairs to pre-existing baseline XML are documented without modifying the original files. Portable LibreOffice renders in an isolated profile. The checked PDF, page images and DOCX hashes must match; page counts are measured from actual rendering.

`scripts/export_online_docx.py` is the preserved October 4 exporter with literal TeX equations. It is not the current native-equation delivery workflow. Earlier Word files and receipts remain historical.

## Source and research archives

```powershell
python scripts/package_paper_sources.py --out runs/source-package
python scripts/package_research_supplement.py --working-copy --out runs/research-package
```

Each destination must be new. The source package curates the anonymous manuscript and dependencies. The research package captures an explicit source snapshot, saved inputs, analyses, numerical outputs and current display builder. Its manifest hashes exact members. A recorded base commit is not a claim that later working-copy edits came from that commit. Concurrent changes invalidate packaging.

The separate reviewer archive includes preserved R1 contexts and raw tensors required for registered-analysis replay. Reproducing saved outputs is distinct from GPU regeneration of a policy execution. Verify all manifests after extraction, then use the archive's reproduction launcher. Do not claim complete analysis reproduction from the smaller archive that omits the raw tensor bundle.

Historical templates and their Apache 2.0/LPPL notices remain preserved; see [NOTICE](../NOTICE) and [template provenance](../paper/tmlr-source.json). Every delivered archive and document requires its own exact-byte checks and receipt.
