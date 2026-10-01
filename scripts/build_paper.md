# Working TMLR builds

The default build is an identified preprint. The anonymous flag removes author metadata, the public repository URL and acknowledgments, and uses the official anonymous TMLR style. Both are explicitly working drafts pending scientific and artifact gates. Neither is a submission package.

With Python 3 and TeX Live (`texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-science`, `lmodern` on Debian/Ubuntu):

```sh
python scripts/build_paper.py
python scripts/build_paper.py --anonymous
```

Retrospective tables and figures come from the verified artifact at `analysis/revision/results/2026-09-30-v2`. Run `python scripts/build_revision_assets.py` in the analysis environment after any manuscript source, analysis artifact or display-code change. This first verifies the artifact, then writes `paper/tables_revision`, `paper/figures_revision`, the plain-text abstract and the complete cell/figure lineage registry `paper/figures_revision/lineage.json`. The registry contains source and display hashes, result identifiers, physical-record population hashes, estimands, units, grouping and limits. The PDF builder refuses to compile stale manuscript or generated display hashes. It does not turn missing historical context/checkpoint identities into authenticated ones. Older figures remain preserved in `paper/figures` and are not included by the revised manuscript.

Outputs are `paper/paper-revision.pdf` and `paper/paper-anonymous-draft.pdf`, with build logs beside them. The script refuses to overwrite the historical `paper/paper.pdf`. Intermediate TeX files are isolated in a temporary directory. Run both builds after scientific edits and inspect rendered pages, citation resolution, anonymity, links and metadata.

The two numerical tables are regenerated from the frozen partial snapshot in `analysis/numerical_case/2026-10-01-v1`. The asset builder runs its CPU-only reproducer, checks every recorded output hash, and copies the resulting TeX fragments. Manuscript lineage also binds the numerical input projection, reproducer and derived files. This checks the saved-data reporting; it does not rerun the GPU model, recover missing historical artifacts, or certify numerical accuracy.

The additional paired rescoring table and nested-grid counterexample are separate CPU artifacts. Their source, declared protocol, results and the appendix fragment are included in the manuscript lineage. They leave the canonical retrospective v2 outputs and the partial numerical snapshot unchanged. The curated source archive includes the appendix explicitly and rejects nested file-loading directives inside it.

The official TMLR style and bibliography files are unchanged copies from the commit recorded in `paper/tmlr-source.json`. Their upstream Apache 2.0 license is `paper/tmlr-LICENSE`. The source uses the TeX distribution's `fancyhdr` dependency. The official template requires the anonymous option for review and the `preprint` option for identified preprints. The FAQ's first-page AI-assistance disclosure is present in both versions.

The anonymous PDF alone does not make the repository or a source archive anonymous. A future review supplement must be separately assembled and inspected for identities, paths, asset metadata and links. No accepted-paper flag or fabricated OpenReview identifier is used.
