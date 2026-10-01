# General research-paper builds

The default build is the identified general research paper. The anonymous flag removes author metadata, the public repository URL and acknowledgments for independent audit. Both use the pinned template in its neutral preprint mode, with no journal submission or review-status header. The historical filenames and TMLRAnonymous switch remain for compatibility. Neither build implies submission to a journal. The identified acknowledgment records the eight-month research project; AI contributions are disclosed on the first page.

With Python 3 and TeX Live (`texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-science`, `lmodern` on Debian/Ubuntu):

```sh
python scripts/build_paper.py
python scripts/build_paper.py --anonymous
```

Retrospective tables and figures come from the verified artifact at `analysis/revision/results/2026-09-30-v2`. Run `python scripts/build_revision_assets.py` in the analysis environment after any manuscript source, analysis artifact or display-code change. This first verifies the artifact, then writes `paper/tables_revision`, `paper/figures_revision`, the plain-text abstract and the complete cell/figure lineage registry `paper/figures_revision/lineage.json`. The registry contains source and display hashes, result identifiers, physical-record population hashes, estimands, units, grouping and limits. The PDF builder refuses to compile stale manuscript or generated display hashes. It does not turn missing historical context/checkpoint identities into authenticated ones. Older figures remain preserved in `paper/figures` and are not included by the revised manuscript.

Outputs are `paper/paper-revision.pdf` and `paper/paper-anonymous-draft.pdf`, with build logs beside them. The script refuses to overwrite the historical `paper/paper.pdf`. Intermediate TeX files are isolated in a temporary directory. Run both builds after scientific edits and inspect rendered pages, citation resolution, anonymity, links and metadata.

The two numerical tables are regenerated from the frozen partial snapshot in `analysis/numerical_case/2026-10-01-v1`. The asset builder runs its CPU-only reproducer, checks every recorded output hash, and copies the resulting TeX fragments. Manuscript lineage also binds the numerical input projection, reproducer and derived files. This checks the saved-data reporting; it does not rerun the GPU model, recover missing historical artifacts, or certify numerical accuracy.

The additional paired rescoring table and nested-grid counterexample are separate CPU artifacts. Their source, declared protocol, results and the appendix fragment are included in the manuscript lineage. They leave the canonical retrospective v2 outputs and the partial numerical snapshot unchanged. The curated source archive includes the appendix explicitly and rejects nested file-loading directives inside it.
Asset regeneration executes the aliasing toy and requires the full CPU environment, including PyTorch `2.14.1+cpu`; the minimal saved-data `requirements.txt` does not install Torch. The lock file remains a version snapshot. Follow the [checked Windows CPU installation procedure](../docs/cpu_reproduction.md) for explicit CPU-wheel selection and installation into a new isolated environment.

The exploratory episode-influence supplement is separate from the frozen paired primary analysis. Its producer authenticates the original inputs and records all eight cases and 240 whole-episode omissions. Asset regeneration verifies and reproduces this supplement before copying its appendix table and binding its source, protocol, outputs and recorded values into manuscript lineage. Omission ranges are influence diagnostics, not confidence intervals.

The official TMLR style, bibliography and license files preserve the exact upstream LF bytes from the commit recorded in `paper/tmlr-source.json`. The manifest separately retains the original CRLF checkout hashes as historical provenance; those are not accepted build or package identities. Git attributes prevent checkout newline conversion for these files. Run `python scripts/validate_template_provenance.py` to check the exact files and complete pinned metadata offline. Both package builders check the actual members before writing output. The build validator uses the same check before compilation.

The upstream repository's Apache 2.0 license is reproduced in `paper/tmlr-LICENSE`. The original `paper/tmlr.bst` header also retains its LPPL version 1 or later notice and copyright attributions. Both are preserved, with no assertion that one overrides the other; the public repository's MIT license does not relicense these third-party files. The source uses the TeX distribution's `fancyhdr` dependency. The official template requires the anonymous option for review and the `preprint` option for identified preprints. The FAQ's first-page AI-assistance disclosure is present in both versions.

The anonymous PDF alone does not make the repository or a source archive anonymous. Local manuscript-source and research-supplement candidates are separately assembled and inspected for identities, paths, asset metadata and links after changes. They remain subject to human author and distribution review. No accepted-paper flag or fabricated OpenReview identifier is used.
