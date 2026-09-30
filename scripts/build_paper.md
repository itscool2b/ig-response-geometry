# Working TMLR builds

The default build is an identified preprint. The anonymous flag removes author metadata, the public repository URL and acknowledgments, and uses the official anonymous TMLR style. Both are explicitly working drafts pending scientific and artifact gates. Neither is a submission package.

With Python 3 and TeX Live (`texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-science`, `lmodern` on Debian/Ubuntu):

```sh
python scripts/build_paper.py
python scripts/build_paper.py --anonymous
```

Outputs are `paper/paper-revision.pdf` and `paper/paper-anonymous-draft.pdf`, with build logs beside them. The script refuses to overwrite the historical `paper/paper.pdf`. Intermediate TeX files are isolated in a temporary directory. Run both builds after scientific edits and inspect rendered pages, citation resolution, anonymity, links and metadata.

The official TMLR style and bibliography files are unchanged copies from the commit recorded in `paper/tmlr-source.json`. Their upstream Apache 2.0 license is `paper/tmlr-LICENSE`. The source uses the TeX distribution's `fancyhdr` dependency. The official template requires the anonymous option for review and the `preprint` option for identified preprints. The FAQ's first-page AI-assistance disclosure is present in both versions.

The anonymous PDF alone does not make the repository or a source archive anonymous. A future review supplement must be separately assembled and inspected for identities, paths, asset metadata and links. No accepted-paper flag or fabricated OpenReview identifier is used.
