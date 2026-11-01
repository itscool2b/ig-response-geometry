# Manuscript and research downloads

The [canonical manuscript](../paper.tex) and [bibliography](../references.bib) define the current revision. The identified Word copy is in [Online](Online/). It uses full inline citations, one-inch margins and 12-point Times New Roman. The abstract has 249 words. The three main figures and two tables are accompanied by a numbered supplement.

The bibliography contains 46 references: 42 primary papers, including 30 from 2022 through 2026. The manuscript distinguishes seven preplanned method contrasts from exploratory preferences under raw means, scaled means and episode medians. New retrospective bootstrap results are kept in a separate reporting layer; frozen scientific records and protocols are preserved.

[downloads.json](downloads.json) lists current public packages, their byte sizes and SHA-256 checksums. These public URLs are separate from private reviewer access.

| Package | Contents and purpose |
| --- | --- |
| Manuscript sources | Current article, supplement, bibliography, figures and tables, with instructions for rebuilding the PDFs. |
| Research supplement | Current reporting code and manuscript sources, preserved small data, and the retrospective bootstrap report. |
| Complete study data | Full archived tensors, plans, outcomes, code and results for the registered CPU analysis replay. |
| Stored four-task evidence | Preserved observations and numerical evidence underlying the separate diagnostic collection. |
| Regeneration kit | Inputs and instructions for reconstructing selected policy contexts. This is a separate workflow from CPU rescoring. |

The saved curves reproduce response summaries, plots, success splits and completeness checks on a CPU. They cannot regenerate original robot executions or rankings because the original observations and diffusion noise were not saved. The matched experiment preserves those inputs. The regeneration evidence records exact reconstruction of selected contexts, not hardware independence in general.

For the complete registered analysis, extract the complete study data package into a new directory, install its exact Linux x86-64 Python 3.12 runtime lock, and run its `reproduce.py --out NEW_DIRECTORY`. Outputs must be written outside the authenticated input tree. The verifier checks episode membership, analysis plans, counts, missingness, ties, decisions and numerical equivalence. Numerical equivalence is distinct from byte identity; any differences remain explicitly reported.

The retrospective report uses 10,000 paired episode-resampling draws, seed 930011. Its winner frequencies describe stability of the recorded samples and do not estimate a probability that a true population winner changes. The [reporting guide](../../analysis/manuscript_revision/nhsjs_audit_2026_11_01/README.md) gives its reproduction command.

Current packages exclude superseded manuscript text and internal drafting notes. Earlier archives and verification records remain preserved as historical records. AI tools assisted with code, analyses, scientific checks, manuscript text and reviewer documentation; the manuscript contains the full disclosure.

[MANIFEST.json](MANIFEST.json) binds the published Word copy, supplement, guide and public download index. The clean Standard and Online manuscripts render to 17 and 20 pages in LibreOffice. Microsoft Word pagination and equation inspection remain pending; other renderers do not certify that check. This is an AI-assisted working package, with no journal submission or claim of compliance with the author-only-writing condition.
