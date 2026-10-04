# Polished publication snapshot verification

The manuscript snapshot is identified by tag `manuscript-2026-10-04-polish`, with reporting version `2026-10-03-v2`. Its canonical source SHA-256 is `9952399f5a6e1ee2ff02741436c570a0459b7653ab2086d8f6a8fb19245a3741`. The blinded Standard PDF has 19 pages, the identified reading copy has 19 and the Online Word document has 20. The abstract contains 248 words and six keywords. The manuscript retains five figures, two numbered tables, 17 references and two proof appendices. Word uses three native table components.

## Final polish P1-P5

The polish preserves the 248-word abstract byte-for-byte, all scientific values and the author-supplied facts, declarations and AI statement. Existing E1-E20 and O1-O4 remain implemented; their prior ledger and acceptance receipts are preserved as historical records.

| Item | Disposition | Verified change |
|---|---|---|
| P1 | Implemented | Removed the single Discussion float barrier and split the opening subsection 2.2 paragraph before Define. Subsection 2.2 starts on page 2 and Discussion on page 13 in both PDFs; Define stays with Equation 1. Existing figure placements, fonts, margins and appendix layout are retained. |
| P2 | Implemented | Reference 5 uses the official BMVA archive PDF URL, with every other bibliography field unchanged. Every Online occurrence is regenerated from the compiled bibliography. |
| P3 | Implemented | All five AUC operator expressions now use upright ordinary math lettering in Equations 7-8 and Table 2's caption, correcting punctuation and operator spacing. |
| P4 | Implemented | The inline RMS formula uses the half-power form. Nonnegative squared residuals and positive action dimension make it equivalent to the square-root form; the B > 0 normalization condition is retained. |
| P5 | Implemented | The two specified prose Ns are upright. Epsilon and the smooth-action-map qualification are unchanged. |

## Corrections and author information

All 14 corrections in the historical [Final-v5 ledger](final_v5_checklist.md) are retained. The preserved [October 4 edit ledger](exact_edits_checklist.md) records all E1--E20 and O1--O4, including existing correct items. The retained exact-edit wording identifies all budget pairs, conditional selection requirements, endpoint-normalized RMS for positive baseline residuals, the recorded versus inferred solver-step provenance, and the exact scope of language sensitivity. Small baseline residuals are necessary but insufficient for the observed extreme vision-insertion calls. The epsilon statement is explicitly conditional on a smooth action map. Reference 2 uses the official ICML proceedings entry. The AUC example displays 0.0102, above the unchanged 0.01 criterion. Model/task wrapping, caption math, Equation 8 punctuation and the Online superscript markers are checked in rendered output.

Independent recomputation from the sealed criterion rows gives 6,048 budget checks: 302 violated, 5,602 satisfied and 144 not applicable. This subset is the union of 5,760 cross-budget comparisons and 288 conditional completeness checks, not the complement of repeatability checks. All 5,544 other checks pass. Thus the total remains 11,592, with 11,146 satisfied. The 144 not-applicable results are top-five-percent overlap comparisons with no selected state groups. Adjacent pairs contribute 153 failures, smallest-to-largest pairs 126, and completeness thresholds 23. No scientific inputs or results changed.

The author supplied Windermere Preparatory School, Windermere, Florida, United States for the Online author block, with the existing corresponding email. As specifically requested, the identified reading PDF retains its original author block. Identified acknowledgments contain the supplied funding, contribution and competing-interest declarations. The author confirms both mentors read the manuscript and approved acknowledgment and submission. These are author-confirmed facts, separate from automated verification. The blinded PDF and source exclude the identifying block and declarations. The AI statement is unchanged.

## Builds, document checks and tests

Both PDFs are rebuilt with cached Tectonic; Word is exported from fresh intermediates with the existing Pandoc workflow. Source/bibliography parity, repeated identical full citations, native tables, figures, equation markup and absence of tracked changes/comments are checked against the delivered bytes. Every PDF and Word page is rendered for visual review, with required fonts and margins preserved. Source/archive anonymity checks include metadata, links, embedded objects and assets. Both archives are extracted into fresh directories, member hashes checked, and manuscripts rebuilt for text and pixel comparisons with the accepted blinded PDF.

The fresh focused reporting/build/packaging and native Word suites passed with no skips. This polish changes no Python implementation, schema or package allowlist. The earlier four-decimal display, author markers and package scope remain intact. Prior CPU results (783 passed in each repository, 95 private supplemental tests) remain evidence only for unchanged tested inputs, authenticated by hash; they are not newly rerun full suites. Word preserves 292 inline and 25 display math spans from 293 source inline spans, with exactly one duplicated Table 2 caption span removed by the existing exporter.

All 155 authenticated scientific inputs, 99 raw files, frozen scientific outputs and the current reporting manifest remain byte-identical to the accepted October 4 snapshot. The prior clean reproduction of all 36 frozen scientific artifacts is retained, with scientific inputs, reproducer code and outputs reauthenticated. The follow-up requires no model, simulator, training or new scientific experiment. Git's existing canonical-LF and permitted CRLF-checkout hashes for raw files are preserved; publication does not rewrite the raw Git blobs.

## Exact-byte acceptance and publication

The untracked local delivery is `artifacts/current`. Its current `Manifest.json` hashes each delivered file; `Verification.json` identifies exact-byte document, scientific and test receipts. Current records, original snapshots, superseded build attempts and final synchronization receipts are in the sibling `paper-local-records-2026-10-04-final-polish` directory. The October 3 publication and October 4 exact-edits records, Final-v5, Round 3 and older records remain historical. A changed file invalidates its corresponding receipt.

The public commit and annotated tag use only Arjun Bajpai's configured identity, without an AI co-author trailer. The separate publication receipt records the actual remote commit/tag outcome. The private repository receives only guarded shared-file and delivery updates; its HEAD, all 2,371 preexisting deletions, the five recorded repository-specific variants and historical text are preserved. Publication is a normal public-branch update, with no force push or history rewrite. The October 3 and original October 4 tags remain unchanged. GitHub account-email association remains pending; the named Git author and committer are verified.

The editor inquiry and prior-version disclosure are prepared locally and remain unsent. The inquiry covers earlier versions, eligibility, fees, copyright, journal placement, anonymous archive access and section order. Journal eligibility/form requirements and answers, an outside-scientist cold read, and author-handled AI policy remain pending. Repository publication is not journal submission or acceptance; no payment or submission has been made.
