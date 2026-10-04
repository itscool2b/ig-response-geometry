# Round 3 revision ledger

This ledger covers all 54 checkbox actions in `round3_fix_list.md`, including its aggregate and nested actions, and all 34 issue identifiers R1-R30 and N1-N4. Source line numbers refer to the supplied checklist, SHA-256 `9db60076838097848519fc3f74f9d58d7f980b7c6001e6832298fb1354c5d5c3`. The exact checklist and pre-edit file hashes are preserved in the sibling local records directory `paper-local-records-2026-10-03-round3`.

This is the current revision ledger. The [preceding ledger](revision_checklist.md) remains historical. Every action below has a local disposition. Local checks cover the source, analysis, software and prepared documents; they do not supply missing author facts or journal decisions. Final rendering, extracted-package acceptance and public/private synchronization are bound to the exact delivered bytes by [local verification](final_local_verification.md), `artifacts/current/Manifest.json` and `Verification.json`. A source edit requires fresh document and package receipts before acceptance.

## Authorized scope and evidence

All three tiers, including figure and typesetting polish, apply to the current study. The author chose to retain its existing population; optional rescoring of excluded cohorts is not performed. Scientific inputs and frozen results remain unchanged. Reporting version `2026-10-03-v2` adds descriptions and sensitivity summaries. No model runs, pushes, uploads, editor messages or submission are part of this local work. The existing AI statement remains unchanged at the author's request.

Author affiliation details, funding/contribution/competing-interest declarations, journal answers, an independent scientist's cold read and mentor sign-off remain author tasks. The author chose to supply those facts later. Do not invent declarations or describe a draft inquiry as sent. These constraints take precedence over action language in the attached checklist.

Evidence abbreviations:

- **P:** [manuscript](../../paper/paper.tex), [proof appendix](../../paper/appendix_aliasing.tex), [bibliography](../../paper/references.bib). Section numbers describe the revised structure; equation numbers in actions refer to the checklist until the rebuilt source resolves them.
- **R:** [report generator](../../analysis/manuscript_revision/report.py), `analysis/manuscript_revision/results/2026-10-03-v2/`, and [asset generator](../../scripts/build_revision_assets.py), with fresh output and reproduction receipts.
- **F:** [figure/table generator](../../analysis/manuscript_revision/figures.py), `paper/figures_revision/`, `paper/tables_revision/`, and `lineage.json`.
- **N:** [numerical sample](../../analysis/numerical_case/2026-10-01-v1/README.md), its `inputs/sealed_v6.json` and `outputs/`, plus R projections.
- **D:** [Online exporter](../../scripts/export_online_docx.py), [export tests](../../tests/test_online_docx.py), [build](../../scripts/build_paper.py), [source packager](../../scripts/package_paper_sources.py), [research packager](../../scripts/package_research_supplement.py).
- **Q:** fresh Round 3 local QA receipts and regenerated `artifacts/current/Manifest.json` and `Verification.json`; prior receipts are historical only. The receipt directory is `paper-local-records-2026-10-03-round3/qa`, beside the repositories. It includes the source-bound Tier 1 and scientific reviews, PDF/Online and visual checks, suite results, source-package verification and final research-package/reproduction checks.
- **A:** `artifacts/current/author-items/` contains the unsent inquiry, prior-version note and outstanding author facts, prepared in the Round 3 local records directory before final delivery refresh.

Figure 4 contains the target-reversal and aliasing constructions; Figure 5 contains numerical diagnostics. Section 3.4 covers target reversal; Section 3.5 covers numerical checks and refers back to Figure 4B. Both figures precede Discussion in the checked rendering.

## Paper actions

| Action | Checklist line / issue | Required disposition | Evidence | Status |
|---|---|---|---|---|
| R3-01 | 28 / R3 | Complete the seven provenance, blinding and terminology actions below coherently. | P 2.5, 2.7, Appendix B; Q anonymity | Local checks passed |
| R3-02 | 29 / R3 | Remove the month label; describe excluded sets by tasks, saved calls and budget. | P 2.5 | Local checks passed |
| R3-03 | 30 / R3, R5 | Replace authentication language with seed/episode relationships and differing residuals. | P 2.5; R cohort comparison | Local checks passed |
| R3-04 | 31 / R3 | Remove internal integrity/snapshot phrasing; retain retrospective median emphasis. | P 2.5-2.7 | Local checks passed |
| R3-05 | 32 / R3, R20 | Explain the unavailable original checkpoint at its first recorded pretrained identity. | P 2.5, 4.2 | Local checks passed |
| R3-06 | 33 / R3 | Use endpoint-inclusive average in the estimator and aliasing extension. | P 2.3, Appendix B | Local checks passed |
| R3-07 | 34 / R3 | Give context counts: 24 planned, 12 complete, nine unfinished, three selected calls never reached. | P 2.7; N roster | Local checks passed |
| R3-08 | 35 / N2 | Separate missing headline noise from retained noise in the collection capped at 13 calls per episode. | P 2.1, 2.7, Table 1, 4.2; N | Local checks passed |
| R3-09 | 36 / N3 | Restrict roundoff correction to self-reference; distinguish intervention undershoots and untrimmed AUC/difference values. | P 2.5-2.6; R endpoints | Local checks passed |
| R3-10 | 37 / R24 | Define fully in-range curves; require positive weight on a residual strictly between 0 and B for strict decrease. | P 2.4, 3.1, Appendix A.1 | Local checks passed |
| R3-11 | 38 / R5 | Explain shared initial-state identifiers, differing B in 750 aligned calls and later divergence; seven shared extreme episodes are not independent evidence. | P 2.5, 3.3; R | Local checks passed |
| R3-12 | 39 / R1 | Define Q/N, ranking labels, deletion direction and first call before use; connect range terms to Methods. | P abstract, Introduction, 2.1, 2.4 | Local checks passed |
| R3-13 | 40 / R1, R25 | Plain abstract wording, simulated PickCube task and extreme calls; keep word limit. | P abstract; generated abstract; Q | Local checks passed |
| R3-14 | 41 / R6 | Separate guaranteed signs from measured prevalence and magnitude through the four actions below. | P 2.6, 3.1-3.3; R | Local checks passed |
| R3-15 | 42 / R6 | Explain zero-epsilon limit and convex rescaling without claiming new convexity theory. | P 3.1, Appendix A.1 | Local checks passed |
| R3-16 | 43 / R6 | Label the in-range contribution nonnegative by construction. | P 3.3; R decomposition | Local checks passed |
| R3-17 | 44 / R6 | Report fully in-range share and positivity among overshooting calls. | P 3.2-3.3; R paired summary | Local checks passed |
| R3-18 | 45 / R6 | Restrict predicted per-call signs to fully in-range calls. | P 2.6 | Local checks passed |
| R3-19 | 46 / R8 | Scope threshold flips and give response-matched calibration; show variable shifts and an explicit analytic reversal without claiming a matched real-model reversal. | P Introduction, 3.1, 4.1, Appendix A.4; R quartiles | Local checks passed |
| R3-20 | 47 / R23 | Say overshoots can dominate means; distinguish six reversed means from vision deletion. | P 3.3, Conclusion; R | Local checks passed |
| R3-21 | 48 / R2, N1 | Give the numerical thread its separate purpose and structure through the five actions below. | P 3.4-3.5; F 4-5 | Local checks passed |
| R3-22 | 49 / R2 | Split target reversal/numerical checks; remove placeholder; reorder floats/citations and keep them before Discussion. | P 3.4-3.5; D references; Q renders | Local checks passed |
| R3-23 | 50 / R2 | State fixed-ranking rescoring does not require accurate IG. | P 3.4-3.5 | Local checks passed |
| R3-24 | 51 / R2 | Separate offline function from saved m=64 bf16 rankings; cite endpoint-inclusive aliasing extension. | P 2.7, 3.5, Appendix B | Local checks passed |
| R3-25 | 52 / N1 | Show both state-Q examples: call-0 rank/AUC change and call-12 coordinate change with unchanged curves. | P 3.5; N; R | Local checks passed |
| R3-26 | 53 / N1 | Put dependent budget violations beside repeat passes; give finest-pair context-arm counts. | P 3.5; N; R | Local checks passed |
| R3-27 | 54 / R7 | Give checked-processing/coverage reasons per excluded set; identify fine-tuned 1B as next test. No unchecked robustness table. | P 2.5, 4.2; R success | Local checks passed; optional rescoring excluded by chosen scope |
| R3-28 | 55 / R4 | State sampler/steps, differentiated denoising, noise, encoders, grouping rule, selected calls, offline function and diagnostic definitions. | P 2.1-2.7; execution records; N | Local checks passed |
| R3-29 | 56 / R16 | Remove false target-to-response arrow. | F pipeline; Q inspection | Local checks passed |
| R3-30 | 62 / R9 | Name state failures/completed YCB episode; explain call 12 and finite-budget behavior; retain coarse failures and caption context/model count. | P 3.5, Figure 5; N | Local checks passed |
| R3-31 | 63 / R29 | Label changes by finer budget 2m and completeness by m; remove empty 128 change tick. | F numerical; P Eq. 9; Q | Local checks passed |
| R3-32 | 64 / R30 | Use full-mean contributions consistently; show later language-deletion positivity and first-call median -0.845. | P 3.3; R call-index summary | Local checks passed |
| R3-33 | 65 / R17, N4 | Give seven shared extreme episodes and B scale; scope early-call statements to vision. | P 3.3; R | Local checks passed |
| R3-34 | 66 / R10 | Specify Hama's linear-inner-score condition, deletion orientation and full-prefix reference; identify Appendix A.2. | P Discussion, Appendix A.2; bibliography | Local checks passed |
| R3-35 | 67 / R25 | Correct IDGI, Adebayo citation, quadrature attribution, metadata/placement and Zhang scope. | P Introduction, 2.3, 3.4, 4.2; bibliography | Local checks passed |
| R3-36 | 68 / R20 | Introduce both RDT sizes and cite the public small-model release. | P 2.5; bibliography | Local checks passed |
| R3-37 | 69 / R13 | Explain own-response storage, recovery of both responses and finite-precision self-reference checks. | P 2.4-2.5; R | Local checks passed |
| R3-38 | 70 / R14 | Quantify dropping nominal 1%; keep common weights; distinguish missing historical counts from separate masks. | P 2.5; R sensitivity | Local checks passed |
| R3-39 | 71 / R15 | Explain additive random-order calibration and zero-epsilon N; compare both cohorts consistently; end on implication. | P 3.1, Appendix A.3; R | Local checks passed |
| R3-40 | 72 / R19 | State insertion/deletion directions, align findings and name reporting steps. | P abstract, Introduction, Discussion, Conclusion | Local checks passed |
| R3-41 | 73 / R21 | Explain Q's early-path weighting and suppression of feature 1's late-path gradient. | P 3.4, Appendix A.2; F | Local checks passed |
| R3-42 | 74 / R26, R27, R28 | Clarify availability, action arithmetic, tokens/groups, terminology and controls; add review-link question to unsent inquiry. | P Methods, 4.1-4.2, availability; A inquiry question 6 | Local checks passed and inquiry prepared; journal answer deferred |
| R3-43 | 80 / R11 | Fix Figure 2 title/legend/overshoot mark, Figure 3 annotation, Figure 4B edge node, symlog captions and table alignment/decimals. | F; P captions; Q | Local checks passed |
| R3-44 | 81 / R12 | Distinguish B, completeness denominator, AUC schedule, IG nodes, constructions and bounds. | P; F labels | Local checks passed |
| R3-45 | 82 / R18, R14 | End Results paragraphs on findings; consolidate caveats without removing conditions; round main-text values. | P Results, 4.2 | Local checks passed |
| R3-46 | 83 / R22 | Fix abstract/Equation 8 spacing, leading zeros, Figure 1 paragraph boundary and Keywords colon. | P; D; Q renders | Local checks passed |
| R3-47 | 84 / R24 | Complete bridge identities/uniform-grid explanation; define J/bold 1; disambiguate residual and sharper bound. | P Appendices A-B; notation | Local checks passed |
| R3-48 | 85 / N4 | Add observed r/B marks from deterministic median-nearest Q vision-insertion call among all 750, with record-ID tie break; retain five figures. | F response mechanism; R; lineage; Q | Local checks passed |

## Document delivery and author actions

| Action | Checklist line / issue | Required disposition | Evidence | Status |
|---|---|---|---|---|
| R3-49 | 91 / Online version | Generate full repeated citations, existing author/email, native tables, figures and repository link. Missing city/country and declarations remain author tasks. | D; A; Q | Local export checks passed; final delivered bytes bound by Q; author facts deferred |
| R3-50 | 92 / Three-file package | Rebuild blinded PDF, Online DOCX and anonymous sources; keep equation TeX only; attach figures; verify extracted packages. | D; Q manifests/rebuilds/anonymity | Local preparation complete; final delivered archive acceptance bound by Q |
| R3-51 | 93 / File name | Name blinded PDF exactly as the manuscript title, plus extension. | Q delivery manifest | Local naming check passed |
| R3-52 | 94 / Editor email | Prepare unsent inquiry on both prior versions, eligibility, fees, copyright, venue and anonymous review link. Do not send. | A journal-inquiry-unsent.md | Local draft prepared; sending and answers author-deferred |
| R3-53 | 95 / Editor note | Prepare local prior-version disclosure for author review and later submission. | A prior-version-note-unsent.md | Local draft prepared; submission author-deferred |
| R3-54 | 96 / Final checks | Check page limits, numbers, Tier 1 claims, shared public/private bytes and preservation; keep human review separate. | R; D; Q; A | Local technical checks passed; final delivery and synchronization bound by Q; human review deferred |

## Local verification and remaining author work

The checked formats have 18 identified PDF pages, 18 blinded PDF pages and 20 Online Word pages. The abstract has 230 words; the manuscript has 17 references, five figures and two numbered tables, represented by three native table components in Word. The public CPU suite passed 783 tests with 11 native-document skips; the private checkout passed those 783 tests plus 95 supplemental tests. All 28 Online-export tests passed in the bundled document runtime. The 155 authenticated scientific input hashes and 99 raw-file hashes are preserved, and fresh CPU reproduction verifies all 36 frozen scientific artifacts. Source and scientific reviews found no open substantive findings.

These checks are recorded in [local verification](final_local_verification.md). Final delivery acceptance additionally requires matching source, bibliography and asset hashes, every rendered page inspected, anonymous text/metadata/links/source checks, extracted archive rebuilds, scientific reproduction and verified shared public/private bytes. `Manifest.json` and `Verification.json` identify the accepted delivery; changes invalidate the affected receipts. Earlier builds and Round 2 records cannot substitute.

The unsent author drafts are complete within the authorized local scope. Affiliation/eligibility facts, contribution/funding/competing-interest declarations, journal answers, independent human review and mentor sign-off remain author tasks. The existing AI statement is preserved for the author. Local technical completion does not attest to those declarations, journal approval or submission.
