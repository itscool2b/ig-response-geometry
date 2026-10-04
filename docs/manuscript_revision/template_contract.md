# Document template contract

The current local revision uses the official NHSJS Standard and Online Word templates downloaded from the journal's July 2026 template URLs. Local reference copies and rendered instruction pages are retained under `previous-runs/2026-10-02/templates` and `qa/online-template`. The template instructions and example manuscript are not content for the revised paper.

## Required structure

Use US Letter pages, one-inch margins, Times New Roman 12 point body text, and single spacing. Preserve an identifiable title, author block in the identified version, one-paragraph abstract, keywords, Introduction, Methods, Results, Discussion, acknowledgments when identified, and references. Keep the two compact mathematical appendices. Number and cite displays in order and place them near their first use. Both Standard and Online templates state no more than 20 pages. Both outputs are checked against that limit. The Online export expands repeated citations and retains equation source under the journal's LaTeX submission route.

The templates define Normal as Times New Roman 12 point, single spacing, six points after. The authorized revision uses 1.5em body paragraph indentation and no added paragraph spacing to keep paragraph boundaries clear within the page limit. They include NHSJS Title, Section, Subsection and Subsubsection styles. The title and section hierarchy are mapped explicitly in the local conversion. No body, caption or table text is reduced to force a page limit. Remove all sample text, instructional pages, unused sample headers, template rules and placeholder fields.

## Editable content and citations

The canonical editable source is `paper/paper.tex` with its bibliography, generated facts and tables, figures and proof appendix. Word is generated from this same source. Tables remain native Word tables. Figure images are embedded and supplied separately. Standard citations are numbered superscripts before punctuation. Online citations repeat the full compiled bibliography entry at every use inside separate double parentheses, with a superscript comma between multiple sources. All author names are retained in the bibliography.

The Word export contains literal TeX equation source, including dollar delimiters. This is an intentional representation requested by the NHSJS LaTeX instructions. It is not a failure to render Word equations. Display equations occupy separate paragraphs so the editor can identify and convert them.

## Verification

Check page size, margins, normal text size, native table count, figure count, reference completeness, citation-use parity, cross-reference resolution, abstract length, removal of template instructions, metadata and all rendered pages. Inspect the Standard PDFs and Online Word rendering separately. Store the exact reviewed output hashes in the final manifest. A rendered Word PDF is a local inspection artifact and does not replace the required Word file.

Pandoc 3.12 supplies structural conversion. The bundled Python supplies document libraries and its document renderer. On Windows that renderer explicitly requires native LibreOffice. A signed official LibreOffice 26.2.6.3 MSI was administratively extracted into the separate local tools archive. Its Authenticode signature was valid for The Document Foundation. No system installation or external document conversion service was used.

Official sources are [submission guidelines](https://nhsjs.com/submission-guidelines/), [Standard template](https://nhsjs.com/wp-content/uploads/2026/07/NHSJS-Manuscript-Template-Standard-Citations.docx) and [Online template](https://nhsjs.com/wp-content/uploads/2026/07/NHSJS-Manuscript-Template-Online-Citations.docx).

Historical paths beginning with `previous-runs/` refer to the separate local records archive outside this repository. They are not build dependencies.
