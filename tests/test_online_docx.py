"""Meaningful structural checks for the journal's local Word export."""
import pytest

from scripts.export_online_docx import bibliography, expand_inputs, format_author_block, format_docx, inline_text, normalize_table_stacks, number_headings, select_identified, strip_comments, strip_pdf_layout_definitions, transform_ast


def test_pdf_column_modifiers_cannot_consume_leading_cell_numerals():
    source = r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}p{.36\linewidth}X}Numerical sample & 12 completed contexts \\\end{tabularx}"
    prepared = normalize_table_stacks(source)
    assert r"\raggedright" not in prepared and r"\arraybackslash" not in prepared
    assert 'Numerical sample & 12 completed contexts' in prepared


def test_multiline_headers_cannot_become_spurious_table_rows():
    source = r"Cohort & \shortstack{Positive\\share} & \shortstack{Overshoot\\contribution} \\ Q-ranked & 88.1\% & $-17.21$ \\"
    result = normalize_table_stacks(source)
    assert result.count(r"\\") == 2
    assert result.split(r"\\")[0].split("&") == ["Cohort ", " Positive share ", " Overshoot contribution "]
    assert r"88.1\% & $-17.21$" in result
    with pytest.raises(ValueError, match="shortstack"):
        normalize_table_stacks(r"\shortstack{\textbf{Do not lose this}}")


def test_word_section_numbers_keep_starred_sections_and_appendices_distinct():
    def header(level, title, label="", starred=False):
        return {"t":"Header","c":[level,[label,["unnumbered"] if starred else [],[]],[{"t":"Str","c":title}]]}
    blocks=[header(1,"Introduction"),header(1,"Methods"),header(2,"Design"),
            header(1,"References",starred=True),header(1,"Proofs","proofs"),header(2,"Identity"),
            header(1,"Aliasing","alias"),header(2,"Construction")]
    number_headings(blocks,{"proofs":"A","alias":"B"})
    assert [inline_text(b['c'][2]) for b in blocks] == [
        '1. Introduction','2. Methods','2.1 Design','References','Appendix A. Proofs',
        'A.1 Identity','Appendix B. Aliasing','B.1 Construction']


def test_review_comments_do_not_create_caption_paragraphs():
    source = "\\caption{\n% BEGIN approved\nScientific caption with 95\\%.\n% END approved\n}"
    assert strip_comments(source) == "\\caption{\nScientific caption with 95\\%.\n}"


def test_pdf_layout_definitions_preserve_structural_headings_and_fail_closed():
    source = "\\makeatletter\n\\renewcommand\\section{layout}\n\\renewcommand\\subsection{layout}\n\\renewcommand\\@maketitle{layout}\n\\makeatother\n\\section{Results}\nEvidence."
    assert strip_pdf_layout_definitions(source).strip() == "\\section{Results}\nEvidence."
    with pytest.raises(ValueError, match="Unreviewed"):
        strip_pdf_layout_definitions(r"\makeatletter\newcommand\important{Evidence}\makeatother")


def metrics():
    return dict(citation_uses=[], inline_equations=0, display_equations=0, tables=0, images=[])


def test_identified_nested_conditionals_do_not_leak_blinded_branch():
    source = r"\newif\ifanonymous\ifdefined\TMLRAnonymous\anonymoustrue\else\anonymousfalse\fi A\ifanonymous SECRET\else AUTHOR\fi"
    assert select_identified(source).strip() == "A AUTHOR"
    with pytest.raises(ValueError, match="Unclosed"):
        select_identified(r"\ifanonymous unfinished")


def test_inputs_reject_escape_and_cycles(tmp_path):
    (tmp_path / "one.tex").write_text(r"\input{one}")
    with pytest.raises(ValueError, match="cyclic"):
        expand_inputs(r"\input{one}", tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        expand_inputs(r"\input{../outside}", tmp_path)


def test_nested_table_structure_is_preserved():
    value = {"t": "Table", "c": [["", [], []], [None, []], [], [[], []], [[[], [], [], []]], [[], []]]}
    counter = metrics()
    assert transform_ast(value, {}, counter) == value
    assert counter["tables"] == 1


def test_online_citations_repeat_entries_and_superscript_separator():
    citation = {"t": "Cite", "c": [[{"citationId": "one"}, {"citationId": "two"}], []]}
    refs = {"one": [{"t": "Str", "c": "Author One."}], "two": [{"t": "Str", "c": "Author Two."}]}
    counter = metrics()
    result = transform_ast([citation, citation], refs, counter)
    assert counter["citation_uses"] == ["one", "two", "one", "two"]
    assert sum(x["t"] == "Superscript" for x in result) == 2
    assert sum(x.get("c") == "((" for x in result) == 4
    with pytest.raises(ValueError, match="absent"):
        transform_ast(citation, {}, metrics())


def test_equations_keep_literal_tex_and_unknown_commands_fail_closed():
    equation = {"t": "Math", "c": [{"t": "DisplayMath"}, r"\sum_i A_i"]}
    result = transform_ast(equation, {}, metrics())
    assert result[0]["c"].startswith("$$") and result[-1]["c"].endswith("$$")
    environment = {"t": "Math", "c": [{"t": "DisplayMath"}, r"\begin{equation}x=1\end{equation}"]}
    rendered = transform_ast(environment, {}, metrics())
    assert rendered == [{"t": "Str", "c": r"\begin{equation}x=1\end{equation}"}]
    with pytest.raises(ValueError, match="meaningful"):
        transform_ast({"t": "RawBlock", "c": ["latex", r"\unreviewed{important text}"]}, {}, metrics())


def test_bibliography_keeps_full_entries_in_compiled_order():
    bbl = r"\begin{thebibliography}{2}\bibitem[A(2026)]{a} A. Title.\bibitem{b} B, C and D. Work.\end{thebibliography}"
    assert bibliography(bbl) == {"a": "A. Title.", "b": "B, C and D. Work."}


def test_plain_citation_receipt_preserves_protected_title_groups():
    # Pandoc represents brace-protected bibliography acronyms as Spans.
    nodes = [{"t": "Span", "c": [["", [], []], [{"t": "Str", "c": "RDT-1B"}]]},
             {"t": "Str", "c": ":"}, {"t": "Space"}, {"t": "Str", "c": "Title."}]
    assert inline_text(nodes) == "RDT-1B: Title."
    with pytest.raises(ValueError, match="Unreviewed"):
        inline_text([{"t": "Unknown", "c": "must not disappear"}])


def test_author_affiliation_and_correspondence_are_native_paragraphs():
    docx = pytest.importorskip("docx", reason="Native Word structure is tested in the bundled document runtime")
    doc = docx.Document()
    paragraph = doc.add_paragraph("old author block")
    format_author_block(paragraph)
    assert [p.text for p in doc.paragraphs] == ["Arjun Bajpai1*", "1Windermere Preparatory School, Windermere, Florida, United States", "*Corresponding author: arjunbajpai2009@gmail.com"]
    assert not paragraph._p.xpath(".//w:br")
    assert doc.paragraphs[0].runs[-2].font.superscript
    assert not doc.paragraphs[0].runs[-1].font.superscript
    assert doc.paragraphs[1].runs[0].font.superscript
    assert all(p.paragraph_format.first_line_indent.pt == 0 for p in doc.paragraphs)


def test_online_layout_preserves_type_geometry_and_unbroken_equations(tmp_path):
    docx = pytest.importorskip("docx", reason="Native Word structure is tested in the bundled document runtime")
    import base64
    from io import BytesIO
    doc = docx.Document()
    unused_png = BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aT5sAAAAASUVORK5CYII="))
    doc.part.get_or_add_image(unused_png)
    assert any(relation.reltype.endswith("/image") for relation in doc.part.rels.values())
    doc.add_paragraph("Title", style="Title")
    doc.add_paragraph("Results", style="Heading 1")
    doc.add_paragraph("Response identity", style="Heading 2")
    doc.add_paragraph("Equations give")
    doc.add_paragraph(r"\begin{equation}C_Q-C_N=0\end{equation}")
    doc.add_paragraph("References", style="Heading 1")
    for index in range(15):
        doc.add_paragraph(f"Author {index}. Reference.")
    path = tmp_path / "layout.docx"
    doc.save(path)
    format_docx(path, "Title", expected_reference_count=15)
    result = docx.Document(path)
    assert not any(relation.reltype.endswith("/image") for relation in result.part.rels.values())
    assert result.styles["Normal"].font.size.pt == 12
    assert result.styles["Heading 1"].font.size.pt == 14
    assert result.styles["Heading 2"].font.size.pt == 12
    assert result.paragraphs[3].paragraph_format.keep_with_next
    assert result.paragraphs[4].paragraph_format.keep_together
    assert result.paragraphs[-1].text.startswith("15. ")
    section = result.sections[0]
    assert (section.page_width.inches, section.page_height.inches) == (8.5, 11)
    assert all(value.inches == 1 for value in (section.top_margin, section.bottom_margin, section.left_margin, section.right_margin))


def test_prose_indent_does_not_spread_to_captions_equations_or_tables(tmp_path):
    docx = pytest.importorskip("docx", reason="Native Word structure is tested in the bundled document runtime")
    doc = docx.Document()
    doc.add_paragraph("Title", style="Title")
    doc.add_paragraph("Abstract")
    doc.add_paragraph("Abstract text.")
    doc.add_paragraph("Results", style="Heading 1")
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Continuing prose.")
    doc.add_paragraph("Figure 1 | Scientific caption.")
    doc.add_paragraph(r"$$x=1$$")
    for count in (3, 6):
        table = doc.add_table(rows=2, cols=count)
        for row_index, row in enumerate(table.rows):
            for column_index, cell in enumerate(row.cells):
                cell.text = f"cell {row_index},{column_index}"
    doc.add_paragraph("References", style="Heading 1")
    for index in range(15):
        doc.add_paragraph(f"Author {index}. Reference.")
    path = tmp_path / "paragraph-roles.docx"
    doc.save(path)
    format_docx(path, "Title", expected_reference_count=15)
    result = docx.Document(path)
    assert result.paragraphs[5].paragraph_format.first_line_indent.pt == 18
    assert result.paragraphs[5].paragraph_format.space_after.pt == 0
    assert all(result.paragraphs[i].paragraph_format.first_line_indent.pt == 0 for i in (0, 1, 2, 3, 4, 6, 7, 9))
    assert [len(table.columns) for table in result.tables] == [3, 6]
    for table in result.tables:
        for row_index, row in enumerate(table.rows):
            for column_index, cell in enumerate(row.cells):
                assert cell.text == f"cell {row_index},{column_index}"
                assert all(p.paragraph_format.first_line_indent.pt == 0 for p in cell.paragraphs)


def test_unknown_table_width_fails_before_dropping_columns(tmp_path):
    docx = pytest.importorskip("docx", reason="Native Word structure is tested in the bundled document runtime")
    doc = docx.Document()
    doc.add_table(rows=1, cols=7)
    path = tmp_path / "unexpected-table.docx"
    doc.save(path)
    with pytest.raises(ValueError, match="column count"):
        format_docx(path, "Title", expected_reference_count=15)


@pytest.mark.parametrize("reference_count", [1, 15, 17])
def test_reference_numbering_matches_variable_compiled_bibliography(tmp_path, reference_count):
    docx = pytest.importorskip("docx", reason="Native Word structure is tested in the bundled document runtime")
    doc = docx.Document()
    doc.add_paragraph("References", style="Heading 1")
    for index in range(reference_count):
        doc.add_paragraph(f"Author {index}. Reference.")
    doc.add_paragraph("Appendix A. Proofs", style="Heading 1")
    doc.add_paragraph("This paragraph is not a reference.")
    path = tmp_path / "references.docx"
    doc.save(path)

    format_docx(path, "Title", expected_reference_count=reference_count)

    result = docx.Document(path)
    assert [p.text for p in result.paragraphs[1:1 + reference_count]] == [
        f"{index + 1}. Author {index}. Reference." for index in range(reference_count)
    ]
    assert result.paragraphs[-1].text == "This paragraph is not a reference."


@pytest.mark.parametrize("reference_count, include_heading", [(2, True), (4, True), (0, True), (3, False)])
def test_missing_or_extra_references_fail_without_saving(tmp_path, reference_count, include_heading):
    docx = pytest.importorskip("docx", reason="Native Word structure is tested in the bundled document runtime")
    doc = docx.Document()
    if include_heading:
        doc.add_paragraph("References", style="Heading 1")
    for index in range(reference_count):
        doc.add_paragraph(f"Author {index}. Reference.")
    path = tmp_path / "incomplete-references.docx"
    doc.save(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="expected 3, found"):
        format_docx(path, "Title", expected_reference_count=3)

    assert path.read_bytes() == before


@pytest.mark.parametrize("expected_count", [0, -1, None, True, 2.0])
def test_reference_count_requires_nonempty_compiled_bibliography(tmp_path, expected_count):
    with pytest.raises(ValueError, match="positive integer"):
        format_docx(tmp_path / "not-opened.docx", "Title", expected_reference_count=expected_count)
