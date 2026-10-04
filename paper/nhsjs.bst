% Minimal numeric bibliography for this manuscript.
% Citation order is preserved. Every author is printed with initials first.
ENTRY
  { author title journal booktitle volume number pages year doi url publisher note }
  {} {}
INTEGERS { nameptr namesleft numnames }
STRINGS { s t }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
  ""
  { namesleft #0 > }
  { s nameptr "{f.~}{vv~}{ll}{, jj}" format.name$ 't :=
    nameptr #1 >
      { ", " * t * }
      { t * }
    if$
    nameptr #1 + 'nameptr :=
    namesleft #1 - 'namesleft :=
  }
  while$
}

FUNCTION {output.entry}
{ newline$
  "\bibitem{" write$ cite$ write$ "}" write$ newline$
  author empty$
    { }
    { author format.names add.period$ write$ " " write$ }
  if$
  title empty$
    { }
    { title "t" change.case$ add.period$ write$ " " write$ }
  if$
  journal empty$
    { booktitle empty$
        { publisher empty$ { "" } { publisher } if$ }
        { booktitle }
      if$
    }
    { journal }
  if$
  duplicate$ empty$
    { pop$ }
    { "\emph{" swap$ * "}. " * write$ }
  if$
  volume empty$
    { }
    { "Vol. \textbf{" write$ volume write$ "}" write$
      number empty$ { } { "(" write$ number write$ ")" write$ } if$
      ", " write$
    }
  if$
  pages empty$
    { }
    { "pg. " write$ pages write$ ", " write$ }
  if$
  year empty$ { } { year write$ } if$
  doi empty$
    { url empty$ { "." write$ } { ", \url{" write$ url write$ "}." write$ } if$ }
    { ", \url{https://doi.org/" write$ doi write$ "}." write$ }
  if$
  note empty$ { } { " " write$ note add.period$ write$ } if$
  newline$
}

FUNCTION {article} { output.entry }
FUNCTION {inproceedings} { output.entry }
FUNCTION {conference} { output.entry }
FUNCTION {book} { output.entry }
FUNCTION {incollection} { output.entry }
FUNCTION {misc} { output.entry }
FUNCTION {techreport} { output.entry }
FUNCTION {phdthesis} { output.entry }
FUNCTION {mastersthesis} { output.entry }
FUNCTION {unpublished} { output.entry }
FUNCTION {default.type} { output.entry }

READ
FUNCTION {begin.bib}
{ "\begin{thebibliography}{99}" write$ newline$ }
EXECUTE {begin.bib}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }
EXECUTE {end.bib}
