\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hwz}[2018/11/15 HWZ Latex class]

%#############################
% Color all the things!
%#############################
% This needs to stay up here to prevent option clash warning
\usepackage[table]{xcolor}
 
%#############################
% Base class
%#############################
% font-size as in "Richtlinien Semesterarbeit"
\LoadClass[12pt, twoside]{extarticle}

%#############################
% Plumbing...
%#############################
\usepackage[utf8]{inputenc}
\usepackage[german]{babel}

%#############################
% TODO Notes
%#############################
\definecolor{todocolor}{HTML}{CCCCCC}
\usepackage[color=todocolor]{todonotes}

%#############################
% Merge list of figures and list of tables
%#############################
\def\table{\def\figurename{Tabelle}\figure}
\let\endtable\endfigure 

%#############################
% Page layout
%#############################
% - Set margin analogous to word template
\usepackage[left=2.5cm, right=2.5cm, top=2.5cm, bottom=2cm]{geometry}

% - line-height as in "Richtilinien Semesterarbeit"
\renewcommand{\baselinestretch}{1.15}

% - custom paragraph spacing
\setlength{\parskip}{0.25em}

%#############################
% Page header & footer
%#############################
\usepackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\sectionmark}[1]{\markboth{#1}{#1}}
\renewcommand{\subsectionmark}[1]{\markright{#1}}
\setlength{\headheight}{15pt}
\fancyhf{}
\fancyfoot[LE,RO]{\thepage}

%#############################
% citation style APA
%#############################
% apacite uses bibtex and conforms mostly to APA 6th edition
% see https://tex.stackexchange.com/a/263800 on why we use biblatex-apa instead
%usepackage{apacite}
%bibliographystyle{apacite}
\usepackage[backend=biber, style=apa]{biblatex}
\bibliography{./references.bib}
\usepackage{csquotes}
\DeclareLanguageMapping{german}{german-apa}

%#############################
% List
%#############################
% allows for custom list spacings
\usepackage{enumitem}

%#############################
% Tables
%#############################
% make the colored, because who doesn't love colors? :)
\usepackage{colortbl}

%#############################
% Captions
%#############################
% Allows the use of \caption* to prevent the caption from being numbered
\usepackage{caption}

%#############################
% Keep paragraphs together
%#############################
\widowpenalties 1 10000
\raggedbottom

%#############################
% Commands
%#############################
\input{commands/abstract.tex}
\input{commands/begin-main.tex}
\input{commands/bibliography.tex}
\input{commands/declaration-of-authorship.tex}
\input{commands/list-of-tables-and-figures.tex}
\input{commands/preface.tex}
\input{commands/section-unnumbered.tex}
\input{commands/titlepage.tex}