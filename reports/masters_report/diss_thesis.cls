%%
%%  Department of Electrical, Electronic and Computer Engineering
%%  MEng Dissertation / PhD Thesis - Class File
%%  Copyright (C) 2011-2016 University of Pretoria.
%%
%% Revision: 23 Aug 2017

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{diss_thesis}

%% UP specific macros
\newcommand{\upeng}{University of Pretoria}
\newcommand{\upafr}{Universiteit van Pretoria}
\newcommand{\ebiteng}{Faculty of Engineering, Built Environment and Information Technology}
\newcommand{\ebitafr}{Fakulteit Ingenieurswese, Bou-omgewing en Inligtingtegnologie}
\newcommand{\eeceeng}{Mechanical and Aeronautical Engineering}
\newcommand{\eeceafr}{Elektriese, Elektroniese en Rekenaar-Ingenieurswese}
\newcommand{\doctype}{Report}
%% Define the placeholders for the various field definitions which
%%   are automatically set given the degree code option.
\newcommand{\dtthedegreecode}{XXXXXXX}
\newcommand{\dtthemodulecode}{XXX XXX}
\newcommand{\dtthedegreeabbreng}{XXXX}
\newcommand{\dtthedegreeabbrafr}{XXXX}
\newcommand{\dtthespectypeeng}{<missing degree code option>}
\newcommand{\dtthespectypeafr}{<missing degree code option>}
\newcommand{\dtthedegreedesignationeng}{<missing degree code option>}
\newcommand{\dtthedegreedesignationafr}{<missing degree code option>}
\newcommand{\dtthestudyleaderdesignationeng}{XXXXXXXXXX}
\newcommand{\dtthestudyleaderdesignationafr}{XXXXXXXXXX}
\newcommand{\dtthestudyleadersdesignationeng}{XXXXXXXXXX}
\newcommand{\dtthestudyleadersdesignationafr}{XXXXXXXXXX}
\newcommand{\dtthetitleeng}{TITLE}
\newcommand{\dtthetitleafr}{TITEL}
\newcommand{\dtthecandidatename}{Ryan Cameron Balshaw}
\newcommand{\dtthedate}{August 2016}
\newcommand{\dtthesupervisor}{Prof R. P. Feynman}
\newcommand{\StudentNo}{15020658}
%% Each of the official degree codes are defined here and are used
%%   to configure the default texts.
\DeclareOption{12250031}{
  \renewcommand{\dtthedegreecode}{12250031}
  \renewcommand{\dtthemodulecode}{EIR 890}
  \renewcommand{\dtthedegreeabbreng}{MEng}
  \renewcommand{\dtthedegreeabbrafr}{MIng}
  \renewcommand{\dtthespectypeeng}{Electrical Engineering}
  \renewcommand{\dtthespectypeafr}{Elektriese Ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Master of Engineering (Electrical Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Elektriese Ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12250061}{
  \renewcommand{\dtthedegreecode}{12250061}
  \renewcommand{\dtthemodulecode}{EAI 732}
  \renewcommand{\dtthedegreeabbreng}{BEng}
  \renewcommand{\dtthedegreeabbrafr}{BIng}
  \renewcommand{\dtthespectypeeng}{Mechanical Engineering}
  \renewcommand{\dtthespectypeafr}{Elektriese Ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Honours in Engineering (Mechanical Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Elektriese Ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}

\DeclareOption{12250091}{
  \renewcommand{\dtthedegreecode}{12250091}
  \renewcommand{\dtthemodulecode}{EIN 890}
  \renewcommand{\dtthedegreeabbreng}{MEng}
  \renewcommand{\dtthedegreeabbrafr}{MIng}
  \renewcommand{\dtthespectypeeng}{Electronic Engineering}
  \renewcommand{\dtthespectypeafr}{Elektroniese Ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Master of Engineering (Electronic Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Elektroniese Ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12250211}{
  \renewcommand{\dtthedegreecode}{12250211}
  \renewcommand{\dtthemodulecode}{ERI 890}
  \renewcommand{\dtthedegreeabbreng}{MEng}
  \renewcommand{\dtthedegreeabbrafr}{MIng}
  \renewcommand{\dtthespectypeeng}{Computer Engineering}
  \renewcommand{\dtthespectypeafr}{Rekenaaringenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Master of Engineering (Computer Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Rekenaaringenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12250201}{
  \renewcommand{\dtthedegreecode}{12250201}
  \renewcommand{\dtthemodulecode}{EIB 890}
  \renewcommand{\dtthedegreeabbreng}{MEng}
  \renewcommand{\dtthedegreeabbrafr}{MIng}
  \renewcommand{\dtthespectypeeng}{Bio-Engineering}
  \renewcommand{\dtthespectypeafr}{Bio-ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Master of Engineering (Bio-Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Bio-ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12250191}{
  \renewcommand{\dtthedegreecode}{12250191}
  \renewcommand{\dtthemodulecode}{EEY 890}
  \renewcommand{\dtthedegreeabbreng}{MEng}
  \renewcommand{\dtthedegreeabbrafr}{MIng}
  \renewcommand{\dtthespectypeeng}{Microelectronics}
  \renewcommand{\dtthespectypeafr}{Mikro\"{e}lektroniese Ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Master of Engineering (Microelectronics)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Mikro\"{e}lektroniese Ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12250202}{
  \renewcommand{\dtthedegreecode}{12250202}
  \renewcommand{\dtthemodulecode}{EPR 890}
  \renewcommand{\dtthedegreeabbreng}{MEng}
  \renewcommand{\dtthedegreeabbrafr}{MIng}
  \renewcommand{\dtthespectypeeng}{Software Engineering}
  \renewcommand{\dtthespectypeafr}{Programmatuuringenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Master of Engineering (Software Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Magister in Ingenieurswese (Programmatuuringenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12253046}{
  \renewcommand{\dtthedegreecode}{12253046}
  \renewcommand{\dtthemodulecode}{EER 891}
  \renewcommand{\dtthedegreeabbreng}{MSc}
  \renewcommand{\dtthedegreeabbrafr}{MSc}
  \renewcommand{\dtthespectypeeng}{Applied Sciences}
  \renewcommand{\dtthespectypeafr}{Toegepaste Wetenskap}
  \renewcommand{\dtthedegreedesignationeng}{Master of Science (Applied Sciences)}
  \renewcommand{\dtthedegreedesignationafr}{Magister Scientiae (Toegepaste Wetenskap)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Supervisor}
  \renewcommand{\dtthestudyleaderdesignationafr}{Studieleier}
  \renewcommand{\dtthestudyleadersdesignationeng}{Supervisor(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Studieleier(s)}
}
\DeclareOption{12263021}{
  \renewcommand{\dtthedegreecode}{12263021}
  \renewcommand{\dtthemodulecode}{EIR 990}
  \renewcommand{\dtthedegreeabbreng}{PhD}
  \renewcommand{\dtthedegreeabbrafr}{PhD}
  \renewcommand{\dtthespectypeeng}{Electrical Engineering}
  \renewcommand{\dtthespectypeafr}{Elektriese Ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Philosophiae Doctor (Electrical Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Philosophiae Doctor (Elektriese Ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Promoter}
  \renewcommand{\dtthestudyleaderdesignationafr}{Promotor}
  \renewcommand{\dtthestudyleadersdesignationeng}{Promoter(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Promotor(s)}
}
\DeclareOption{12263081}{
  \renewcommand{\dtthedegreecode}{12263081}
  \renewcommand{\dtthemodulecode}{EIN 990}
  \renewcommand{\dtthedegreeabbreng}{PhD}
  \renewcommand{\dtthedegreeabbrafr}{PhD}
  \renewcommand{\dtthespectypeeng}{Electronic Engineering}
  \renewcommand{\dtthespectypeafr}{Elektroniese Ingenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Philosophiae Doctor (Electronic Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Philosophiae Doctor (Elektroniese Ingenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Promoter}
  \renewcommand{\dtthestudyleaderdesignationafr}{Promotor}
  \renewcommand{\dtthestudyleadersdesignationeng}{Promoter(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Promotor(s)}
}
\DeclareOption{12263102}{
  \renewcommand{\dtthedegreecode}{12263102}
  \renewcommand{\dtthemodulecode}{ERI 990}
  \renewcommand{\dtthedegreeabbreng}{PhD}
  \renewcommand{\dtthedegreeabbrafr}{PhD}
  \renewcommand{\dtthespectypeeng}{Computer Engineering}
  \renewcommand{\dtthespectypeafr}{Rekenaaringenieurswese}
  \renewcommand{\dtthedegreedesignationeng}{Philosophiae Doctor (Computer Engineering)}
  \renewcommand{\dtthedegreedesignationafr}{Philosophiae Doctor (Rekenaaringenieurswese)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Promoter}
  \renewcommand{\dtthestudyleaderdesignationafr}{Promotor}
  \renewcommand{\dtthestudyleadersdesignationeng}{Promoter(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Promotor(s)}
}
\DeclareOption{12263151}{
  \renewcommand{\dtthedegreecode}{12263151}
  \renewcommand{\dtthemodulecode}{EIR 991}
  \renewcommand{\dtthedegreeabbreng}{PhD}
  \renewcommand{\dtthedegreeabbrafr}{PhD}
  \renewcommand{\dtthespectypeeng}{Electrical}
  \renewcommand{\dtthespectypeafr}{Elektries}
  \renewcommand{\dtthedegreedesignationeng}{Philosophiae Doctor (Electrical)}
  \renewcommand{\dtthedegreedesignationafr}{Philosophiae Doctor (Elektries)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Promoter}
  \renewcommand{\dtthestudyleaderdesignationafr}{Promotor}
  \renewcommand{\dtthestudyleadersdesignationeng}{Promoter(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Promotor(s)}
}
\DeclareOption{12263161}{
  \renewcommand{\dtthedegreecode}{12263161}
  \renewcommand{\dtthemodulecode}{EIN 991}
  \renewcommand{\dtthedegreeabbreng}{PhD}
  \renewcommand{\dtthedegreeabbrafr}{PhD}
  \renewcommand{\dtthespectypeeng}{Electronics}
  \renewcommand{\dtthespectypeafr}{Elektronies}
  \renewcommand{\dtthedegreedesignationeng}{Philosophiae Doctor (Electronics)}
  \renewcommand{\dtthedegreedesignationafr}{Philosophiae Doctor (Elektronies)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Promoter}
  \renewcommand{\dtthestudyleaderdesignationafr}{Promotor}
  \renewcommand{\dtthestudyleadersdesignationeng}{Promoter(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Promotor(s)}
}
\DeclareOption{12263202}{
  \renewcommand{\dtthedegreecode}{12263202}
  \renewcommand{\dtthemodulecode}{EIC 990}
  \renewcommand{\dtthedegreeabbreng}{PhD}
  \renewcommand{\dtthedegreeabbrafr}{PhD}
  \renewcommand{\dtthespectypeeng}{Biosystems}
  \renewcommand{\dtthespectypeafr}{Biosisteme}
  \renewcommand{\dtthedegreedesignationeng}{Philosophiae Doctor (Biosystems)}
  \renewcommand{\dtthedegreedesignationafr}{Philosophiae Doctor (Biosisteme)}
  \renewcommand{\dtthestudyleaderdesignationeng}{Promoter}
  \renewcommand{\dtthestudyleaderdesignationafr}{Promotor}
  \renewcommand{\dtthestudyleadersdesignationeng}{Promoter(s)}
  \renewcommand{\dtthestudyleadersdesignationafr}{Promotor(s)}
}

%% This class is an extension of the standard book class and will pass
%%   all unknown options to the foundation class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper,oneside]{book}

%% Establish input relative to a directory.
\RequirePackage{import}

%% Use the "geometry" package to maximize page usage whilst retaining enough
%%   margin space for staples and editorial notes.
\RequirePackage[a4paper,includeheadfoot,
                top=20mm,inner=30mm,outer=25mm,bottom=20mm
               ]{geometry}

%% Load support for colour.
\RequirePackage[table]{xcolor}

% COLORS (Tango) Mostly by Philip Bunge
% http://pbunge.crimson.ch/
\definecolor{White}{gray}{0.9}
\definecolor{Black}{gray}{0.0}
\definecolor{LightButter}{rgb}{0.98,0.91,0.31}
\definecolor{LightOrange}{rgb}{0.98,0.68,0.24}
\definecolor{LightChocolate}{rgb}{0.91,0.72,0.43}
\definecolor{LightChameleon}{rgb}{0.54,0.88,0.20}
\definecolor{LightSkyBlue}{rgb}{0.45,0.62,0.81}
\definecolor{LightPlum}{rgb}{0.68,0.50,0.66}
\definecolor{LightScarletRed}{rgb}{0.93,0.16,0.16}
\definecolor{Butter}{rgb}{0.93,0.86,0.25}
\definecolor{Orange}{rgb}{0.96,0.47,0.00}
\definecolor{Chocolate}{rgb}{0.75,0.49,0.07}
\definecolor{Chameleon}{rgb}{0.45,0.82,0.09}
\definecolor{SkyBlue}{rgb}{0.20,0.39,0.64}
\definecolor{Plum}{rgb}{0.46,0.31,0.48}
\definecolor{ScarletRed}{rgb}{0.80,0.00,0.00}
\definecolor{DarkButter}{rgb}{0.77,0.62,0.00}
\definecolor{DarkOrange}{rgb}{0.80,0.36,0.00}
\definecolor{DarkChocolate}{rgb}{0.56,0.35,0.01}
\definecolor{DarkChameleon}{rgb}{0.30,0.60,0.02}
\definecolor{DarkSkyBlue}{rgb}{0.12,0.29,0.53}
\definecolor{DarkPlum}{rgb}{0.36,0.21,0.40}
\definecolor{DarkScarletRed}{rgb}{0.64,0.00,0.00}
\definecolor{Aluminium1}{rgb}{0.93,0.93,0.92}
\definecolor{Aluminium2}{rgb}{0.82,0.84,0.81}
\definecolor{Aluminium3}{rgb}{0.73,0.74,0.71}
\definecolor{Aluminium4}{rgb}{0.53,0.54,0.52}
\definecolor{Aluminium5}{rgb}{0.33,0.34,0.32}
\definecolor{Aluminium6}{rgb}{0.18,0.20,0.21}


%% Enable more advanced character protrusion and font expansion, adjustment
%%   of inter-word spacing and additional kerning, etc.
\RequirePackage[activate={true,nocompatibility},tracking=true,
    kerning=true,spacing=true]{microtype}
\microtypecontext{spacing=nonfrench}

%% Load modern Latin font encoding template T1, as well as additional
%%   symbols.
\RequirePackage{lmodern}
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}
\RequirePackage{fix-cm}

%% Use the standard PostScript Type 1 version fonts for Roman, Sans Serif
%%   and Typewriter font classes. This allows the PDF files to be smaller
%%   as the custom TeX fonts do not have to be embedded.
\RequirePackage{mathptmx}
\RequirePackage[scaled=0.92]{helvet}
\RequirePackage{courier}

%% Enable fancy PostScript based characters.
\RequirePackage{pifont}

%% Load AMS mathematics support, fonts and symbols.
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amstext}
\RequirePackage{amssymb}
\RequirePackage{mathtools}

%% Set maths equation number to reset per chapter
\numberwithin{equation}{chapter}

%% Yet more mathematics font related modules...
\RequirePackage{bm}
\RequirePackage[mathcal]{eucal}
\RequirePackage{upgreek}

%% Load additional symbols.
\RequirePackage{wasysym}
\RequirePackage{latexsym}
%%\RequirePackage{MnSymbol}

%% Define input language as British English.
\RequirePackage[british]{babel}

%% Add calculator for dimensions.
\RequirePackage{calc}

%% Load support for advanced control structures in macros.
\RequirePackage{ifthen}

%% Load support for direct PDF generation.
\RequirePackage{ifpdf}

%% Special spaces in macros.
\RequirePackage{xspace}

%% Enable multiple column support
\RequirePackage{multicol}

%% Make nice looking Table and Figure captions.
\RequirePackage[labelsep=period]{caption}
\renewcommand{\captionlabelfont}{\bfseries}
\RequirePackage[labelformat=simple]{subcaption}
\renewcommand\thesubfigure{(\alph{subfigure})}

%% Change the "Contents" header to "Table of Contents". The addto approach
%%   much be used because of the use of Babel.
\addto\captionsbritish{%
  \renewcommand{\contentsname}%
    {Table of Contents}%
}

%% Remove the "Bibliography" so that the section header can be used.
\addto\captionsbritish{%
  \renewcommand{\bibname}%
    {References}%
}

%% Redefine the bibliography environment so that the title is displayed
%%   using the standard section command, but made upper-case as is required.
\renewenvironment{thebibliography}[1]
     {\chapter*{\refname}%
      \@mkboth{\refname}{\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

%% Modify section titles to comply with the formatting requirements of thesis
%%   and dissertations. The nobottomtitles command ensures that no titles
%%   appear at the end of a page. The bottomtitlespace value determines the
%%   restricted zone for bottom titles.
\RequirePackage[small,compact,nobottomtitles*,explicit,uppercase]{titlesec}
\renewcommand{\bottomtitlespace}{.12\textheight}

\titleformat{\chapter}
  {\normalfont\LARGE\bfseries\raggedright}%
  {\chaptertitlename\hspace*{0em} \thechapter\hspace*{0em}}%
  {0.2em}{#1}
\titleformat{\section}
  {\normalfont\LARGE\bfseries}%
  {\thesection\hspace*{1em}}{0em}{#1}
  
\titleformat{\subsection}
  {\normalfont\bfseries}{\thesubsection\hspace*{1em}}{0em}{#1}
  
\titleformat{\subsubsection}
  {\normalfont\bfseries}{\thesubsubsection\hspace*{1em}}{0em}{#1}

%% Set table of contents title.
\RequirePackage[titles]{tocloft}
\usepackage{etoolbox}
\setlength{\cftbeforetoctitleskip}{0 pt}
\setlength{\cftaftertoctitleskip}{4ex}
\renewcommand{\cfttoctitlefont}{\bfseries\Large}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftchappresnum}{\chaptername~}
\newlength{\mylen} % a "scratch" length
\settowidth{\mylen}{\bfseries\cftchappresnum\cftchapaftersnum} % extra space
\addtolength{\cftchapnumwidth}{\mylen} % add the extra space
\makeatletter
\newcommand*\updatechaptername{%
   \addtocontents{toc}{\protect\renewcommand*\protect\cftchappresnum{\@chapapp\ }}
}
\makeatother

\makeatletter
\patchcmd{\l@chapter}{#1}{#1}{}{}
\patchcmd{\l@section}{#1}{#1}{}{}
\makeatother

%% Include support for Appendices
\RequirePackage[title,header]{appendix}
\renewcommand{\appendixname}{Appendix}
\renewcommand{\appendixtocname}{Appendix}
\renewcommand{\appendixpagename}{Appendix}

%% Reduce spacing between bibliography items
\let\oldthebibliography=\thebibliography
\let\endoldthebibliography=\endthebibliography
\renewenvironment{thebibliography}[1]{%
	\begin{oldthebibliography}{#1}%
		\setlength{\itemsep}{0pt}%
	}%
	{%
	\end{oldthebibliography}%
}

%% Allow importing of EPS figures
\RequirePackage{epsfig}

%% Pull in support for EPS -> PDF conversion for the UP logo.
\ifpdf{}
  \usepackage{epstopdf}
\else
\fi

%% Add support for pseudo code for algorithms
\RequirePackage{algorithm}
\RequirePackage{algorithmic}

%% Enable fancy long tables.
\RequirePackage{booktabs}
%% Parametrized enumerations
\RequirePackage{enumerate}

%% Package to support advanced theorem
\RequirePackage{theorem}

%% Enable multiple column / row support
\RequirePackage{multicol}
\RequirePackage{multirow}

%% Include support for formatting of large numbers.
\RequirePackage{seqsplit}

%% Include support for SI units.
\RequirePackage[redefsymbols=false]{siunitx}

%% Support for date / time operations
\RequirePackage{datetime}
% Switch to ISO date displayed form 2000/03/08.
\yyyymmdddate{}

%% Load package to support standard graphics.
\RequirePackage{graphicx}

%% Load package to support advanced vector graphics.
\RequirePackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,matrix,mindmap,
                shadows,fit,calc,intersections}
\pgfdeclarelayer{background}
\pgfdeclarelayer{foreground}
\pgfsetlayers{background,main,foreground}

%% Required for inserting graphs.
\RequirePackage{pgfplots}
\pgfplotsset{compat=newest}

%% Required for PGF based tables.
\RequirePackage{pgfplotstable}

%% Disable paragraph indenting.
\RequirePackage[parfill]{parskip}
\addtolength{\parskip}{\baselineskip}

%% Enable support for extended list environments.
\RequirePackage{paralist}

%% Space below captions -- add some leeway to avoid orphan / stray lines
\setlength{\belowcaptionskip}{10pt plus 3pt minus 3pt}
\setlength{\abovecaptionskip}{\the\abovecaptionskip plus 3pt minus 3pt}
\addtolength{\floatsep}{0pt plus 3pt minus 3pt}
\addtolength{\textfloatsep}{0pt plus 3pt minus 3pt}
\addtolength{\intextsep}{0pt plus 3pt minus 3pt}
\widowpenalty=9000
\clubpenalty=9000

% Not going to use margins
\setlength{\marginparwidth}{0 pt}
\setlength{\marginparsep}{0 pt}
\setlength{\evensidemargin}{0 pt}
\setlength{\oddsidemargin}{0 pt}

%% Reset maths style parameters.
\setlength{\abovedisplayskip}{4.5pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{4.5pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{0pt plus 1pt}
\setlength{\belowdisplayshortskip}{0pt minus 1pt}

%% Define shorthand font family selection macros
%%                   Computer Concrete font family: \ccrfamily
\newcommand{\ccrfamily} {\fontencoding{T1}\fontfamily{ccr}\selectfont}
%%                   Computer Modern Roman font family: \cmrfamily
\newcommand{\cmrfamily} {\fontencoding{T1}\fontfamily{cmr}\selectfont}
%%                   Computer Modern Sans font family: \cmssfamily
\newcommand{\cmssfamily} {\fontencoding{T1}\fontfamily{cmss}\selectfont}
%%                   Computer Fibonacci font family: \cmfibfamily
\newcommand{\cmfibfamily} {\fontencoding{T1}\fontfamily{cmfib}\selectfont}
%%                   Computer Dunhill font family: \cmdhfamily
\newcommand{\cmdhfamily} {\fontencoding{T1}\fontfamily{cmdh}\selectfont}
%%                   Computer Modern Bright font family: \cmbrfamily
\newcommand{\cmbrfamily} {\fontencoding{T1}\fontfamily{cmbr}\selectfont}

%%                   Latin Modern font family: \lmrfamily
\newcommand{\lmrfamily} {\fontencoding{T1}\fontfamily{lmr}\selectfont}
%%                   Latin Modern Sans font family: \lmssfamily
\newcommand{\lmssfamily} {\fontencoding{T1}\fontfamily{lmss}\selectfont}

%%                   Pandora Roman font family: \panrfamily
\newcommand{\panrfamily} {\fontencoding{T1}\fontfamily{panr}\selectfont}
%%                   Pandora Sans font family: \pssfamily
\newcommand{\pssfamily} {\fontencoding{T1}\fontfamily{pss}\selectfont}

%%                   Postscript font: Times family: \ptmfamily
\newcommand{\ptmfamily} {\fontencoding{T1}\fontfamily{ptm}\selectfont}
%%                   Postscript font: Palatino family: \pplfamily
\newcommand{\pplfamily} {\fontencoding{T1}\fontfamily{ppl}\selectfont}
%%                   Postscript font: New Century family: \pncfamily
\newcommand{\pncfamily} {\fontencoding{T1}\fontfamily{pnc}\selectfont}
%%                   Postscript font: Bookman family: \pbkfamily
\newcommand{\pbkfamily} {\fontencoding{T1}\fontfamily{pbk}\selectfont}
%%                   Postscript font: Helvetica family: \phvfamily
\newcommand{\phvfamily} {\fontencoding{T1}\fontfamily{phv}\selectfont}
%%                   Postscript font: Avant Garde family: \pagfamily
\newcommand{\pagfamily} {\fontencoding{T1}\fontfamily{pag}\selectfont}
%%                   Postscript font: Courier family: \pcrfamily
\newcommand{\pcrfamily} {\fontencoding{T1}\fontfamily{pcr}\selectfont}
%%                   Postscript font: Utopia family: \putfamily
\newcommand{\putfamily} {\fontencoding{T1}\fontfamily{put}\selectfont}
%%                   Postscript font: Charter family: \bchfamily
\newcommand{\bchfamily} {\fontencoding{T1}\fontfamily{bch}\selectfont}

%% Define new font shapes:
%%                   Outline: \olshape
\newcommand{\olshape} {\fontshape{ol}\selectfont}
%%                   Upright italic: \uishape
\newcommand{\uishape} {\fontshape{ui}\selectfont}

%% Define slanted formatting for Latin phrases.
%%                   a posteriori:    \apost
\newcommand{\apost} {\emph{a posteriori}\xspace}
%%                   a prior:    \aprior
\newcommand{\apriori} {\emph{a priori}\xspace}
%%                   c.:    \circa
\newcommand{\circa} {\emph{c.}\xspace}
%%                   cf.:    \compare
\newcommand{\compare} {\emph{cf.}\xspace}
%%                   crica:    \circal
\newcommand{\circal} {\emph{circa}\xspace}
%%                   e.g.:      \eg
\newcommand{\eg} {\emph{e.g.},\xspace}
%%                   etc.:      \etc
\newcommand{\etc} {\emph{etc.}\@\xspace} % \etc
%%                   et al.:    \etal
\newcommand{\etal} {\emph{et al.}\xspace} % \etal
%%                   et seq.:    \etseq
\newcommand{\etseq} {\emph{et seq.}\xspace}
%%                   i.e.:      \ie
\newcommand{\ie} {\emph{i.e.}\xspace}

%% Miscellaneous
\DeclareRobustCommand{\BibTeX}{\textsc{Bib}\TeX\ }

%% Mathematical Units/Symbols
\newcommand{\leftnorm}{\left\|}
\newcommand{\rightnorm}{\right\|}
\newcommand{\Celsius}{{\ensuremath{\;^{\circ}C}}}    %% Degrees Celsius
\newcommand{\degree}{^{\circ}}
\newcommand{\tr}{{\ensuremath{\operatorname{tr}}}}       %% matrix trace operator
\newcommand{\var}{{\ensuremath{\operatorname{var}}}}     %% variance operator
\newcommand{\sd}{{\ensuremath{\operatorname{sd}}}}       %% standard deviation operator
\newcommand{\mode}{\ensuremath{\operatorname{mode}}}
\newcommand{\m}[1]{\ensuremath{\mathbf{#1}}}
\newcommand{\mt}[1]{\ensuremath{\mathbf{#1}^T}}
\newcommand{\median}[1]{\ensuremath{\underset{#1}{\operatorname{median}}}}
\newcommand{\argmin}[1]{\ensuremath{\underset{#1}{\operatorname{argmin}}}}
\newcommand{\argmax}[1]{\ensuremath{\underset{#1}{\operatorname{argmax}}}}
\newcommand{\expect}[1]{\ensuremath{E[#1]}}
\newcommand{\uv}[1]{\ensuremath{\underline{#1}}}
\newcommand{\iv}{\uv{i}}
\newcommand{\jv}{\uv{j}}
\newcommand{\kv}{\uv{k}}

%% TradeMark/Copyright etc. symbols
\newcommand{\Copyright}{\Pisymbol{psy}{227}\xspace} %% Registered Name
\newcommand{\RegMark}{\Pisymbol{psy}{226}\xspace} %% Registered Name
\newcommand{\TradeMark}{\hspace{1pt}\Pisymbol{psy}{228}\xspace}   %% Registered TradeMark

%% Change floats to be more realistic
\clubpenalty=9999
\widowpenalty=9999
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.2}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}

%% Verbatim size
\newlength{\verbatimizeup}
\setlength{\verbatimizeup}{-\parskip}%
\addtolength{\verbatimizeup}{-\parskip}%
\addtolength{\verbatimizeup}{-10pt}

%% Lengths for itemized environments
\RequirePackage{enumitem}
\setlist{itemsep=-18pt}

%% Enable fancy headers, verbatim sections and boxes.
\RequirePackage{fancyhdr}
\RequirePackage{fancyvrb}
\RequirePackage{fancybox}

% Make the header wide enough to fit the header bar
\addtolength{\headheight}{3pt}
\fancyheadoffset[L]{5mm}
\fancyfootoffset[L]{5mm}

%% Required header / footer style
\fancypagestyle{empty}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

%% Required header / footer style for plain pages... which implies
%% no header.
\fancypagestyle{plain}{
  \fancyhf{}                              % Flush all current headers
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
  \fancyfoot[R]{\thepage}
}

%%the first page of a chapter is a normal style
%% Required header / footer style for normal pages
\fancypagestyle{normal}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{1pt}
  \fancyhead[L]{\chaptertitlename\ \thechapter}
  \fancyhead[R]{\leftmark}
  \renewcommand{\footrulewidth}{1pt}
  \fancyfoot[L]{Department of \eeceeng\newline\upeng}
  \fancyfoot[R]{\thepage}
}

%% Required header / footer style for bib
\fancypagestyle{bibliography}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{1pt}
  \fancyhead[L]{References}
  \fancyhead[R]{}
  \renewcommand{\footrulewidth}{1pt}
  \fancyfoot[L]{Department of \eeceeng\newline\upeng}
  \fancyfoot[R]{\thepage}
}

%% Define automatic macro that gets applied at all font changes to reset
%%   the spacing before and after equations.
\RequirePackage{etoolbox}
\newcommand{\zerodisplayskips}{%
  \setlength{\abovedisplayskip}{4.5pt plus 1pt minus 1pt}
  \setlength{\belowdisplayskip}{4.5pt plus 1pt minus 1pt}
  \setlength{\abovedisplayshortskip}{0pt plus 1pt}
  \setlength{\belowdisplayshortskip}{0pt minus 1pt}
}
\appto{\normalsize}{\zerodisplayskips}
\appto{\small}{\zerodisplayskips}
\appto{\footnotesize}{\zerodisplayskips}

%% Include space between rows in equarray
\setlength{\jot}{5 pt}

%% Allow extra whitespace in lines, decrease number of hyphenations
\sloppy

%% Standard title / cover page for research proposal.
\newcommand{\maketitlepage}{
\thispagestyle{empty}
{
  \renewcommand{\baselinestretch}{1.2}
  \setlength{\parindent}{0mm}
  \setlength{\parskip}{0mm}
  \centering
  \includegraphics[scale=1]{UniversityLogo.png}
  \vfill
  \line(1,0){400}\\
  \vspace{0.5cm}
  {\bfseries\Large\MakeUppercase{\dtthetitleeng}}
  \line(1,0){400}\\
  \vspace*{\stretch{1}}

  by

  \vspace*{\stretch{1}}

  {\bfseries\dtthecandidatename}
  
  {\bfseries\StudentNo}

  \vspace*{\stretch{2}}

  Submitted in partial fulfillment of the requirements for the degree

  \dtthedegreedesignationeng

  \vspace*{\stretch{1}}

  in the

  \vspace*{\stretch{1}}

  Department of \eeceeng \\
  \ebiteng

  \vspace*{\stretch{1}}

  \MakeUppercase{\upeng}

  \vspace*{\stretch{1}}

  \dtthedate

  \vspace*{\stretch{6}}
}

\newpage
}

%% Ensure that all headers are numbered as per requirements.
\setcounter{secnumdepth}{4}

%% However only the first to levels must appear in the TOC.
\setcounter{tocdepth}{2}

%% Set to single line spacing as required.
\renewcommand{\baselinestretch}{1.5}

%% Set the initial page number and style for document.
\pagenumbering{arabic}
\pagestyle{plain}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}

\endinput

%% End of File.
