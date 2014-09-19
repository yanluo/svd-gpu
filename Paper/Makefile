OBJ=BTSVD
SRC=0_abstract.tex 1_intro.tex 2_relate.tex 3_algorithm.tex 4_implement.tex 5_experiment.tex 6_conclusion.tex algorithm_bisection.tex bibliography.tex

all: $(OBJ).tex $(SRC)
	pdflatex $(OBJ)
	pdflatex $(OBJ)

normal: $(OBJ).tex $(SRC)
#	bibtex $(OBJ)
	latex $(OBJ)
	latex $(OBJ)
	dvips -f -tletter -Ppdf -G0 $(OBJ) > $(OBJ).ps
	ps2pdf $(OBJ).ps $(OBJ).pdf
	
clean:
	rm -f *.ps *.dvi *.aux *.log *.pdf *.bbl *.blg *~ *.gz
