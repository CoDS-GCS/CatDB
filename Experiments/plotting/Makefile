all: create clean

create:	
	 #pdflatex -shell-escape exp1_catalog.tex
	 #pdflatex -shell-escape exp1_micorbenchmark
	 #pdflatex -shell-escape exp0_statistics
	 #pdflatex -shell-escape exp_tmp
	 pdflatex -shell-escape exp_incremental
	 #pdflatex -shell-escape exp_statistics

clean:
	rm -rf *.aux  *.nav  *.vrb  *.log  *.out *.snm *.toc *.gz *.mtc* *.maf *.fls *.tmp *.bbl *.blg  *.mtc  *.bcf *.xml *.fdb_latexmk plots/*.dep plots/*.dpth plots/*.log plots/*.md5 *.auxlock exp1_fe.pdf

