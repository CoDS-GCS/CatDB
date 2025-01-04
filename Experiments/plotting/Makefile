all: create clean

create:	
	 #pdflatex -shell-escap exp0 # Statistics and Profiling
	 #pdflatex -shell-escap exp1 # Cleaning
	 pdflatex -shell-escap exp2 # 10 Iteration and Verify
	 #pdflatex -shell-escap exp3 # 1 Iteration
	 #pdflatex -shell-escap exp4 # Error
	 #pdflatex -shell-escap exp5 # End to End Experiment

	 #pdflatex -shell-escap exp-demo # Demo Experiment

clean:
	rm -rf *.aux  *.nav  *.vrb  *.log  *.out *.snm *.toc *.gz *.mtc* *.maf *.fls *.tmp *.bbl *.blg  *.mtc  *.bcf *.xml *.fdb_latexmk plots/*.dep plots/*.dpth plots/*.log plots/*.md5 *.auxlock exp1_fe.pdf

