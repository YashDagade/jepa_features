TECTONIC ?= tectonic
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
SOURCE_DATE_EPOCH ?= 0
export SOURCE_DATE_EPOCH

.PHONY: build build-tectonic clean

build:
	mkdir -p build/main
	$(PDFLATEX) -interaction=nonstopmode -output-directory=build/main main.tex
	BIBINPUTS=.: BSTINPUTS=.: $(BIBTEX) build/main/main
	$(PDFLATEX) -interaction=nonstopmode -output-directory=build/main main.tex
	$(PDFLATEX) -interaction=nonstopmode -output-directory=build/main main.tex
	cp build/main/main.pdf main.pdf

build-tectonic:
	$(TECTONIC) -X build --keep-logs
	cp build/main/main.pdf main.pdf

clean:
	@if [ -d build ]; then find build -type f -delete; find build -depth -type d -empty -delete; fi
