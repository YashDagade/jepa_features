TECTONIC ?= tectonic
SOURCE_DATE_EPOCH ?= 0
export SOURCE_DATE_EPOCH

.PHONY: build watch clean

build:
	$(TECTONIC) -X build --keep-logs
	cp build/main/main.pdf main.pdf

watch:
	@echo "Watching TeX sources; updating main.pdf after each successful build."
	python3 -u watch_pdf.py

clean:
	@if [ -d build ]; then find build -type f -delete; find build -depth -type d -empty -delete; fi
