TECTONIC ?= tectonic
SOURCE_DATE_EPOCH ?= 0
export SOURCE_DATE_EPOCH

.PHONY: build watch clean

build:
	$(TECTONIC) -X build --keep-logs
	cp build/main/main.pdf main.pdf

watch:
	$(TECTONIC) -X watch

clean:
	@if [ -d build ]; then find build -type f -delete; find build -depth -type d -empty -delete; fi
