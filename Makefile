# This is a developer helper makefile. When developing install once with 'pip install -e .'
# Upon making changes, simply re-run make and the .so files will be updated inplace.
.PHONY: all
all:
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -r build
	find . -name '*.c' -exec rm {} \;
	find . -name '*.so' -exec rm {} \;

.PHONY: watch
watch:
	bash watch.sh
