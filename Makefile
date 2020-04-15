.PHONY: help build install install-dev test clean

.DEFAULT: help
help:
	@echo "make build"
	@echo "	   build extensions in place"
	@echo "make install"
	@echo "	   install in the current python environment"
	@echo "make install-dev"
	@echo "	   install in development mode with dev dependencies"
	@echo "make test"
	@echo "	   run tests"
	@echo "make clean"
	@echo "	   remove build artifacts and __pycache__"

build:
	python setup.py build_ext -i

install:
	pip install .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	python setup.py pytest

clean:
	rm -rf ./**/__pycache__
	rm -rf build
	rm -rf *.egg-info
	find . -wholename "./jakteristics/*.cpp" -type f -delete
	find . -wholename "./jakteristics/*.c" -type f -delete
	find . -wholename "./jakteristics/*.html" -type f -delete
	find . -wholename "./jakteristics/*.so" -type f -delete
	find . -wholename "./jakteristics/*.pyd" -type f -delete

tag:
	@if [ -z $(tag) ]; then\
		echo "Please provide the 'tag' variable.";\
	else\
		git tag -a $(tag) -m "$(tag)";\
		git push --delete upstream latest;\
		git tag -d latest;\
		git tag -a -m "latest" latest;\
	fi
