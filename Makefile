# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
# Dependencies : python3 venv internal module
# Recall: .PHONY  defines special targets not associated with files
#
# Some Makefile global variables can be set in make command line:
# PANDORA_VENV: Change directory of installed venv (default local "venv" dir)
#

############### GLOBAL VARIABLES ######################

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: PANDORA_VENV="other-venv/" make install
ifndef PANDORA_VENV
	PANDORA_VENV = "venv"
endif

# Check CMAKE variable before venv creation (only for install target)
CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)

# Check Docker variable (only for docker target)
CHECK_DOCKER = $(shell docker -v)

# Check python3 globally
PYTHON=$(shell command -v python3)
ifeq (, $(PYTHON))
    $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

# Check Python version supported globally
PYTHON_VERSION_MIN = 3.8
PYTHON_VERSION_CUR=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')
ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif


################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      PANDORA MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: check
check: ## check if cmake is installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )

.PHONY: venv
venv: check ## create virtualenv in PANDORA_VENV directory if not exists
	@test -d ${PANDORA_VENV} || python3 -m venv ${PANDORA_VENV}
	@${PANDORA_VENV}/bin/python -m pip install --upgrade pip meson-python meson ninja setuptools_scm setuptools wheel pybind11 # no check to upgrade each time
	@touch ${PANDORA_VENV}/bin/activate

.PHONY: cpp_deps
cpp_deps: ## retrieve cpp dependencies
	@${PANDORA_VENV}/bin/meson wrap update-db

.PHONY: install
install: venv ## install pandora without plugins
	@test -f ${PANDORA_VENV}/bin/pandora || . ${PANDORA_VENV}/bin/activate; ${PANDORA_VENV}/bin/pip install --no-build-isolation --editable .[dev,docs,notebook] --config-settings=setup-args=-Dbuild_cpp_tests=enabled
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${PANDORA_VENV}/bin/pre-commit install
	@echo "PANDORA installed in dev mode in virtualenv ${PANDORA_VENV}"
	@echo "PANDORA venv usage : source ${PANDORA_VENV}/bin/activate; pandora -h"

## Test section

.PHONY: test
test: install ## run all tests (except notebooks) + coverage (source venv before)
	@${PANDORA_VENV}/bin/pytest -m "not notebook_tests" --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-notebook
test-notebook: install ## run notebook tests only
	@${PANDORA_VENV}/bin/pytest -m "notebook_pandora"

## Code quality, linting section

### Format with black

.PHONY: format
format: install format/black  ## run black formatting (depends install)

.PHONY: format/black
format/black: install  ## run black formatting (depends install) (source venv before)
	@echo "+ $@"
	@${PANDORA_VENV}/bin/black pandora tests ./*.py notebooks/snippets/*.py

### Check code quality and linting : black, mypy, pylint

.PHONY: lint
lint: install lint/black lint/mypy lint/pylint ## check code quality and linting (source venv before)

.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${PANDORA_VENV}/bin/black --check pandora tests ./*.py notebooks/snippets/*.py

.PHONY: lint/mypy
lint/mypy: ## check linting with mypy
	@echo "+ $@"
	@${PANDORA_VENV}/bin/mypy pandora tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${PANDORA_VENV}/bin/pylint pandora tests --rcfile=.pylintrc --output-format=parseable --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" # | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

## Documentation section

.PHONY: docs
docs: install ## build sphinx documentation (source venv before)
	@${PANDORA_VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${PANDORA_VENV}/bin/sphinx-build -M html docs/source/ docs/build -W --keep-going

## Notebook section
.PHONY: notebook
notebook: install ## install Jupyter notebook kernel with venv and pandora install (source venv before)
	@echo "Install Jupyter Kernel in virtualenv dir"
	@${PANDORA_VENV}/bin/python -m ipykernel install --sys-prefix --name=pandora-dev --display-name=pandora-dev
	@echo "Use jupyter kernelspec list to know where is the kernel"
	@echo " --> After PANDORA virtualenv activation, please use following command to launch local jupyter notebook to open PANDORA Notebooks:"
	@echo "jupyter notebook"

## Docker section

.PHONY: docker
docker: ## Check and build docker image (cnes/pandora)
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Check Dockerfiles with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker main image PANDORA "
	@docker build -t cnes/pandora:dev . -f Dockerfile

## Clean section

.PHONY: clean-notebook-output ## Clean Jupyter notebooks outputs
clean-notebook-output:
	@echo "Clean Jupyter notebooks"
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb notebooks/advanced_examples/*.ipynb


.PHONY: clean
clean: clean-venv clean-build clean-precommit clean-pyc clean-test clean-doc clean-notebook clean-mypy ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${PANDORA_VENV}

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr build/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-precommit
clean-precommit:
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc:
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -fr {} +
	@find . -type d -name "__pycache__" -exec rm -fr {} +
	@find . -name '*~' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@echo "+ $@"
	@rm -fr .tox/
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml
	@rm -f pylint-report.txt
	@rm -f debug.log

.PHONY: clean-doc
clean-doc:
	@echo "+ $@"
	@rm -rf docs/build/
	@rm -rf docs/source/api_reference/

.PHONY: clean-notebook
clean-notebook:
	@echo "+ $@"
	@find . -type d -name ".ipynb_checkpoints" -exec rm -fr {} +

.PHONY: clean-mypy
clean-mypy:
	@echo "+ $@"
	@rm -rf .mypy_cache/

.PHONY: clean-docker
clean-docker: ## clean docker image
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Clean Docker image cnes/pandora dev"
	@docker image rm cnes/pandora:dev

.PHONY: test-unit-cpp
test-unit-cpp: install ## run unit cpp tests only for dev
	@echo "Run unit cpp tests"
	. ${PANDORA_VENV}/bin/activate; meson test -C build/$(shell ls build)/ -v
