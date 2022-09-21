# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
# Dependencies : python3 venv internal module
# Recall: .PHONY  defines special targets not associated with files
#
# Some Makefile global variables can be set in make command line:
# PANDORA_VENV: Change directory of installed venv (default local "venv" dir)

############### GLOBAL VARIABLES ######################

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: PANDORA_VENV="other-venv/" make install
ifndef PANDORA_VENV
	PANDORA_VENV = "venv"
endif

# Check CMAKE variables before venv creation
CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)


# Check Docker
CHECK_DOCKER = $(shell docker -v)

# PANDORA version from setup.py
PANDORA_VERSION = $(shell python3 setup.py --version)
PANDORA_VERSION_MIN =$(shell echo ${PANDORA_VERSION} | cut -d . -f 1,2,3)
################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      PANDORA MAKE HELP"
	@echo "  build, check or test !"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: check
check: ## check if cmake is installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )

.PHONY: venv
venv: check ## create virtualenv in PANDORA_VENV directory if not exists
	@test -d ${PANDORA_VENV} || python3 -m venv ${PANDORA_VENV}
	@${PANDORA_VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${PANDORA_VENV}/bin/activate

.PHONY: install
install: venv  ## install pandora (not editable)
	@test -f ${PANDORA_VENV}/bin/pandora || ${PANDORA_VENV}/bin/pip install .
	@echo "PANDORA ${PANDORA_VERSION} installed in virtualenv ${PANDORA_VENV}"
	@echo "PANDORA venv usage : source ${PANDORA_VENV}/bin/activate; pandora -h"

.PHONY: install-with-plugin
install-with-plugin: venv  ## install pandora (not editable) with plugins for stereo reconstruction (sgm, mccnn)
	@test -f ${PANDORA_VENV}/bin/pandora || ${PANDORA_VENV}/bin/pip install .[sgm,mccnn]
	@echo "PANDORA ${PANDORA_VERSION} installed in virtualenv ${PANDORA_VENV}"
	@echo "PANDORA venv usage : source ${PANDORA_VENV}/bin/activate; pandora -h"

.PHONY: install-dev
install-dev: venv ## install pandora in dev editable mode (pip install -e .)
	@test -f ${PANDORA_VENV}/bin/pandora || ${PANDORA_VENV}/bin/pip install -e .[dev,docs,notebook,sgm,mccnn]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${PANDORA_VENV}/bin/pre-commit install
	@echo "PANDORA ${PANDORA_VERSION} installed in dev mode in virtualenv ${PANDORA_VENV}"
	@echo "PANDORA venv usage : source ${PANDORA_VENV}/bin/activate; pandora -h"

.PHONY: install-notebook
install-notebook: venv
	@test -f ${PANDORA_VENV}/bin/pandora || ${PANDORA_VENV}/bin/pip install -e .[notebook]
	@echo "PANDORA venv usage : source ${PANDORA_VENV}/bin/activate; pandora -h"

## Install section for ci

.PHONY: install-ci
install-ci:
	@python -m pip install --upgrade pip
	@pip install .[dev]

.PHONY: distribute
distribute:
	@python -m build --sdist

## Test section

.PHONY: test
test: install-dev ## run all tests + coverage html
	@${PANDORA_VENV}/bin/pytest -m "not notebook_tests" --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-notebook
test-notebook: install-dev ## run notebook tests only
	@${PANDORA_VENV}/bin/pytest -m "notebook_tests" --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-ci
test-ci:
	@export NUMBA_DISABLE_JIT=1
	@pytest -m "not notebook_tests" --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-notebook-hpc
test-notebook-hpc: install-notebook
	@pytest -m notebook_pandora

## Code quality, linting section

### Format with isort and black

.PHONY: format
format: install-dev  ## run black formatting (depends install)
	@echo "+ $@"
	@${PANDORA_VENV}/bin/pre-commit run black --all-files
	@${PANDORA_VENV}/bin/pre-commit run nbstripout --all-files

### Check code quality and linting : isort, black, flake8, pylint

.PHONY: lint
lint: install-dev lint/black lint/mypy lint/pylint lint/nbstripout## check code quality and linting

.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${PANDORA_VENV}/bin/pre-commit run black --all-files

.PHONY: lint/mypy
lint/mypy: ## check linting with mypy
	@echo "+ $@"
	@${PANDORA_VENV}/bin/pre-commit run mypy --all-files

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@${PANDORA_VENV}/bin/pre-commit run pylint --all-files

.PHONY: precommit
precommit: install-dev## apply precommit to all files
	@echo "+ $@"
	@${PANDORA_VENV}/bin/pre-commit run --all-files

## Documentation section

.PHONY: doc
doc: install-dev ## build sphinx documentation
	@${PANDORA_VENV}/bin/pre-commit run sphinx-checking --hook-stage push

## Notebook section
.PHONY: notebook
notebook: install ## install Jupyter notebook kernel with venv and pandora install
	@echo "Install Jupyter Kernel in virtualenv dir"
	@${PANDORA_VENV}/bin/python -m ipykernel install --sys-prefix --name=pandora-${PANDORA_VERSION_MIN} --display-name=pandora-${PANDORA_VERSION_MIN}
	@echo "Use jupyter kernelspec list to know where is the kernel"
	@echo " --> After PANDORA virtualenv activation, please use following command to launch local jupyter notebook to open PANDORA Notebooks:"
	@echo "jupyter notebook"


# Dev section

.PHONY: dev
dev: install-dev docs notebook ## Install PANDORA in dev mode : install-dev, notebook and docs

## Docker section

.PHONY: docker
docker: ## Check and build docker image (cnes/pandora)
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Check Dockerfiles with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker main image PANDORA ${PANDORA_VERSION_MIN}"
	@docker build -t cnes/pandora:${PANDORA_VERSION_MIN} . -f Dockerfile

## Clean section
.PHONY: clean
clean: clean-venv clean-build clean-precommit clean-pyc clean-test clean-doc clean-notebook ## remove all build, test, coverage and Python artifacts

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
	@rm -rf doc/build/
	@rm -rf doc/sources/api_reference/

.PHONY: clean-notebook
clean-notebook:
	@echo "+ $@"
	@find . -type d -name ".ipynb_checkpoints" -exec rm -fr {} +

.PHONY: clean-docker
clean-docker: ## clean docker image
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Clean Docker image cnes/pandora ${PANDORA_VERSION_MIN}"
	@docker image rm cnes/pandora:${PANDORA_VERSION_MIN}
