# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: build install install-dev clean docs-clean docs-build docs-serve docs-deploy docs-set-default docs-list test test-unit test-integration mypy lint-imports

DOCS_DIR := build/docs
DOCS_BRANCH := gh-pages
TEST_LEVEL ?= L0
PUSH ?= true
PUSH_FLAG := $(if $(filter true,$(PUSH)),--push,)

build: clean
	python3 -m build --wheel

install: build
	pip3 install dist/*.whl

install-dev:
	uv pip install --group dev -e .

clean: docs-clean

docs-clean:
	rm -rf $(DOCS_DIR)
	rm -rf site/  # Default mkdocs output dir (if someone runs mkdocs build directly)

# Build docs locally without versioning (for development/preview)
docs-build: docs-clean
	mkdocs build --clean --site-dir $(DOCS_DIR)

# Serve docs locally for development
docs-serve:
	mkdocs serve

# Deploy versioned docs using mike (for CI or manual release)
# Usage: make docs-deploy VERSION=v1.0.0 ALIAS=latest
#        make docs-deploy VERSION=dev
#        make docs-deploy VERSION=dev PUSH=false  # Don't push (for CI)
docs-deploy:
ifndef VERSION
	$(error VERSION is required. Usage: make docs-deploy VERSION=v1.0.0 ALIAS=latest)
endif
ifdef ALIAS
	mike deploy --update-aliases $(VERSION) $(ALIAS) -b $(DOCS_BRANCH) $(PUSH_FLAG)
else
	mike deploy $(VERSION) -b $(DOCS_BRANCH) $(PUSH_FLAG)
endif

# Set the default version for mike
# Usage: make docs-set-default VERSION=latest
#        make docs-set-default VERSION=latest PUSH=false  # Don't push (for CI)
docs-set-default:
ifndef VERSION
	$(error VERSION is required. Usage: make docs-set-default VERSION=latest)
endif
	mike set-default $(VERSION) -b $(DOCS_BRANCH) $(PUSH_FLAG)

# List all deployed doc versions
docs-list:
	mike list -b $(DOCS_BRANCH)

test-integration:
	@error=0; \
	for dir in tests/integration/*/; do \
		if [ -f "$${dir}test.sh" ] && [ "$$(echo $${dir} | grep -E "$(TEST_LEVEL)")" ]; then \
			echo ""; \
			echo "########################################"; \
			echo "Setting up isolated venv for $${dir}"; \
			echo "########################################"; \
			echo ""; \
			if [ ! -f "$${dir}requirements.txt" ]; then \
				echo "Warning: $${dir}requirements.txt not found, skipping..."; \
				continue; \
			fi; \
			if [ ! -d "$${dir}.venv" ]; then \
				echo "Creating venv at $${dir}.venv"; \
				python3 -m venv --copies $${dir}.venv || { error=1; continue; }; \
			else \
				echo "Using existing venv at $${dir}.venv"; \
			fi; \
			echo "Installing dependencies..."; \
			( \
				set -e; \
				trap 'deactivate 2>/dev/null || true' EXIT; \
				source $${dir}.venv/bin/activate && \
				uv pip install --group test -e . -r $${dir}requirements.txt && \
				echo "========================================"; \
				echo "Running $${dir}test.sh"; \
				echo "========================================"; \
				echo ""; \
				$${dir}test.sh \
			) || error=1; \
		fi; \
	done; \
	exit $$error


test-unit:
	pytest -sv tests/unit --cov=flextensor --cov-report=html --cov-report=xml --cov-report=term-missing

mypy:
	uv run --frozen --group test mypy src/flextensor

lint-imports:
	PYTHONPATH=src uv run --no-project --with import-linter==2.12 lint-imports --config pyproject.toml --no-cache

test: test-integration test-unit
