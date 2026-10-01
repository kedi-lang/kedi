BLUE := \033[1;34m
GREEN := \033[1;32m
RESET := \033[0m

UV ?= uv
PYTHON_VERSIONS ?= 3.10 3.11 3.12 3.13 3.14
MATRIX_ENV_ROOT ?= /tmp/kedi-python-matrix

.PHONY: all format check-formatted lint ty basedpyright check check-docs check-release tests coverage check-coverage check-matrix playground-tree-sitter prod

all: format check tests

format:
	@printf "$(BLUE)==>$(RESET) Formatting Python code with ruff...\n"
	@$(UV) run --group dev ruff format
	@printf "$(GREEN)✔ Formatting complete.$(RESET)\n"

check-formatted:
	@printf "$(BLUE)==>$(RESET) Checking Python formatting with ruff...\n"
	@$(UV) run --group dev ruff format --check
	@printf "$(GREEN)✔ Formatting check complete.$(RESET)\n"

lint:
	@printf "$(BLUE)==>$(RESET) Linting with ruff...\n"
	@$(UV) run --group dev ruff check
	@printf "$(GREEN)✔ Ruff check complete.$(RESET)\n"

ty:
	@printf "$(BLUE)==>$(RESET) Type checking source packages with ty...\n"
	@$(UV) run --group dev ty check src
	@printf "$(GREEN)✔ ty check complete.$(RESET)\n"

basedpyright:
	@printf "$(BLUE)==>$(RESET) Type checking source packages with basedpyright...\n"
	@$(UV) run --group dev basedpyright src
	@printf "$(GREEN)✔ basedpyright check complete.$(RESET)\n"

check: lint check-formatted ty basedpyright

check-docs:
	@printf "$(BLUE)==>$(RESET) Checking extension DSL docs are synced...\n"
	@cmp -s docs/KEDI_DSL.md kedi-zed/docs/KEDI_DSL.md
	@cmp -s docs/KEDI_DSL.md kedi-vscode/docs/KEDI_DSL.md
	@printf "$(GREEN)✔ Extension DSL docs are synced.$(RESET)\n"

.PHONY: check-distribution
check-distribution:
	@$(UV) run --no-project python scripts/check_distribution.py

check-release: check-distribution
	@printf "$(BLUE)==>$(RESET) Running release surface checks...\n"
	@bash scripts/check_release_surfaces.sh
	@printf "$(GREEN)✔ Release surface checks complete.$(RESET)\n"

playground-tree-sitter:
	@printf "$(BLUE)==>$(RESET) Building browser Tree-sitter assets...\n"
	@bash scripts/build_playground_tree_sitter.sh
	@printf "$(GREEN)✔ Browser Tree-sitter assets built.$(RESET)\n"

tests:
	@printf "$(BLUE)==>$(RESET) Running pytest...\n"
	@$(UV) run --group dev pytest
	@printf "$(GREEN)✔ Tests complete.$(RESET)\n"

coverage:
	@printf "$(BLUE)==>$(RESET) Running pytest with coverage...\n"
	@$(UV) run --group dev pytest --cov=src/kedi -q
	@printf "$(GREEN)✔ Coverage run complete.$(RESET)\n"

check-coverage:
	@printf "$(BLUE)==>$(RESET) Checking full test coverage...\n"
	@$(UV) run --group dev pytest --record-mode=none --cov=src/kedi --cov-report=term-missing --cov-fail-under=100 -q
	@printf "$(GREEN)✔ Coverage threshold satisfied.$(RESET)\n"

check-matrix:
	@for version in $(PYTHON_VERSIONS); do \
		printf "$(BLUE)==>$(RESET) Running checks on Python $$version...\n"; \
		if [ ! -f "$(MATRIX_ENV_ROOT)/$$version/pyvenv.cfg" ]; then \
			$(UV) venv --clear --force --python $$version "$(MATRIX_ENV_ROOT)/$$version" || exit $$?; \
		fi; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(MATRIX_ENV_ROOT)/$$version $(UV) run --group dev --python $$version ruff check || exit $$?; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(MATRIX_ENV_ROOT)/$$version $(UV) run --group dev --python $$version ruff format --check || exit $$?; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(MATRIX_ENV_ROOT)/$$version $(UV) run --group dev --python $$version ty check src --python-version $$version || exit $$?; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(MATRIX_ENV_ROOT)/$$version $(UV) run --group dev --python $$version basedpyright --pythonversion $$version src || exit $$?; \
	done
	@printf "$(GREEN)✔ Matrix checking complete.$(RESET)\n"

prod: check check-coverage
