BLUE := \033[1;34m
GREEN := \033[1;32m
RESET := \033[0m

UV ?= uv
PYTHON_VERSIONS ?= 3.11 3.12

.PHONY: all format check-formatted lint ty basedpyright check tests coverage check-coverage check-matrix prod

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
	@printf "$(BLUE)==>$(RESET) Type checking src with ty...\n"
	@$(UV) run --group dev ty check src
	@printf "$(GREEN)✔ ty check complete.$(RESET)\n"

basedpyright:
	@printf "$(BLUE)==>$(RESET) Type checking src with basedpyright...\n"
	@$(UV) run --group dev basedpyright src
	@printf "$(GREEN)✔ basedpyright check complete.$(RESET)\n"

check: lint check-formatted ty basedpyright

tests:
	@printf "$(BLUE)==>$(RESET) Running pytest...\n"
	@$(UV) run --group dev pytest
	@printf "$(GREEN)✔ Tests complete.$(RESET)\n"

coverage:
	@printf "$(BLUE)==>$(RESET) Running pytest with coverage...\n"
	@$(UV) run --group dev pytest --cov=. -q
	@printf "$(GREEN)✔ Coverage run complete.$(RESET)\n"

check-coverage:
	@printf "$(BLUE)==>$(RESET) Checking full test coverage...\n"
	@$(UV) run --group dev pytest --cov=. --cov-fail-under=100 -q
	@printf "$(GREEN)✔ Coverage threshold satisfied.$(RESET)\n"

check-matrix:
	@for version in $(PYTHON_VERSIONS); do \
		printf "$(BLUE)==>$(RESET) Running checks on Python $$version...\n"; \
		$(UV) run --group dev --python $$version ruff check || exit $$?; \
		$(UV) run --group dev --python $$version ruff format --check || exit $$?; \
		$(UV) run --group dev --python $$version ty check src --python-version $$version || exit $$?; \
		$(UV) run --group dev --python $$version basedpyright --pythonversion $$version src || exit $$?; \
	done
	@printf "$(GREEN)✔ Matrix checking complete.$(RESET)\n"

prod: check check-coverage
