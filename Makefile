PYTHON = uv run python
MODULE = src

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m $(MODULE)

debug:
	$(PYTHON) -m pdb -m $(MODULE)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "All cleaned!"

lint:
	uv run flake8 $(MODULE)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(MODULE)

lint-strict:
	uv run flake8 $(MODULE)
	uv run mypy $(MODULE) --strict
