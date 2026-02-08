# Mumble Voice Bot - Development Makefile
# Uses uv for fast Python package management

.PHONY: help install dev lint format typecheck test test-cov clean run

# Default target
help:
	@echo "Mumble Voice Bot - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install dev dependencies (lint, test, etc.)"
	@echo ""
	@echo "Quality:"
	@echo "  make lint        Run ruff linter"
	@echo "  make format      Format code with ruff"
	@echo "  make typecheck   Run mypy type checker"
	@echo "  make check       Run all checks (lint + typecheck)"
	@echo ""
	@echo "Testing:"
	@echo "  make test        Run tests with pytest"
	@echo "  make test-cov    Run tests with coverage report"
	@echo ""
	@echo "Other:"
	@echo "  make clean       Remove build artifacts and caches"
	@echo "  make compile     Syntax-check all Python files"

# Installation
install:
	uv sync

dev:
	uv sync --dev

# Linting and formatting
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy mumble_voice_bot/ --ignore-missing-imports

check: lint typecheck

# Testing
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=mumble_voice_bot --cov-report=term-missing

test-quick:
	uv run pytest tests/ -v -x --tb=short

# Compile check (syntax validation)
compile:
	uv run python -m py_compile mumble_tts_bot.py
	uv run python -m py_compile mumble_voice_bot/*.py
	uv run python -m py_compile mumble_voice_bot/**/*.py
	@echo "All files compile OK"

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf mumble_voice_bot/__pycache__ mumble_voice_bot/**/__pycache__
	rm -rf tests/__pycache__
	rm -rf *.egg-info build dist
	rm -rf .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Run the bot (requires config)
run:
	uv run python mumble_tts_bot.py --config config.yaml
