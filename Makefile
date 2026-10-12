.PHONY: setup lint format typecheck test verify clean

setup:
	pip install -r requirements.txt

lint:
	flake8 ai digital_twin backend tests
	isort --check-only ai digital_twin backend tests

format:
	black ai digital_twin backend tests
	isort ai digital_twin backend tests

typecheck:
	mypy ai digital_twin --ignore-missing-imports

test:
	pytest tests/ -v --cov=ai --cov=digital_twin --cov-report=term-missing

verify:
	python scripts/verify_env.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
