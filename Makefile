install:
	uv sync --all-groups
	uv run pre-commit install

clean:
	rm -rf .venv
	rm -rf uv.lock

pre-commit-all:
	uv run pre-commit run --all-files

pre-commit:
	uv run pre-commit run
