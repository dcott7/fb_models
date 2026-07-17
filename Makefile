.PHONY: data train test lint format requirements docs clean

## Fetch and cache nflverse data (pbp, participation, rosters, players, games)
data:
	uv run python -m fb_models.dataset

## Train the play-type classifier end to end and save the artifact
train:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_play_call.ipynb

## Run the test suite
test:
	uv run pytest

## Check formatting and style without making changes
lint:
	uv run black --check src tests examples
	uv run flake8 src tests examples

## Auto-format source
format:
	uv run black src tests examples

## Regenerate requirements.txt from uv.lock, for consumers not using uv
requirements:
	uv export --no-hashes --format requirements-txt -o requirements.txt

## Serve the mkdocs site locally
docs:
	uv run mkdocs serve

## Remove caches
clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
