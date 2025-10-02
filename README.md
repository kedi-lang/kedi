# marker-pattern

Minimal Python package scaffold with pytest.

## Requirements
- Python 3.9+
- macOS/Linux/Windows

## Setup (virtual environment)
```bash
python3 -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests
```bash
pytest
```

## Using the package
```python
from marker_pattern import add
print(add(2, 3))  # 5
```

## Project layout
- `src/marker_pattern/` package code
- `tests/` test suite
- `requirements.txt` minimal dependencies
- `pytest.ini` pytest configuration

## Notes
- You can later add packaging metadata (`setup.cfg`/`pyproject.toml`) if needed.