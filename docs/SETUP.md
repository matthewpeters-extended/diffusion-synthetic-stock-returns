# Setup

Requires Homebrew Python 3.12. Do not use the system Python 3.9.

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions; use requirements.txt to resolve fresh
pytest
```

Everything runs on CPU. On macOS the default `torch` wheel has no CUDA, so no extra index URL is needed.

To regenerate the lock after changing `requirements.txt`:

```bash
pip install -r requirements.txt && pip freeze > requirements-lock.txt
```
