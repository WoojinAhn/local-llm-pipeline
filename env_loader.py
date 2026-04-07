"""
Minimal .env file loader using only Python stdlib.

Reads KEY=VALUE pairs from a .env file and sets them in os.environ.
Existing environment variables take precedence (so explicit env vars
in the shell command line override .env values).

Usage:
    from env_loader import load_env
    load_env()  # loads .env from current directory
    load_env("/path/to/custom.env")
"""

import os


def load_env(path: str = ".env") -> int:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    - Lines starting with # are comments.
    - Blank lines are ignored.
    - Quoted values (single or double) are unquoted.
    - Existing os.environ values are NOT overwritten.

    Returns the number of variables loaded. Returns 0 silently if file missing.
    """
    if not os.path.isfile(path):
        return 0

    loaded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            # Don't overwrite existing env vars (shell takes precedence)
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1

    return loaded
