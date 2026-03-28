"""
Test configuration — ensure imports work in the local development layout.

In Docker, app.py, s3_upload.py, and network_volume.py are all at /.
Locally, network_volume.py lives under src/. This conftest adds both
the project root and src/ to sys.path so imports resolve correctly.
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")

for path in (ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)
