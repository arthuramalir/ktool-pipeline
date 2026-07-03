import os, sys

os.environ.setdefault("KTOOL_PLATFORM_ID", "173_synthetic")
os.environ.setdefault("KTOOL_OUTPUT_SUBDIR", "test")
os.environ.setdefault("KTOOL_PROJECT_NAME", "ALC")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import app
