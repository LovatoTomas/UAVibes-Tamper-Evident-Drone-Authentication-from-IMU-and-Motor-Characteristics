#!/usr/bin/env python3
"""
Launch the UAVibes Streamlit dashboard.

  python run_dashboard.py
  python run_dashboard.py --server.port 8502
"""
import subprocess
import sys
from pathlib import Path

app = Path(__file__).with_name("ui") / "app.py"
cmd = [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:]
subprocess.run(cmd)
