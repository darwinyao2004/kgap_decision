#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_gap_decision.run_experiment import run
from knowledge_gap_decision.report import generate_markdown

if __name__ == "__main__":
    run(100, quick=True)
    generate_markdown()
    print("smoke ok")
