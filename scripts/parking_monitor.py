#!/usr/bin/env python3
import sys
import os

# Garante que o diretório src/ esteja no PYTHONPATH quando o script for executado diretamente
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from parking_monitor.cli import main

if __name__ == "__main__":
    sys.exit(main())
