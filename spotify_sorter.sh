#!/bin/bash
# shellcheck shell=bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
  echo "Expected virtual environment at .venv. Please run 'python -m venv .venv' and install dependencies." >&2
  exit 1
fi

source "$SCRIPT_DIR/.venv/bin/activate"
python ./spotify_sorter.py
