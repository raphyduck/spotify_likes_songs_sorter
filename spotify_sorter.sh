#!/bin/bash
# shellcheck shell=bash


if [ ! -f .venv/bin/activate ]; then
  echo "Expected virtual environment at .venv. Please run 'python -m venv .venv' and install dependencies." >&2
  exit 1
fi

source ~/.venv/bin/activate
python ./spotify_sorter.py
