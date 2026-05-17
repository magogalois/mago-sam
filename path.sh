#! /usr/bin/env bash
# encoding: utf-8
# Copyright (c) 2026- MAGO
# AUTHORS
# Sukbong Kwon (Galois)

export PATH="${PATH}:${PWD}"
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"

VENV_PATH=./.venv
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi
