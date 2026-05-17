#!/bin/bash
# encoding: utf-8
# Copyright (c) 2026- MAGO
# AUTHORS
# Sukbong Kwon (Galois)

set -euo pipefail

. ./path.sh || exit 1;

PYTHON_BIN=${PYTHON_BIN:-python3}
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-requirements.txt}

# Set python virtual environment
if [ ! -x "${VENV_PATH}/bin/python" ]; then
    echo "Installing '${VENV_PATH}' python virtual environment."
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

if ! "${VENV_PATH}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Installing pip in '${VENV_PATH}'."
    "${VENV_PATH}/bin/python" -m ensurepip --upgrade
fi

# Activate python virtual environment
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install requirements.
echo "Installing requirements..."
python -m pip install -r "${REQUIREMENTS_FILE}"

cat <<'EOF'

SAM-Audio setup is complete.

Before running models, request access to:
  https://huggingface.co/facebook/sam-audio-large

Then authenticate in this environment with:
  huggingface-cli login

EOF
echo "Done."
