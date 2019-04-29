#!/usr/bin/env bash
CURR_DIR=`pwd`
PACKAGE_NAME="./wheelhouse/pypadre-0.0.0-py3-none-any.whl"
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo $SCRIPT_DIR
cd $SCRIPT_DIR

if [ ! -d './wheelhouse' ]; then
    mkdir "./wheelhouse"
fi
pip install sphinx
pip wheel . --wheel-dir ./wheelhouse/
pip install --force-reinstall $PACKAGE_NAME
python ./padre/tests/example.py

cd $CURR_DIR
