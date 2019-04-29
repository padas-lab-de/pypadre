#!/usr/bin/env bash
CURR_DIR=`pwd`
PACKAGE_NAME="./wheelhouse/pypadre-0.0.0-py3-none-any.whl"
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo $CURR_DIR
echo $SCRIPT_DIR
cd $SCRIPT_DIR

if [ ! -d './wheelhouse' ]; then
    mkdir "./wheelhouse"
fi
pip install sphinx
pip wheel . --wheel-dir ./wheelhouse/
pip install --force-reinstall $PACKAGE_NAME

cd ../test_folder
python3 ../PyPaDRe/padre_package_test.py

cd $CURR_DIR
