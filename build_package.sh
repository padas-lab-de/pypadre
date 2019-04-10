#!/usr/bin/env bash
CURR_DIR=`pwd`
PACKAGE_NAME="./wheelhouse/PyPaDRE_Python_Client_for_PADAS_Data_Science_Reproducibility_Environment-"$VERSION"-py3-none-any.whl"
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $SCRIPT_DIR
cd $SCRIPT_DIR

if [ ! -d './wheelhouse' ]; then
    mkdir "./wheelhouse"
fi
pip wheel . --wheel-dir ./wheelhouse/
pip install --force-reinstall $PACKAGE_NAME

cd $CURR_DIR
