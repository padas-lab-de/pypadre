#!/usr/bin/env bash
CURR_DIR=`pwd`
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $SCRIPT_DIR
cd $SCRIPT_DIR

if [ ! -d '../wheelhouse' ]; then
    mkdir "../wheelhouse"
fi
pip wheel . --wheel-dir ../wheelhouse/
pip uninstall PyPaDRE-Python-Client-for-PADAS-Data-Science-Reproducibility-Environment
pip install ../wheelhouse/PyPaDRE_Python_Client_for_PADAS_Data_Science_Reproducibility_Environment-0.0.1-py3-none-any.whl

cd $CURR_DIR
