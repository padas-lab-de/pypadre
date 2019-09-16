#!/bin/bash

# Get version number
if env | grep -q ^CI_COMMIT_TAG=
    then
    export VERSION=$CI_COMMIT_TAG
    else
    export VERSION=$CI_JOB_ID
    fi

if [[ $VERSION == prep-v* ]]; then VERSION=${VERSION:6}; else echo "false"; fi

DVERSION=$(echo $VERSION | awk -F. -v OFS=. 'NF==1{print ++$NF}; NF>1{$NF=sprintf("%0*d", length($NF), ($NF+1)); print}')

echo $VERSION
echo $DVERSION