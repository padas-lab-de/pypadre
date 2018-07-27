#!/bin/bash

set -x

# build python and java classes from datasetV1.proto in current directory
protoc --python_out=./ --java_out=./ datasetV1.proto
