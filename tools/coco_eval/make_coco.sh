#!/usr/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
# git clone https://github.com/cocodataset/cocoapi.git $CURRENT_DIR/cocoapi
cd $CURRENT_DIR/cocoapi/PythonAPI/
make

