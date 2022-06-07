#!/usr/bin/bash

target="facetinyfpnsize640"
prefix="/home/ww/projects/yudet/workspace/${target}/weights/"
python test.py -m ${prefix}${target}_final.pth -d "cpu"

begin=1000
end=500
while ((begin>end))
do
    python test.py -m ${prefix}${target}_epoch_$begin.pth
    ((begin-=50))
done
