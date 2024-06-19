#!/bin/bash
cd Chamfer3D
python setup.py install
cd ../
echo "---- Chamfer3D--->Finish! ----"

cd pointnet2_ops_lib
python setup.py install
cd ../
echo "---- pointnet2_ops_lib--->Finish! ----"

cd pointops
python setup.py install
cd ../
echo "---- pointops--->Finish! ----"

python setup.py install