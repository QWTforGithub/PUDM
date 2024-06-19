#!/bin/bash
# cd pointnet2
#rm -r exp_pu1k
#rm -r exp_pugan
#rm -r exp_scannet


cd Chamfer3D
rm -r build
echo "Chamfer3D/build --> Finish!"
rm -r dist
echo "Chamfer3D/dist --> Finish!"
rm -r chamfer_3D.egg-info
echo "Chamfer3D/chamfer_3D.egg-info --> Finish!"
pip uninstall chamfer-3D


cd ..
cd pointnet2_ops_lib
rm -r build
echo "pointnet2_ops_lib/build --> Finish!"
rm -r dist
echo "pointnet2_ops_lib/dist --> Finish!"
rm -r pointnet2_ops.egg-info
echo "pointnet2_ops_lib/pointnet2_ops.egg-info --> Finish!"
pip uninstall pointnet2-ops

cd ..
cd pointops
rm -r build
echo "pointops/build --> Finish!"
rm -r dist
echo "pointops/dist --> Finish!"
rm -r pointops.egg-info
echo "pointops/pointops.egg-info --> Finish!"
pip uninstall pointops

cd ..
rm -r build
rm -r dist
rm -r pointnet2.egg-info
pip uninstall pointnet2