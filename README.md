# Minimal 3D Gaussian Splatting without pytorch

This is a personal learning project replicating [vanilla 3D Gaussian](https://github.com/graphdeco-inria/gaussian-splatting) without using pytorch. Most of the work is in CUDA buffer manual management and implementing the densification logic. Minimal changes to forward and backward codes. 

## Building the project

This project uses standard CMake workflows. Clone the repository and then do the build.

``` sh
mkdir build
cd build
# Configure 
cmake ..

# Build the 'train_app' executable, parallel compilation
make -j8
```

## Usage

Starting point should be colmap sparse data and images. Expecting two folders in the data folder `images/` and `sparse/`

Run train_app on command line with the following example. The code will periodically save point clouds, training checkpoints, and debug rendered images. Training and saving parameters can be modified in the `/src/train.cu` file.

``` sh
# start a fresh run
./train_app ./data/garden garden_experiment_01

# resume a previous run
./train_app ./data/garden garden_experiment_01 step_500
```

## External libraries used

[nanoflann](https://github.com/jlblancoc/nanoflann) for building KDTree in the initialization phase
[fused_ssim](https://github.com/rahul-goel/fused-ssim) for calculating ssim loss
[stb_image](https://github.com/nothings/stb) for image IO
