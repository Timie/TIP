# Ts Image Processing library

This is an experimental project aimed to implement selected image processing algorithms. It is a personal work-in-progress toy-project - things may change anytime.

## Algorithms
* *Single Image Haze Removal Using Dark Channel Prior* by Kaiming He, Jian Sun, and Xiaoou Tang (2011)
  * Core part of the algorithm is the same, but instead of extremmely computation heavy soft-matting algorithm described in the paper (*A Closed Form Solution to Natural Image Matting* by Anat Levin, Dani Lischinski, Yair Weiss), this implementation uses matting based on fast guided filter (*Fast Guided Filter* by Kaiming He, Jian Sun)
* Full RGB Matting based on *Fast Guided Filter* by Kaiming He, Jian Sun

## Dependencies
* [OpenCV](https://github.com/opencv/opencv) - image processing (tested with 4.5.4 + contrib package) - Apache 2.0 Licence
* [Eigen](https://gitlab.com/libeigen/eigen) - linear algebra (tested with 3.3.7) - Apache 2.0 Licence
* [Hedley](https://github.com/nemequ/hedley)  - compiler macros (tested with v15 - included) - CC0-1.0 Licence (public domain)

## Licence

Note that implementation is released under MIT licence (see the LICENCE file in the repo root) but the implemented algorithms may be covered by patents of their respective holders, or protected by other means of protection of intelectual property.

(c) 2023 Adam Babinec


## TODO
Items to be tackled next (not in any particular order):
* Move to git submodules for internal libraries
* Integrate/reuse some of the benchmarks for dehazing, eg. http://www.alphamatting.com
* Dark Channel Prior Dehazing:
  * Experiment with postprocessing of transmission map (multiple passes of guided filter, additional blur/dilation, calculating it at half the resolution).
  * Experiment with bilateral filter before dehazing.
  * Experiment a combination with other priors too (Colour Atenuation Prior)
* Matting using Fast Guided Filter
  * Add iterations
  * Add support for single channel image
  * Add simpler RGB matting and check the quality (without matrix inverse -> Sum_c(u_c - x_c)^2; Sum_c(var_c) for c => r/g/b)
  * Add subsampling (all) - see Fast guided filter
  * Add subsampling (just for local image variance, I-p covariance, and a)
  * Find faster implementation of (3x3matrix)^-1 *vector.
* New algorithms:
  * *A fusiob-based enhancing method for weakly illuminated images* by Xueyang Fu et alii
  * *Fast Single Image Fog Removal using Edge-Preserving Smoothing* by Jing Yu and Qingmin Liao
  * *A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior* by Qingsong Zhu et alii
* Try on CUDA

