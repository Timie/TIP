This is an experimental project for implementation of image processing algorithms. It is work-in-progress toy-project - things may change anytime.

Note that implementation is released under MIT licence (see the LICENCE file) but the implemented algorithms may be covered by patents of their respective holders, or protected by other means of protection of intelectual property.


# TODO

* Move to git submodules for internal libraries
* Dehazing:
  * Experiment with postprocessing of transmission map (multiple passes of guided filter, additional blur/dilation, calculating it at half the resolution).
  * Experiment with bilateral filter before dehazing.
  * Experiment with other priors too (Colour Atenuation Prior)
* Matting (guided filter)
  * Add iterations
  * Add support for single channel image
  * Add simpler RGB matting and check the quality (without matrix inverse -> Sum_c(u_c - x_c)^2; Sum_c(var_c) for c => r/g/b)
  * Add subsampling (all) - see Fast guided filter
  * Add subsampling (just for local image variance, I-p covariance, and a)
  * Find faster implementation of (3x3matrix)^-1 *vector.
* Try on CUDA

