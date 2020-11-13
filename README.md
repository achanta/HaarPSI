# HaarPSI
Implementation of the HaarPSI metric


This is a re-implmentation of the Python code that implements the HaarPSI metric introduced in
the following paper:

R. Reisenhofer, S. Bosse, G. Kutyniok and T. Wiegand.
A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment. (PDF)
Signal Processing: Image Communication, vol. 61, 33-43, 2018.

The original Python implmentation can be found here:

http://www.haarpsi.org/

or here:

https://github.com/rgcda/haarpsi

The original Python code computes haar gradients that are inaccurate and inefficient. This has
been fixed in this code. As a result, this version is more accurate, and about 3 times faster (than the CPU version).
This version of the code is also simpler to understand.

# NOTES:

[1] Please note that as a result of using more accurate haar gradients, the similarity value
returned may be slightly different from the one obtained from the original code.

[2] The original code limits the gradient computation to 3 scales only. This is the case here
too. But the code generalizes to a greater number of scales too. For higher scales the returned similarity value may exceed 1 by a little, which is an artifact of the method.

[3] For a rather weak reason (namely, viewing scale), in the original code, every input image is
downsampled by 2 in both dimensions. This is mimicked in this code.
