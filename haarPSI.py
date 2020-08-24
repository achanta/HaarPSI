
'''
This is a re-implmentation of the Python code that implements the HaarPSI metric introduced in
the following paper:

R. Reisenhofer, S. Bosse, G. Kutyniok and T. Wiegand.
A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment. (PDF)
Signal Processing: Image Communication, vol. 61, 33-43, 2018.

The original Python implmentation can be found here:

http://www.haarpsi.org/

or here:

https://github.com/rgcda/haarpsi

The original Python code computes haar gradients that are iaccurate and inefficient. This has
been fixed in this code. As a result, this version is more accurate, and about 3 times faster.
This version of the code is also simpler to understand.

NOTES:

[1] Please note that as a result of using more accurate haar gradients, the similarity value
returned may be slightly different from the one obtained from the original code.

[2] The original code limits the gradient computation to 3 scales only. This is the case here
too. But the code generalizes to a greater number of scales too.

[3] For a rather weak reason (viewing scale), in the original code, every input image is
downsampled by 2 in both dimensions. This is mimicked in this code.

--------------------------
24 August 2020
(c) Radhakrishna Achanta
--------------------------
'''

import numpy as np


# This function takes caer of both color and grayscale images
def compute_HaarPSI_similarity(ref,img):

    
    def subsample(mat):

        mat = mat.astype(np.float64)
        out = (mat[0:-1, 0:-1,...] + mat[1:, 1:,...] + mat[1:, 0:-1,...] + mat[0:-1, 1:,...])/4
        return out[::2,::2,...]


    def RGB2YIQ(rgb):
        Y = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        I = 0.596 * rgb[:, :, 0] - 0.274 * rgb[:, :, 1] - 0.322 * rgb[:, :, 2]
        Q = 0.211 * rgb[:, :, 0] - 0.523 * rgb[:, :, 1] + 0.312 * rgb[:, :, 2]

        return Y,I,Q

    def compute_haar_gradients(x,scales=3,doavg=True):

        grady = np.zeros((scales,)+x.shape)
        gradx = np.zeros((scales,)+x.shape)

        for s in range(scales):

            x2 = (x[:,:-1] + x[:,1:])*0.5 # average along rows
            grady[s,:-1,:-1] = x2[:-1,:] - x2[1:,:] # compute vertical gradients
            
            y2 = (x[:-1,:] + x[1:,:])*0.5 # average along columns
            gradx[s,:-1,:-1] = y2[:,:-1] - y2[:,1:] # compute horizontal gradients

            x[:-1,:-1] = (x2[:-1,:] + y2[:,:-1]) # average and reassign to x
            x = x*0.5

        return np.concatenate((grady, gradx),axis=0)


    def compute_avg(inp):

        out = np.zeros(inp.shape)
        out[:-1,:-1] = (inp[0:-1, 0:-1] + inp[1:, 1:] + inp[1:, 0:-1] + inp[0:-1, 1:])/4
        return out


    def compute_weights(coeff_refy, coeff_imgy):
        # Take the maxmimum between the absolute value of the gradients of reference and distorted images
        # for the coarsest level gradients
        v,h = scales-1, scales+scales-1
        wts_vert = np.maximum(np.abs(coeff_refy[v]),np.abs(coeff_imgy[v])) # coarsest vertical gradients
        wts_hori = np.maximum(np.abs(coeff_refy[h]),np.abs(coeff_imgy[h])) # coarsest horizontal gradients
        # wts_hv = (wts_hori+wts_vert)/2
        return wts_hori, wts_vert


    def compute_local_similarities_Y(coeff_refy, coeff_imgy):
        # Collect the absolute value of all the fine gradients for the reference image
        mag_ref_vert = np.abs(np.stack([coeff_refy[i] for i in range(scales-1)]))
        mag_ref_hori = np.abs(np.stack([coeff_refy[i+scales] for i in range(scales-1)]))

        # Collect the absolute value of all the fine gradents for the distorted image 
        mag_img_vert = np.abs(np.stack([coeff_imgy[i] for i in range(scales-1)]))
        mag_img_hori = np.abs(np.stack([coeff_imgy[i+scales] for i in range(scales-1)]))

        # Compute the normalized correlation of the gradient magnitudes at the finest level
        local_sim_vert = np.sum((2 * mag_ref_vert * mag_img_vert + C)/(mag_ref_vert**2 + mag_img_vert**2 + C),axis=0)/2 # vertical
        local_sim_hori = np.sum((2 * mag_ref_hori * mag_img_hori + C)/(mag_ref_hori**2 + mag_img_hori**2 + C),axis=0)/2 # horizontal

        return local_sim_hori, local_sim_vert


    def compute_local_similarities_IQ(coeff_refi, coeff_refq, coeff_imgi, coeff_imgq):

        similarity_i = (2 * coeff_refi * coeff_imgi + C) / (coeff_refi**2 + coeff_imgi**2 + C)
        similarity_q = (2 * coeff_refq * coeff_imgq + C) / (coeff_refq**2 + coeff_imgq**2 + C)
        local_sim_iq = (similarity_i + similarity_q)/2

        return local_sim_iq


    # sigmoid function scaed by alpha
    def sigmoid(value, alpha):
        return 1.0 / (1.0 + np.exp(-alpha * value))

    # the inverse of the sigmoid function (i.e recovering x from sigmoid values)
    def logit(value, alpha):
        return np.log(value/(1 - value)) / alpha

    #----------------------------------
    # The main function.
    # Expected image shape is H,W (gray) or H,W,C (color) for the reference (ref) and distorted (img) images
    #----------------------------------
    def compute_similarity(ref,img):
        
        color_image = (3 == len(img.shape)) # expected image shape is H,W (gray) or H,W,C (color)

        refy = subsample(ref)
        imgy = subsample(img)

        if color_image:
            refy,refi,refq = RGB2YIQ(refy.astype(np.float64))
            imgy,imgi,imgq = RGB2YIQ(imgy.astype(np.float64))

        coeff_refy = compute_haar_gradients(refy,scales,True)
        coeff_imgy = compute_haar_gradients(imgy,scales,True)

        wts_hori, wts_vert      = compute_weights(coeff_refy, coeff_imgy)
        sim_yhori, sim_yvert    = compute_local_similarities_Y(coeff_refy, coeff_imgy)

        weights                 = np.stack((wts_hori, wts_vert))
        local_similarities      = np.stack((sim_yhori, sim_yvert))

        if color_image:
            # compute one additional term for weights and local_similarities in case of color images
            coeff_refi = np.abs(compute_avg(refi))
            coeff_refq = np.abs(compute_avg(refq))
            coeff_imgi = np.abs(compute_avg(imgi))
            coeff_imgq = np.abs(compute_avg(imgq))

            sim_iq     = compute_local_similarities_IQ(coeff_refi, coeff_refq, coeff_imgi, coeff_imgq)

            weights                 = np.stack((wts_hori, wts_vert, (wts_hori+wts_vert)/2))
            local_similarities      = np.stack((sim_yhori, sim_yvert, sim_iq))

        
        similarity = logit(np.sum(sigmoid(local_similarities[:], alpha) * weights[:]) / np.sum(weights[:]), alpha)**2

        return similarity


    if ref.shape != img.shape:
        raise ValueError("The shapes of the reference image and the distorted image do not match.")
    #----------------------------------
    # Constants for the whole function
    #----------------------------------
    C = 30.0    # experimentally determined constant
    alpha = 4.2 # experimentally determined constant
    scales = 4

    return compute_similarity(ref,img)





