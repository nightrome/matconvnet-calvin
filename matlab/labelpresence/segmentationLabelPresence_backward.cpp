#include <cmath>
#include "mex.h"

/*
 * [dzdx] = segmentationLabelPresence_backward(imageSizeY, imageSizeX, mask, dzdy)
 *
 * This script maps from one label to back to the highest scoring pixel in the segmentation.
 * This uses the mask saved in the forward pass.
 * 
 * Copyright by Holger Caesar, 2015
 */

void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[])
{
    if (nlhs == 0) {
        return;
    } else if (nlhs != 1 || nrhs != 4) {
        mexErrMsgTxt("Error. Usage: [dzdx] = segmentationLabelPresence_backward(imageSizeY, imageSizeX, mask, dzdy)");
        return;
    }
    
    // Get pointers
    const mxArray* imageSizeYMx  = input[0];
    const mxArray* imageSizeXMx  = input[1];
    const mxArray* maskMx = input[2];
    const mxArray* dzdyMx = input[3];
    
    // Check inputs
    if (!mxIsDouble(imageSizeYMx) || !mxIsScalar(imageSizeYMx)) {
        mexErrMsgTxt("Error: imageSizeY must be a scalar double!");
    }
    if (!mxIsDouble(imageSizeXMx) || !mxIsScalar(imageSizeXMx)) {
        mexErrMsgTxt("Error: imageSizeX must be a scalar double!");
    }
    if (!mxIsDouble(maskMx) || mxGetNumberOfDimensions(maskMx) != 2) {
        mexErrMsgTxt("Error: mask must be double with format labelCount x sampleCount!");
    }
    int labelCount  = mxGetM(maskMx);
    int sampleCount = mxGetN(maskMx);
    const mwSize* dzdyDims = mxGetDimensions(dzdyMx);
    if (!mxIsSingle(dzdyMx) || dzdyDims[0] != 1 || dzdyDims[1] != 1 || dzdyDims[2] != labelCount ||
              (!(mxGetNumberOfDimensions(dzdyMx) == 4 && dzdyDims[3] == sampleCount)
            && !(mxGetNumberOfDimensions(dzdyMx) == 3))) {
        mexErrMsgTxt("Error: dzdy must be single with format 1 x 1 x labelCount x sampleCount!");
    }
    
    // Get arrays
    int imageSizeY = (int) mxGetScalar(imageSizeYMx);
    int imageSizeX = (int) mxGetScalar(imageSizeXMx);
    double* mask = (double*) mxGetData(maskMx);
    float* dzdy = (float*) mxGetData(dzdyMx);
    
    // Create output and initialize it to all zeros (in mxCreateNumericArray)
    mwSize dzdxSize[4];
    dzdxSize[0] = imageSizeY;
    dzdxSize[1] = imageSizeX;
    dzdxSize[2] = labelCount;
    dzdxSize[3] = sampleCount;
    out[0] = mxCreateNumericArray(4, dzdxSize, mxSINGLE_CLASS, mxREAL);
    float* dzdx = (float*) mxGetData(out[0]);
    
    for (int sampleIdx = 0; sampleIdx < sampleCount; sampleIdx++) {
        for (int labelIdx = 0; labelIdx < labelCount; labelIdx++) {
            // We can safely ignore the first two dimensions of these
            // matrices as they are always 1
            int maskIdx = labelIdx + sampleIdx * labelCount;
            double pixCoordD = mask[maskIdx];
            if (!mxIsNaN(pixCoordD)) {
                int pixCoordX = (int) pixCoordD / imageSizeY - 1;                   // Convert from Matlab to C indexing
                int pixCoordY = (int) pixCoordD - pixCoordX * imageSizeY - 1; // Convert from Matlab to C indexing
                int dzdxIdx = pixCoordY + pixCoordX * (imageSizeY) + labelIdx * (imageSizeY * imageSizeX) + sampleIdx * (imageSizeY * imageSizeX * labelCount);
                dzdx[dzdxIdx] = dzdx[dzdxIdx] + dzdy[maskIdx]; // dzdx(y, x, labelIdx, sampleIdx) += dzdy(1, 1, labelIdx, sampleIdx)
            }
        }
    }
}