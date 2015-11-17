#include <cmath>
#include "mex.h"

/*
 * [dzdxout] = regiontopixel_backward(boxCount, spMap, dzdx)
 *
 * Go from a pixel level back to region level.
 * This uses the mask saved in the forward pass.
 * 
 * Copyright by Holger Caesar, 2015
 */

void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[])
{
    if (nlhs == 0) {
        return;
    } else if (nlhs != 1 || nrhs != 3) {
        mexErrMsgTxt("Error. Usage: [dzdxout] = regiontopixel_backward(boxCount, spMap, dzdx)");
        return;
    }
    
    // Get pointers
    const mxArray* boxCountMx = input[0];
    const mxArray* spMapMx = input[1];
    const mxArray* dzdxMx = input[2];
    
    // Check inputs
    if (!mxIsDouble(boxCountMx) || !mxIsScalar(boxCountMx)) {
        mexErrMsgTxt("Error: boxCount must be a scalar double!");
    }
    if (!mxIsDouble(spMapMx) || mxGetNumberOfDimensions(spMapMx) != 2) {
        mexErrMsgTxt("Error: spMap must be double with format labelCount x spCount!");
    }
    int labelCount = mxGetM(spMapMx);
    int spCount    = mxGetN(spMapMx);
    const mwSize* dzdxDims = mxGetDimensions(dzdxMx);
    if (!mxIsSingle(dzdxMx) || dzdxDims[0] != 1 || dzdxDims[1] != 1 || dzdxDims[2] != labelCount ||
              (!(mxGetNumberOfDimensions(dzdxMx) == 4 && dzdxDims[3] == spCount)
            && !(mxGetNumberOfDimensions(dzdxMx) == 3))) {
        mexErrMsgTxt("Error: dzdx must be single with format 1 x 1 x labelCount x spCount!");
    }
    
    // Get arrays
    int boxCount  = (int) mxGetScalar(boxCountMx);
    double* spMap = (double*) mxGetData(spMapMx);
    float* dzdx   = (float*) mxGetData(dzdxMx);
    
    // Create output and initialize it to all zeros (in mxCreateNumericArray)
    mwSize dzdxoutSize[4];
    dzdxoutSize[0] = 1;
    dzdxoutSize[1] = 1;
    dzdxoutSize[2] = labelCount;
    dzdxoutSize[3] = boxCount;
    out[0] = mxCreateNumericArray(4, dzdxoutSize, mxSINGLE_CLASS, mxREAL);
    float* dzdxout = (float*) mxGetData(out[0]);
    
    for (int spIdx = 0; spIdx < spCount; spIdx++) {
        for (int labelIdx = 0; labelIdx < labelCount; labelIdx++) {
            // We can safely ignore the first two dimensions of these
            // matrices as they are always 1
            int spMapIdx = labelIdx + spIdx * labelCount;
            double boxIdxD = spMap[spMapIdx];
            int boxIdx = (int) boxIdxD - 1; // Convert from Matlab to C indexing
            if (!mxIsNaN(boxIdxD)) {
                int dzdxoutIdx = labelIdx + boxIdx * labelCount;
                dzdxout[dzdxoutIdx] = dzdxout[dzdxoutIdx] + dzdx[spMapIdx];
            }
        }
    }
}