#include "mex.h"

/*
 * [scoresWeighted] = similarityMap_mapping(scoresSoftMax, similarities)
 *
 * Maps from softmaxed scores / probabilities to a linear combination of
 * those using the given similarity matrix M_{ij} between classes i and j.
 *
 * Inputs and outputs are all single-precision!
 *
 * Copyright by Holger Caesar, 2016
 */

void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[])
{
    // Check number of input and output arguments
    if (nlhs == 0) {
        return;
    } else if (nlhs != 1 || nrhs != 2) {
        mexErrMsgTxt("Error. Usage: [scoresWeighted] = similarityMap_mapping(scoresSoftMax, similarities)");
        return;
    }
    
    // Get pointers
    const mxArray* scoresSoftMaxMx = input[0];
    const mxArray* similaritiesMx = input[1];
    
    // Check inputs
    if (!mxIsSingle(scoresSoftMaxMx) || mxGetNumberOfDimensions(scoresSoftMaxMx) != 3) {
        mexErrMsgTxt("Error: scoresSoftMax must be single with format height x width x labelCount!");
    }
    if (!mxIsSingle(similaritiesMx) || mxGetNumberOfDimensions(similaritiesMx) != 2) {
        mexErrMsgTxt("Error: similarities must be single with format labelCount x labelCount!");
    }
    const mwSize* scoresSoftMaxDims = mxGetDimensions(scoresSoftMaxMx);
    int sizeY = scoresSoftMaxDims[0];
    int sizeX = scoresSoftMaxDims[1];
    int labelCount = scoresSoftMaxDims[2];
    int labelCount2 = mxGetM(similaritiesMx);
    int labelCount3 = mxGetN(similaritiesMx);
    if (sizeY == 0 || sizeX == 0 || labelCount == 0) {
        mexErrMsgTxt("Error: All input dimensions must be > 0!");
    }
    if (labelCount != labelCount2 || labelCount2 != labelCount3) {
        mexErrMsgTxt("Error: labelCount must be the same in scoresSoftMax and similarities!");
    }
    
    // Get arrays
    float* scoresSoftMax = (float*) mxGetData(scoresSoftMaxMx);
    float* similarities = (float*) mxGetData(similaritiesMx);
    
    // Create output and initialize it to all zeros (in mxCreateNumericArray)
    out[0] = mxCreateNumericArray(3, scoresSoftMaxDims, mxSINGLE_CLASS, mxREAL);
    float* scoresWeighted = (float*) mxGetData(out[0]);
    
    // Apply mapping
    // for y = 1 : size(obj.scoresSoftMax, 1)
    //     for x = 1 : size(obj.scoresSoftMax, 2)
    //         for z = 1 : size(obj.scoresSoftMax, 3)
    //             scoresWeighted(y, x, z) = sum(squeeze(scoresSoftMax(y, x, :)) .* similarities(z, :)');
    //         end
    //     end
    // end
    for (int y = 0; y < sizeY; y++) {
        for (int x = 0; x < sizeX; x++) {
            for (int z = 0; z < labelCount; z++) {
                float curSum = 0.0f;
                for (int s = 0; s < labelCount; s++) {
                    // curSum += scoresSoftMax[y, x, s] * similarities[z, s];
                    curSum += scoresSoftMax[y + x * sizeY + s * sizeY * sizeX] * similarities[z + s * labelCount];
                }
                scoresWeighted[y + x * sizeY + z * sizeY * sizeX] = curSum;
            }
        }
    }
}