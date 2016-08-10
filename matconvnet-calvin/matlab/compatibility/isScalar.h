/*
 * Workaround in case Matlab's mxIsScalar is not available
 *
 * Copyright by Holger Caesar, 2016
 */

bool isScalar(const mxArray *arr) {
    int mrows = mxGetM(arr);
    int ncols = mxGetN(arr);
    
    // Check that this matrix has <= 2 dimensions
    if (mxGetNumberOfDimensions(arr) <= 2) {
        mexErrMsgTxt("Error: Compatibility script isScalar() does not support matrices with >2 dimensions!");
    }
    
    // Check format
    if (mrows == 1 && ncols == 1) {
        return true;
    } else {
        return false;
    }
}