function [doubleIm, uint8im] = ImageRead(imageName)
% image = ImageRead(imageName)
%
% Loads an image from the dataset usind the DATAopts.imgpath. Returns a
% 3 dimensional RGB array.
%
% If DATAopts.maxImSize exists, make sure that the length and width of the
% image is never bigger than maxImSize.
%
% imageName:        Name of the image
%
% image:            N x M x 3 double array with image data
%
%           Jasper Uijlings - 2013

global DATAopts;

% Load image
im = imread(sprintf(DATAopts.imgpath, imageName));
doubleIm = im2double(im);
uint8im = im2uint8(im);

% Deal with maxsize
if isfield(DATAopts, 'maxImSize')
    currSize = max(size(doubleIm));
    if currSize > DATAopts.maxImSize
        scale = DATAopts.maxImSize / currSize;
        doubleIm = imresize(doubleIm, scale);
        uint8im = imresize(uint8im, scale);
    end
end

% Deal with images which are secretly videos
if size(doubleIm,4) > 1
    warning('4 dimensional image!\nTaking only 1st dimension of image:\n%s\n', imageName);
    doubleIm = doubleIm(:,:,:,1);
    uint8im = uint8im(:,:,:,1);
end

% Make RGB image from grayscale image
if size(doubleIm,3) == 1
    doubleIm = repmat(doubleIm, [1 1 3]);
    uint8im = repmat(uint8im, [1 1 3]);
end

