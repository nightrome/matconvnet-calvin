# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.

**MatConvNet-Calvin** is a wrapper around MatConvNet that (re-)implements
several state of-the-art papers in object detection and semantic segmentation.
Copyrights for all added files by Holger Caesar and Jasper Uijlings, 2015-2016.

Implementations:
- Fast R-CNN by Girshick et al.
  ( http://arxiv.org/abs/1504.08083 )
  (incl. bounding box regression, ROI pooling)

- End-to-end region based semantic segmentation by Caesar et al.
  ( TBD )
  (incl. freeform ROI pooling, region-to-pixel layer)

- Fully Convolutional Networks for Semantic Segmentation by Long et al.
  ( http://arxiv.org/abs/1411.4038 )
  (based on matconvnet-fcn, extended for arbitrary classes)
  ( https://github.com/vlfeat/matconvnet-fcn )

- Fully Convolutional Multi-Class Multipe Instance Learning by Pathak et al.
  ( http://arxiv.org/abs/1412.7144 )
  (based on matconvnet-fcn and a modified loss using the SegmentationLabelPresence layer)

Installation:
- Clone the repository "git clone https://github.com/nightrome/matconvnet-calvin.git"
- Install Matlab
- Install MatConvNet
- Run matlab/vl_compilenn_calvin()
- Run matlab/vl_setupnn_calvin()
