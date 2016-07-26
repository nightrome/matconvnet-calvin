# MatConvNet-Calvin

**MatConvNet-Calvin** is a wrapper around MatConvNet that (re-)implements
several state of-the-art papers in object detection and semantic segmentation.
Calvin is a Computer Vision research group at the University of Edinburgh (http://calvin.inf.ed.ac.uk/).
Copyrights by Holger Caesar and Jasper Uijlings, 2015-2016.

**Implementations:**
- **Fast R-CNN** by Girshick et al., ICCV 2015
  \[1\]
  (incl. bounding box regression, ROI pooling)
- **Region-based semantic segmentation with end-to-end training (E2S2)** by Caesar et al., ECCV 2016
  \[2\]
  (incl. freeform ROI pooling, region-to-pixel layer)
- **Fully Convolutional Networks for Semantic Segmentation (FCN)** by Long et al., CVPR 2015
  \[3\]
  (based on MatConvNet-FCN, modified for arbitrary datasets)
- **Fully Convolutional Multi-Class Multipe Instance Learning** by Pathak et al., ICLR 2015 workshop
  \[4\]
  (based on MatConvNet-FCN and the SegmentationLossImage layer)
- **What's the point: Semantic segmentation with point supervision** by Bearman et al., ECCV 2016 \[5\] (only implemented the image-level supervision)

**Dependencies:**
- **MatConvNet**: beta19 (http://github.com/vlfeat/matconvnet).
- **MatConvNet-FCN**: (http://github.com/vlfeat/matconvnet-fcn)

**Installation:**
- Install Matlab
- Clone the repository "git clone --recursive https://github.com/nightrome/matconvnet-calvin.git"
- Setup MatConvNet
  - cd matconvnet/matlab;
  - vl_compilenn('EnableGpu', true);
  - cd ../..;
- Setup MatConvNet-Calvin
  - cd matconvnet-calvin/matlab;
  - vl_compilenn_calvin();
  - cd ../..;
- Add files to Matlab path
  - setup();
- For missing files please contact Holger Caesar.

**Usage:**
- FCN: TBD
- Fast R-CNN: TBD
- E2S2: TBD

**References:**
- \[1\] http://arxiv.org/abs/1504.08083
- \[2\] TBD
- \[3\] http://arxiv.org/abs/1411.4038
- \[4\] http://arxiv.org/abs/1412.7144
- \[5\] http://arxiv.org/abs/1506.02106

**Contact:**
If you run into any problems with this code, please submit a an issue report on the Github site of the project.
For other inquiries contact holger.caesar-at-ed.ac.uk.