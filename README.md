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
- **MatConvNet**: beta20 (http://github.com/vlfeat/matconvnet)
- **MatConvNet-FCN**: (http://github.com/vlfeat/matconvnet-fcn)

**Installation:**
- Install Matlab and Git
- Clone the repository and its submodules
  - git clone https://github.com/nightrome/matconvnet-calvin.git
  - cd matconvnet-calvin
  - git submodule update --init
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

**Usage:**
- **E2S2**: Run demo_e2s2(). This trains a region-based end-to-end network based on VGG-16 for semantic segmentation on the SIFT Flow dataset. The script automatically extracts Selective Search region proposals from the dataset. To limit the high GPU memory requirements (~6GB), the default setting uses tied weights (Sect. 3.3 of \[2\]). The network is trained with an inverse-class frequency weighted loss (Sect. 3.3 of \[2\]). The results should be around TBD!!.
- **FCN**: Run demo_fcn(). This trains an FCN-16s network based on VGG-16 for semantic segmentation on the SIFT Flow dataset. The performance varies a bit compared to the implementation of [3], as they first train FCN-32s and use it to finetune FCN-16s. Instead we directly train FCN-16s. The results should be around pixelAcc: 83.8%, meanAcc: 48.8%, meanIU: 36.7%. For weakly and semi supervised training \[4,5\], see the options in fcnTrainGeneric().
- **Fast R-CNN**: Run demo_fastrcnn_detection(). This trains Fast R-CNN using VGG-16 for object detection on PASCAL VOC 2010. The parametrization of the regressed bounding boxes is slightly simplified, but we found this to make no difference in performance. It achieves 63.5% mAP on the validation set using no external training data.

**References:**
- \[1\] http://arxiv.org/abs/1504.08083
- \[2\] http://arxiv.org/abs/1607.07671
- \[3\] http://arxiv.org/abs/1411.4038
- \[4\] http://arxiv.org/abs/1412.7144
- \[5\] http://arxiv.org/abs/1506.02106

**Disclaimer:**
Except for \[2\], none of the methods implemented in MatConvNet-Calvin is authorized by the original authors. These are (possibly simplified) reimplementations of parts of the described methods and they might vary in terms of performance. This software is covered by the FreeBSD License. See LICENSE.MD for more details.

**Contact:**
If you run into any problems with this code, please submit a bug report on the Github site of the project. For other inquiries contact holger.caesar-at-ed.ac.uk.