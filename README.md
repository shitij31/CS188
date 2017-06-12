## Holistically-Nested Edge Detection

Created by Saining Xie at UC San Diego. Modified by Shitij Gupta and Ashvin Vinodh for CS188 (Computational Methods for Medical Imaging) Final Project.

### Introduction:

This is a new edge detection algorithm, holistically-nested edge detection (HED), which performs image-to-image prediction by means of a deep learning model that leverages fully convolutional neural networks and deeply-supervised nets.  HED automatically learns rich hierarchical representations (guided by deep supervision on side responses) that are important in order to resolve the challenging ambiguity in edge and object boundary detection.

We use this algorithm to perform cerebral blood vessel segmentation in 2D DSA images. We make use of the pretrained model which gives ODS=.790 and OIS=.808 result on BSDS benchmark dataset. This model can be found at CS188/examples/hed/hed_pretrained_bsds.caffemodel.

### Installing 

 0. Install prerequisites for Caffe (http://caffe.berkeleyvision.org/installation.html#prequequisites). We also found https://gist.github.com/kylemcdonald/0698c7749e483cd43a0e guide to be useful. We only used the CPU (no GPU mode) for our testing so we didn't require CUDA. Also, we used OpenCV 2.4.13. OpenCV 3 tends to give some unexpected errors.
 1. Clone this repository
 2. Update the Makefile.config to make sure PYTHON_INCLUDE, PYTHON_LIB, BIAS_INCLUDE etc point to the right paths and versions (eg. python version may not be the same in your system). We use openblas.
 3. Run the following commands: 
      make clean
      make all
      make pycaffe
 4. Set PYTHONPATH to the right version of python found at CS188/python by running: export PYTHONPATH=<path to CS188/python>
 5. Make sure OpenCV can be found at CS188/python. One way is to symlink the cv.py and cv2.so files as shown here: http://www.mobileway.net/2015/02/14/install-opencv-for-python-on-mac-os-x/

### Training HED

We couldn't train the model with our DSA dataset as we didn't have the computing resources. If you would like to train the model for your purpose, the training procedure is listed on the HED repo (https://github.com/s9xie/hed)

### Testing HED

Code for reproducing our outputs on cerebral DSA datasets can be found at CS188/examples/hed/HED-tutorial.ipynb. Use jupyter to access the ipython notebook. In this code we pre-process the input image (specified using data_root and imname) by equalizing histograms and denoising. We also perform some pre-learning processing, wherein the image intensity values are normalized for better learning by the model. Subsequently, 6 output images are produced - 5 side-outputs and 1 fused - each 1024px by 1024px. You can choose where you want the output images to be saved by modifying savefig function's parameters. 

### Acknowledgment:

This code is based on Caffe. Thanks to the contributors of Caffe. Thanks to the authors of HED algorithm, Saining Xie and Zhuowen Tu.

@InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
      Year  = {2015},
    }

### Additional resources

If you have further issues setting up caffe or reproducing our results, please check out the following resources:
Caffe installation guide for Mac: https://gist.github.com/kylemcdonald/0698c7749e483cd43a0e
HED github repo: https://github.com/s9xie/hed
Original published paper: http://arxiv.org/abs/1504.06375