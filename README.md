# Classify-Cifar-100

## Abstract
Image classification is the objective of identifying whether the image belongs to a particular predefined class or not. It's of vital importance in this era. latest advances in that field enabled self driving cars to be a reality, doctors to detect deseases, and people with disabilities to have a better life. Image classification can be extended to Image detection - detect whether something exists in an image - , and image recognition - which is the ability of software to identify objects, places, people, writing and actions in images. In this paper we try to tackle the challange by providing two solutions to classify 100 classes in a large data set. we explain our methods, show the results, and discuss how we acheived them.

## Problem Definition
It's required to classify 100 classes of objects found in images of the CIFAR-100 dataset, which contains 50000 training example and 10000 test samples. Images are of size 32x32x3 RGB colored 3-channels images.

## Solution
In this paper we propose to solutions to the classification problem:
- Multi-Class SVM: in the form of a shallow neural network with the number of hidden units (in the hidden layer) equal to the number of classes and using the *catagorical_hinge_loss* as the loss function of the model. The model acheives relatively poor performance which we try to enhance in the next proposed model.
- Deep Neurak Network: we introduce a deep neural network withseveral convoluntional layers  to extract and capture the many features of the images, which resulted in a better performance.

### Programming Tools and APIs


- This Notebook is developed totally in **Python**, we used keras API of the **tensorflow 2.0** framework for building deeplearning models.
- we used the **Eager mode of tensorflow** via its sequential API for its simpilicity.
- **Numpy** for calculations
- **Pandas** for data preprocessing
- **matplotlib** for plotting figures
- **scikit-learn** for feature extraction
