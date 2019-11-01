# Face-Similarity-On-CelebA

#### Implementation of face similarity model. Neural Network is trained on celebA Dataset to generate face embedding.
* ## Dataset
We have used [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The dataset consists:
- **10,177 number of identities** 
- **202,599 number of face images**

* PreProcessed data and Model
- Download preprocessed data and model trained on xception network with triplet loss from [here](https://drive.google.com/open?id=1f_6dgDXyWPQbyKbCuIsVoQTEGm2PHktu).

1. **Model1**  	Dlib + CNN + Softmax
Face extraction has been carried out through Dlib. Training of the model is 4-layer Convolution network. It is not showing the good result. It require very deep neural network.

2. **Model2**  	Haarcascade + Xception Network + Triplet Loss
Face extraction has been carried out through haarcascade. Model has been trained on Xception Network with Triplet Loss.
