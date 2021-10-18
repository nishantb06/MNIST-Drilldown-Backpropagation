# MNIST-Drilldown-Backpropagation
# EVA7 - Assignment 4
## Part 1 - Explaining BackPropagation through a simple Multi Layer Perceptron

Back Propagation is a way with which we update the model parameters by subtracting it's gradients so that it can converge towards the minima. This is done with the help of Chain rule.

Our aim at each each iteration of the training loop is to calculate  partial derivative of  the loss function with respect to each weights(parameters of the models)

![Imgur](https://imgur.com/XbCkHTN.png)

After calculating the above value we update the Weights according to the following rule. \alpha is the learning rate. It shows how fast or how slow should the update be made in the direction of the minima.

![Imgur](https://imgur.com/VFm8F0h.png)

We'll start with a following simple Multi Layer perceptron ,calculate its gradients and update the weights in the excel sheet step by step. Notations can be seen from the image itself. Excel sheet can be found in the repository or [here](https://1drv.ms/x/s!Aq4yG-GWU78QeEiREdL7iBFwHKE?e=6C8pqC)

![Imgur](https://imgur.com/tk3SxbR.png)

Lets say that you wanted to calculate derivative of E_total with respect to w5. The first thing we need to do is map out all the ways of reaching E_total from w5.

![Imgur](https://imgur.com/uBOP0Zl.png)

Therefore

![Imgur](https://imgur.com/gbBm6Am.png)

Similarly,for w6,w7,w8

![Imgur](https://imgur.com/RHqhuEk.png)

To calculate gradients for the weights in the first layer, let us first look at the gradient of E_total w.r.t out_h1 and out_h2

![Imgur](https://imgur.com/wH4Vv1b.png)

Now, to calculate gradient of w5, have a look at the following chain rule

![Imgur](https://imgur.com/95xxwuQ.png)

Finally we can calculate the values of gradients of weights in the first layer, with the help of the above derivation.

![Imgur](https://imgur.com/tqTqOMM.png)

These gradients were used to calculate the final value of loss for 200 iterations. The graphs of the loss for different learning rates are 

![Imgur](https://imgur.com/jaA5gyh.png)

---
## Part-2 MNIST Drilldown
#### Write a custom model architecture with less than 20k parameters which is able to get a validation accuracy of 99.4% with no more than 20 epochs on the MNIST Dataset
---

### File Structure
1. **dataloader.py** : Contains code for the train and test data loaders for the MNIST digits. Various augmentation can be added here as well
2. **models.py** : Contains different models I have experimented with. Last model - Net8 gives the best resluts, with a validation accuracy of 99.36 %. It is my best model because it consistently gives a accuracy greater than 99.3 % in the last few epochs and has a better accuracy in the first epoch than any other model
3. **train.py** : training function
4. **test.py** : test function :
5. **EVA7-Ass4.ipynb** : Colab notebook for training on GPU's

## Model Structure 
* Model uses two Max-Pooling layers and applies a Global Average Pooling when the channel size becomes 3x3.
* In order to divide the model into 3 blocks, I have used padding in the first 2 layers to keep the image size constant.
* 3x3 Kernels were used for Convolutions operations and 1x1  for transition blocks.
* Net2 and Net8 give best resluts of **99.35% validation accuracy**

## Training Logs

![Imgur](https://imgur.com/qnPIfcM.png)

