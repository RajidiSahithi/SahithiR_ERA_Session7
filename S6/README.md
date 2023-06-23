# PART 1
## What is BackPropagation?
Backpropagation is an algorithm that backpropagates the errors from the output nodes to the input nodes. Therefore, it is simply referred to as the backward propagation of errors.

## Need for Backpropagation:
Backpropagation is “backpropagation of errors” and is very useful for training neural networks. It’s fast, easy to implement, and simple. Backpropagation does not require any parameters to be set, except the number of inputs. Backpropagation is a flexible method because no prior knowledge of the network is required.
Here are some of the advantages of the backpropagation algorithm:

* It’s memory-efficient in calculating the derivatives, as it uses less memory compared to other optimization algorithms, like the genetic algorithm. This is a very important feature, especially with large networks.
* The backpropagation algorithm is fast, especially for small and medium-sized networks. As more layers and neurons are added, it starts to get slower as more derivatives are calculated. 
* This algorithm is generic enough to work with different network architectures, like convolutional neural networks, generative adversarial networks, fully-connected networks, and more.
* There are no parameters to tune the backpropagation algorithm, so there’s less overhead. The only parameters in the process are related to the gradient descent algorithm, like learning rate.

### Example to understand Backpropagation (Feedforward network with one hidden layer and sigmoid loss)
Let us see the simple Neural Network shown below

![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/Simple_NN.png)  

This Neural Network is having a input layer, hidden layer and output layer.

This figure shows an example of a fully-connected artificial neural network (FCANN), the simplest type of network for demonstrating how the backpropagation algorithm works. The network has an input layer, 1 hidden layers, and an output layer. In the figure, the network architecture is presented horizontally so that each layer is represented vertically from left to right. 

Each layer consists of 1 or more neurons represented by circles. Because the network type is fully-connected, then each neuron in layer i is connected with all neurons in layer i+1.
In the above image 
<pre>
-i1,i2 are the inputs
-w1,w2,w3,w4 are weights of the input layer
-w5,w6,w7,w8 are weights from hidden layer 
-t1,t2 are desired or target values
</pre>
For each connection, there is an associated weight. The weight is a floating-point number that measures the importance of the connection between 2 neurons. The higher the weight, the more important the connection. The weights are the learnable parameter by which the network makes a prediction. If the weights are good, then the network makes accurate predictions with less error. Otherwise, the weight should be updated to reduce the error.
<br/> Assume that a neuron i1 at input layer  is connected to another neuron at hidden layer. Assume also that the value of h1 is calculated according to the next linear equation.
<pre>
h1,h2 are outputs of hidden layer. These are  the sum of products (SOP) between each input and its corresponding weight:
h1 = w1*i1 + w2*i2   
h2 = w3*i1 + w4*i2
</pre>
Each neuron in the hidden layer uses an activation function like sigmoid. The neurons in the output layer also use activation functions like sigmoid (for regression).
<pre>
-a_h1,a_h2 are sigmoid function (activation function) of h1,h2 respectively
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2) = 1/(1 + exp(-h2))
</pre>
Similarly for Output Layer we have
<pre>
-o1 = w5*a_h1 + w6*a_h2
-o2 = w7*a_h1 + w8*a_h2
-a_o1,a_o2 are are sigmoid function (activation function) of o1,o2 respectively
a_o1 = σ(o1) = 1/(1 + exp(-o1))
a_o2 = σ(o2) = 1/(1 + exp(-o2))
</pre>
To train a neural network, there are 2 passes (phases):
* Forward
* Backward
<br/>
In the forward pass, we start by propagating the data inputs to the input layer, go through the hidden layer(s), measure the network’s predictions from the output layer, and finally calculate the network error based on the predictions the network made. 
<br/> This network error measures how far the network is from making the correct prediction. For example, if the correct output is 0.5 and the network’s prediction is 0.3, then the absolute error of the network is 0.5-0.3=0.2. Note that the process of propagating the inputs from the input layer to the output layer is called forward propagation. Once the network error is calculated, then the forward propagation phase has ended, and backward pass starts.
<br/> The following formulas represent Errors with respect to targets.
<pre>
E1,E2 are the error with respect to target values t1,t2
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²
-Total error E_Total = E1 + E2
</pre>
<br/>In the backward pass, the flow is reversed so that we start by propagating the error to the output layer until reaching the input layer passing through the hidden layer(s). The process of propagating the network error from the output layer to the input layer is called backward propagation, or simple backpropagation. The backpropagation algorithm is the set of steps used to update network weights to reduce the network error.

The forward and backward phases are repeated from some epochs. In each epoch, the following occurs:
* The inputs are propagated from the input to the output layer.
* The network error is calculated.
* The error is propagated from the output layer to the input layer.

Calculating gradients with the chain rule
Since a neural network has many layers, the derivative of C at a point in the middle of the network may be very far removed from the loss function, which is calculated after the last layer.
<br/> The output of the activation function from the output neuron reflects the predicted output of the sample. It’s obvious that there’s a difference between the desired and expected output.
<br/> Knowing that there’s an error, what should we do? We should minimize it. To minimize network error, we must change something in the network. Remember that the only parameters we can change are the weights and biases. We can try different weights and biases, and then test our network.

<br/> We calculate the error, then the forward pass ends, and we should start the backward pass to calculate the derivatives and update the parameters.

To practically feel the importance of the backpropagation algorithm, let’s try to update the parameters directly without using this algorithm.

<br/> To calculate the derivative of the error W.R.T the weights, simply multiply all the derivatives in the chain from the error to each weight,

#### Calculating Total Loss (E_Total) Gradient with respect to weights (w5,w6,w7,w8)
<pre>
∂E_total/∂w5 = ∂(E1 + E2)/∂w5
        E2 is independent of w5 so  ∂E_total/∂w5 = ∂E1/∂w5
        ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
        Let’s calculate partial derivatives of each part of the chain we created.
            we have
                ∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)
                ∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)
                ∂o1/∂w5 = a_h1
        Therefore ∂E1/∂w5 =  (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
 
 ∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1 
 ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
 ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
 ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
 </pre>
 #### Calculating Total Loss (E_Total) Gradient with respect to ----- (h1,h2)
 <pre> 
 ∂E_total/∂a_h1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) 
 Similarly
 ∂E_total/∂a_h2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8)
 </pre>
#### Calculating Total Loss (E_Total) Gradient with respect to weights (w1,w2,w3,w3) 
<pre>
∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1 
         we have
                ∂E_total/∂a_h1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) 
                ∂a_h1/∂h1 = a_h1 * (1 - a_h1) 
                ∂h1/∂w1 = i1
        Therefore ∂E_total/∂w1 =  ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
 
 ∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
 ∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
 ∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
 ∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2
 </pre>  
##### Calculation of New Parameters
New parameters are calculated by using the formulas
<pre>
  New weight(w1) = Old weight(w1) - ƞ * ∂E_total/∂w1
  The same formula for remaining weights (w2,w3,w4,w5,w6,w7,w8)
</pre>
Based on the new parameters, we will recalculate the predicted output. The new predicted output is used to calculate the new network error. The network parameters are updated according to the calculated error. The process continues to update the parameters and recalculate the predicted output until it reaches an acceptable value for the error.
<br/> One important operation used in the backward pass is to calculate derivatives. Before getting into the calculations of derivatives in the backward pass, we can start with a simple example to make things easier.

##### Learning Rate: 
Also, the learning rate doesn’t have to have a fixed value. Learning rate will decrease as epochs for training increase. Besides that, some adaptive learning rate optimization methods modify the learning rate during the training.

 ###### Calculating the backpropagation for target values t1=0.5,t2=0.5,i1=0.05,i2=0.1,w1=0.15,w2=0.2,w3=0.25,w4=0.3,w5=0.4,w6=0.45,w7=0.5,w8=0.55 with learning rate = 1
 
 <br/> After calculating the individual derivatives in all chains, we can multiply all of them to calculate the desired derivatives (i.e. derivative of the error W.R.T each weight). we get the following values as mentioned in Screenshot below.
 
 ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/excel_screenshot.png)  
 
 Following is the graph from the above excel sheet
 
 ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr1.png)  
 
 <pre>             
                  *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        
 </pre>
###### ERROR GRAPH WITH LEARNING RATE=0.1 
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr0.1.png)  
 <pre>             
                 *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        
                 </pre>
###### ERROR GRAPH WITH LEARNING RATE=0.2
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr0.2.png) 
 <pre>             
                      *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#       
                      </pre>
###### ERROR GRAPH WITH LEARNING RATE=0.5
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr0.5.png) 
  
 <pre>             
                    *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        
                    </pre>
###### ERROR GRAPH WITH LEARNING RATE=1.0
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr1.0.png)   
 <pre>             
                   *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#       
                   </pre>
###### ERROR GRAPH WITH LEARNING RATE=2.0
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr2.0.png)   

# PART 2
Points discussed in Last 5 Lectures:
### How many layers
In the code related to the assignment is having 29 layers as shown in image below

 ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images2/layer.png)

Layers used here(in assignment) are 3X3 Convolutional layers , 1X1 Convolutional Layer, Maxpooling Layers and Gloabl Average Pooling (GAP) Layer.
<br/>
Neural networks accept an input image/feature vector (one input node for each entry) and transform it through a series of hidden layers, commonly using nonlinear activation functions. Each hidden layer is also made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer. The last layer of a neural network (i.e., the “output layer”) is also fully connected and represents the final output classifications of the network.
<br/> The number of layers in a Neural Network depends on Receptive Field. We add the layers untill the Receptive Field is equal to the size of the image.A Neural Network can have thousands of such layers.
#### Why do we add layers:
We expect that our first layers would be able to extract simple features like edges and gradients. The next layers would then build slightly complex features like textures, and patterns. Then later layers could build parts of objects, which can then be combined into objects. 
<br/> We generally use Maxpooling with Stride 1 and Filter or Kernal Size of 3X3
### MaxPooling
In the assignment Maxpooling is used 3 times at the end of each block with a stride of 2 and Kernal/Filter Size 2X2
<br/> It Calculate the maximum value for each patch of the feature map.
<br/>The main idea behind a pooling layer is to “accumulate” features from maps generated by convolving a filter(kernal) over an image. It progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network.
<br/> By using Maxpooling we can reduce the number of layers (by increasing Receptive Field), which inturn reduces number of parameters
<br/> We generally use Maxpooling with Stride 2 and Filter or Kernal Size of 2X2.

### 1x1 Convolutions
In the assignment code 1 x 1 convolution layer in tandem with global average pooling instead of linear layers for producing a 10 element vector representation without regularization. Here 1X1 is used to minimize the number or parameters and to equalize the number of channels to the number of classes.(before performing GAP)
<br/> A 1x1 convolution is a process of performing a convolution operation using a filter with just one row and one column. It is used in some convolutional neural networks for dimensionality reduction, efficient low dimensional embeddings, and applying non-linearity after convolutions.
<br/>A problem with deep convolutional neural networks is that the number of feature maps often increases with the depth of the network. This problem can result in a dramatic increase in the number of parameters and computation required when larger filter sizes are used, such as 5×5 and 7×7.To address this problem, a 1×1 convolutional layer can be used that offers a channel-wise pooling, often called feature map pooling or a projection layer. This simple technique can be used for dimensionality reduction, decreasing the number of feature maps whilst retaining their salient features. It can also be used directly to create a one-to-one projection of the feature maps to pool features across channels or to increase the number of feature maps, such as after traditional pooling layers.
##### Advantage of 1X1 Convolutions provides following features:
* 1x1 is computation less expensive.
* 1x1 is not even a proper convolution, as we can, instead of convolving each pixel separately, multiply the whole channel with just 1 number
* 1x1 is merging the pre-existing feature extractors, creating new ones, keeping in mind that those features are found together (like edges/gradients which make up an eye)
* 1x1 is performing a weighted sum of the channels, so it can so happen that it decides not to pick a particular feature that defines the background and not a part of the object.
### 3x3 Convolutions
A 3x3 convolution is a process of performing a convolution operation using a filter with  3 rows and 3 columns.
<br/> The Receptive field of last layer in the assignment is 34.
<br/> In the code realted to the assignment I have used six 3X3 Convolutional layers.
### Receptive Field
In the assignment code the Receptive Field is 34
<br/>The receptive field in a Deep Neural Network (DNN) refers to the portion of the input space that a particular neuron is "sensitive" to, or in other words, the region of the input that influences the neuron's output.
<br/>In Convolutional Neural Networks (CNNs), the receptive field is determined by the spatial extent of the filters and the strides of the convolutional operations. Larger receptive fields lead to neurons that have a wider field of view, allowing them to capture more complex relationships in the input data.
<pre>
The formula for calculating receptive field is r_out = rin + (k - 1) * jin
jout = jin * s
r_out - Output Receptive Field
r_in - Input Receptive Field
k - kernal or Filter size
jin - jump
</pre>
### SoftMax
The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1.

### Learning Rate
In the assignment code the learning rate is set to 0.01
<br/> The amount that the weights are updated during training is referred to as the step size or the “learning rate.”
<br/> Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.
<br/> Higher the learning rate the Neural Network starts remembering (memorizing) the we may not get good results for diffrent set of data.

### Kernels and how do we decide the number of kernels
In computer vision we often convolve an image with a kernel/filter to transform an image or search for something.A kernel or convolutional matrix  is used for blurring, sharpening, edge detection, and other image processing functions.
<br/> Krenals hold fixed values (for a aprticular dectection(horizontal lines, vertical lines etc)
<br/> Mostly kernal of size 3X3 is preferred for Convolutional Layers, 2X2 for Maxpooling.
<br/> Number of kernels are not arbitrary. They can be chosen either intuitively or empirically. Depend on the task, number of kernels in each layer can change significantly. The more complex the dataset you expect networks with more kernels perform better. Intuitively, number of kernel at layer are expected to bigger in the previous layers, as number of possible combination grow. That is why, in general, first layer kernels are less than mid- high-level ones.

### Batch Normalization
Batch normalization (also known as batch norm) is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling. 
<br/> We dont know at which layer we need to apply batch normalization. So, it is applied at the end of each layer.
<br/> Batch Normalization will add Regularization which addresses Overfitting problem.

### Image Normalization
Image normalization is a process that changes the range of pixel intensity values. The normalization is done in the program by using the statement transforms.Normalize((0.1307,), (0.3081,)
<br/> Normalize in pytorch context subtracts from each instance (MNIST image in your case) the mean (the first number) and divides by the standard deviation (second number). This takes place for each channel separately, meaning in mnist you only need 2 numbers because images are grayscale

### Position of MaxPooling
Maxpooling is placed at the end of each block.
<br/>The assignment code consists of 3 blocks having two convolutional layers and maxpooling.
<br/> Last Two blocks are 1X1 convolution followed by Global Average Pooling Layer (GAP)

### Concept of Transition Layers
Transition layer which is the combination of [convolution + pooling] which is just a way of downsampling the representations calculated by dense blocks to the end as we move from 512x512 to 256x256 to 128x128 and so on. So in simple words decision on reducing/ increasing mathematical complexity of model happens in transition layers.
<br/>The assignment code consists of 3 transition blocks having two convolutional layers and maxpooling.

### Position of Transition Layer
Transition Layers are placed at the starting of CNN

### DropOut
Deep learning neural networks are likely to quickly overfit a training dataset with few examples.

<br/>Ensembles of neural networks with different model configurations are known to reduce overfitting, but require the additional computational expense of training and maintaining multiple models.

<vr/>A single model can be used to simulate having a large number of different network architectures by randomly dropping out nodes during training. This is called dropout and offers a very computationally cheap and remarkably effective regularization method to reduce overfitting and improve generalization error in deep neural networks of all kinds.

### When do we introduce DropOut, or when do we know we have some overfitting
Dropout is implemented per-layer in a neural network.

<br/>It can be used with most types of layers, such as dense fully connected layers, convolutional layers, and recurrent layers such as the long short-term memory network layer.

<br/>Dropout may be implemented on any or all hidden layers in the network as well as the visible or input layer. It is not used on the output layer.
<br/> Overfitting refers to a model that models the training data too well.
Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize.
<br/>If the Validation accuracy is pretty much lesser than Training accuracy then it is a clear case of overfitting.This can be addressed by adding dropouts.Adding too many Dropouts will lead to underfitting of the network.This can be observed by seeing the training accuracy not improving further with increase in the number of epoch.
<br/> Droput increases number of epochs

### The distance of MaxPooling from Prediction

### The distance of Batch Normalization from Prediction
Prediction and batch normalization provides complementary benefits to existing state-of-the-art approaches for improving robustness 

### When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
I think we can stop the process of convolution when the receptive field of the network is equal to the size of the image

### How do we know our network is not going well, comparatively, very early
* If the training loss doesnt reduce
* If the receptive Field is very less
* If the algorithm is having most variation in accuracy
* If CNN is not having enough training data

### Batch Size, and Effects of batch size
In the assignment code Batch Size is 128
Batch size defines the number of samples we use in one epoch to train a neural network.
<br/> A larger batch size means that more data can be processed in parallel, which can speed up the training and reduce the memory requirements.
<br/> Larger Batch Size will reduce the computation time and increase the amount of memory used.

### etc (you can add more if we missed it here)

 


