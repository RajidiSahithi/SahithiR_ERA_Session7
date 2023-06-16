# Targets and Analysis of Three Models
- [MODEL_1](#model_1)
- [MODEL_2](#model_2)
- [MODEL_3](#model_3)

## MODEL_1

### TARGET
* Get the set-up right
* Set Transforms
* Set Data Loader
* Set Basic Working Code
* Set Basic Training & Test Loop
* Skeleton of CNN model is fixed as Squeeze Expand / Chirstmastree model

### RESULT
* Parameters: 18,344
* Best Training Accuracy: 99.31
* Best Test Accuracy: 98.75

### ANALYSIS
* More Number of Parameters than requirment ( less than 8000)
* A Random model is taken and the images are observed to find out which trasformation to apply on the data. (Here we can apply rotation). Rotation will be applied once the model is set.
* It is observed that the model is underfitting in the initial epochs. We cant say it as underfitting in the initial epochs training starts with random values so iraining accuracy will be less.
* Later after few epochs on it is observed that the model is overfitting ( If the performance on the validation set starts to decrease while the performance on the training set continues to improve, then the model is likely overfitting). we can observe overfitting from epoch=5
* The skeleton is set. Summary below gives information about the Skeleton of CNN.
* To further increse accuracy we shall add batch normalization and regularization, GAP to reduce number of parameters.

### SUMMARY
<pre>
---------------------------------------------------------------------------------
 **          Layer(Type)           Output Shape            Param #           **   
=================================================================================
 **           Conv2d-1            [-1, 8, 26, 26]              72            **
 **             ReLU-2            [-1, 8, 26, 26]               0            **
 **           Conv2d-3           [-1, 16, 24, 24]           1,152            **
 **             ReLU-4           [-1, 16, 24, 24]               0            **
 **        MaxPool2d-5           [-1, 16, 12, 12]               0            **
 **           Conv2d-6            [-1, 8, 12, 12]             128            **
 **             ReLU-7            [-1, 8, 12, 12]               0            **
 **           Conv2d-8           [-1, 12, 10, 10]             864            **
 **             ReLU-9           [-1, 12, 10, 10]               0            **
 **          Conv2d-10             [-1, 16, 8, 8]           1,728            **
 **            ReLU-11             [-1, 16, 8, 8]               0            **
 **          Conv2d-12             [-1, 20, 6, 6]           2,880            **
 **            ReLU-13             [-1, 20, 6, 6]               0            **
 **          Conv2d-14             [-1, 16, 1, 1]          11,520            **
 ===============================================================================
 **          Total params: 18,344
 **          Trainable params: 18,344
 **          Non-trainable params: 0
 --------------------------------------------------------------------------------
 **           Input size (MB): 0.00
 **           Forward/backward pass size (MB):0.30
 **           Params size (MB): 0.07
 **           Estimated Total Size (MB) :0.38
 ----------------------------------------------------------------------------------
</pre>

### RECEPTIVE FIELD CALCULATION TABLE

| Layer   | RF_IN | N-IN   | J_IN  |  s  |  k  | RF_OUT | N_OUT |
|--------:|-------|--------|-------|-----|-----|--------|-------|
| Conv1   |   1   |   28   |   1   |  1  |  3  |    3   |   26  |
| Conv2   |   3   |   26   |   1   |  1  |  3  |    5   |   24  |
| Maxpool |   5   |   24   |   1   |  2  |  2  |    6   |   12  |
| Conv3   |   6   |   12   |   2   |  1  |  1  |    6   |   12  |
| Conv41  |   6   |   12   |   2   |  1  |  3  |   10   |   10  |
| Conv42  |   2   |   10   |   2   |  1  |  3  |    6   |    8  |
| Conv43  |   6   |    8   |   2   |  1  |  3  |   10   |    6  |
| Conv5   |   10  |    6   |   2   |  1  |  6  |   20   |    1  |



### PLOTS and OUTPUT

![alt text](https://github.com/RajidiSahithi/SahithiR_ERA_Session7/blob/main/Images7/model1_res.png)
![alt text](https://github.com/RajidiSahithi/SahithiR_ERA_Session7/blob/main/Images7/model1_graph.png)

## MODEL_2
### TARGET
* Batch Normalization is added
* Dropout is added
* To Make the model lighter Adaptive Global average pooling and additional layers are added
* To achive the better results with data augmentation techniques and filling up the training gap (Rotation is added)

### RESULT
* Parameters: 7,496 (Achieved one requirement, Number of Parameters <  8000 )
* Best Training Accuracy: 99.25
* Best Test Accuracy: 99.37

### ANALYSIS
* Batch Normalization helps us to increase the training acuuracy
* Dropout (Regularization) reduces the difference between training and test accuracy (which in turn reduces overfitting). We are making training harder.Here I used 1 % dropout.
* Dropout and Batch Normalization cannot provide required result. Dropout reduces Training Accuracy.
* The number of parameters are reduced by using Global Average Pooling. Here I used Adaptive average pooling.
* But still Overfitting is there, so we added parameters at the last by adding few more layers.Which incresed the test accuracy.consistent test accuracy of 99.4% is not achieved
* In model1 we  observed the images and found that we can apply Randon Rotation. And after rotation it is filled with black (becuase backgroung is black)
* Still we are unable to achieve consistent test accuracy.
* Still there is overfitting
* So we must add lr scheduling


### SUMMARY
<pre>
-----------------------------------------------------------------------------------
 **          Layer(Type)           Output Shape            Param #             **   
===================================================================================
 **             Conv2d-1            [-1, 8, 26, 26]              72            **
 **        BatchNorm2d-2            [-1, 8, 26, 26]              16            **
 **               ReLU-3            [-1, 8, 26, 26]               0            **
 **            Dropout-4            [-1, 8, 26, 26]               0            **
 **             Conv2d-5           [-1, 16, 24, 24]           1,152            **
 **        BatchNorm2d-6           [-1, 16, 24, 24]              32            **
 **               ReLU-7           [-1, 16, 24, 24]               0            **
 **            Dropout-8           [-1, 16, 24, 24]               0            **
 **          MaxPool2d-9           [-1, 16, 12, 12]               0            **
 **            Conv2d-10            [-1, 8, 12, 12]             128            **
 **       BatchNorm2d-11            [-1, 8, 12, 12]              16            **
 **            Conv2d-12           [-1, 12, 10, 10]             864            **
 **       BatchNorm2d-13           [-1, 12, 10, 10]              24            **
 **              ReLU-14           [-1, 12, 10, 10]               0            **
 **           Dropout-15           [-1, 12, 10, 10]               0            **
 **            Conv2d-16             [-1, 16, 8, 8]           1,728            **
 **       BatchNorm2d-17             [-1, 16, 8, 8]              32            **
 **              ReLU-18             [-1, 16, 8, 8]               0            **
 **           Dropout-19             [-1, 16, 8, 8]               0            **
 **            Conv2d-20             [-1, 20, 6, 6]           2,880            **
 **       BatchNorm2d-21             [-1, 20, 6, 6]              40            **
 **              ReLU-22             [-1, 20, 6, 6]               0            **
 **           Dropout-23             [-1, 20, 6, 6]               0            **
 ** AdaptiveAvgPool2d-24             [-1, 20, 1, 1]               0            **
 **            Conv2d-25             [-1, 16, 1, 1]             320            **
 **       BatchNorm2d-26             [-1, 16, 1, 1]              32            **
 **              ReLU-27             [-1, 16, 1, 1]               0            **
 **           Dropout-28             [-1, 16, 1, 1]               0            **
 **            Conv2d-29             [-1, 10, 1, 1]             160            **
  ===============================================================================
 **          Total params: 7,496
 **          Trainable params: 7,496
 **          Non-trainable params: 0
 --------------------------------------------------------------------------------
 **           Input size (MB): 0.00
 **           Forward/backward pass size (MB):0.57
 **           Params size (MB): 0.03
 **           Estimated Total Size (MB) :0.60
 ----------------------------------------------------------------------------------
</pre>

### RECEPTIVE FIELD CALCULATION TABLE

| Layer   | RF_IN | N-IN   | J_IN  |  s  |  k  | RF_OUT | N_OUT |
|--------:|-------|--------|-------|-----|-----|--------|-------|
| Conv1   |   1   |   28   |   1   |  1  |  3  |    3   |   26  |
| Conv2   |   3   |   26   |   1   |  1  |  3  |    5   |   24  |
| Maxpool |   5   |   24   |   1   |  2  |  2  |    6   |   12  |
| Conv3   |   6   |   12   |   2   |  1  |  1  |    6   |   12  |
| Conv41  |   6   |   12   |   2   |  1  |  3  |   10   |   10  |
| Conv42  |   2   |   10   |   2   |  1  |  3  |    6   |    8  |
| Conv43  |   6   |    8   |   2   |  1  |  3  |   10   |    6  |
| Conv5   |   10  |    6   |   2   |  1  |  6  |   20   |    1  |



### PLOTS



## MODEL_3
### TARGET
* Adding learning rate schedulers for better results.
* To achive the desired accuracy consistantly in atlest last 2 epochs

### RESULT
* Parameters: 7,496
* Best Training Accuracy: 99.26%
* Best Test Accuracy: 99.35%

### ANALYSIS
* To reduce overfitting (flatten the test accuracy graph) Schedular is used.
* Here I used ReduceLrONplatue schedular.

### SUMMARY
<pre>
-----------------------------------------------------------------------------------
 **          Layer(Type)           Output Shape            Param #             **   
===================================================================================
 **             Conv2d-1            [-1, 8, 26, 26]              72            **
 **        BatchNorm2d-2            [-1, 8, 26, 26]              16            **
 **               ReLU-3            [-1, 8, 26, 26]               0            **
 **            Dropout-4            [-1, 8, 26, 26]               0            **
 **             Conv2d-5           [-1, 16, 24, 24]           1,152            **
 **        BatchNorm2d-6           [-1, 16, 24, 24]              32            **
 **               ReLU-7           [-1, 16, 24, 24]               0            **
 **            Dropout-8           [-1, 16, 24, 24]               0            **
 **          MaxPool2d-9           [-1, 16, 12, 12]               0            **
 **            Conv2d-10            [-1, 8, 12, 12]             128            **
 **       BatchNorm2d-11            [-1, 8, 12, 12]              16            **
 **            Conv2d-12           [-1, 12, 10, 10]             864            **
 **       BatchNorm2d-13           [-1, 12, 10, 10]              24            **
 **              ReLU-14           [-1, 12, 10, 10]               0            **
 **           Dropout-15           [-1, 12, 10, 10]               0            **
 **            Conv2d-16             [-1, 16, 8, 8]           1,728            **
 **       BatchNorm2d-17             [-1, 16, 8, 8]              32            **
 **              ReLU-18             [-1, 16, 8, 8]               0            **
 **           Dropout-19             [-1, 16, 8, 8]               0            **
 **            Conv2d-20             [-1, 20, 6, 6]           2,880            **
 **       BatchNorm2d-21             [-1, 20, 6, 6]              40            **
 **              ReLU-22             [-1, 20, 6, 6]               0            **
 **           Dropout-23             [-1, 20, 6, 6]               0            **
 ** AdaptiveAvgPool2d-24             [-1, 20, 1, 1]               0            **
 **            Conv2d-25             [-1, 16, 1, 1]             320            **
 **       BatchNorm2d-26             [-1, 16, 1, 1]              32            **
 **              ReLU-27             [-1, 16, 1, 1]               0            **
 **           Dropout-28             [-1, 16, 1, 1]               0            **
 **            Conv2d-29             [-1, 10, 1, 1]             160            **
  ===============================================================================
 **          Total params: 7,496
 **          Trainable params: 7,496
 **          Non-trainable params: 0
 --------------------------------------------------------------------------------
 **           Input size (MB): 0.00
 **           Forward/backward pass size (MB):0.57
 **           Params size (MB): 0.03
 **           Estimated Total Size (MB) :0.60
 ----------------------------------------------------------------------------------
</pre>

### RECEPTIVE FIELD CALCULATION TABLE

| Layer   | RF_IN | N-IN   | J_IN  |  s  |  k  | RF_OUT | N_OUT |
|--------:|-------|--------|-------|-----|-----|--------|-------|
| Conv1   |   1   |   28   |   1   |  1  |  3  |    3   |   26  |
| Conv2   |   3   |   26   |   1   |  1  |  3  |    5   |   24  |
| Maxpool |   5   |   24   |   1   |  2  |  2  |    6   |   12  |
| Conv3   |   6   |   12   |   2   |  1  |  1  |    6   |   12  |
| Conv41  |   6   |   12   |   2   |  1  |  3  |   10   |   10  |
| Conv42  |   2   |   10   |   2   |  1  |  3  |    6   |    8  |
| Conv43  |   6   |    8   |   2   |  1  |  3  |   10   |    6  |
| Conv5   |   10  |    6   |   2   |  1  |  6  |   20   |    1  |



### PLOTS



