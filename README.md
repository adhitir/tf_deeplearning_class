# CPSC 8810: Deep Learning

This repository contains the code base for the CPSC8810: Deep Learning course homeworks. 

### Author

Adhiti Raman  
Ph.D. Candidate  
Department of Automotive Engineering  
Clemson University  
Greenville, ICAR campus  

## Folder Structre

```
Homework_1:
    Part_1:
	HW1_Part1_DNN_func.py
	HW1_Part1_CNN_MNIST.py
    Part2:
	HW1_Part2_DNN_MNIST_PCA.py
	HW1_Part2_DNN_func_pnorm.py
	HW1_Part2_DNN_MNIST_pnorm.py
    Part3:
	HW1_Part3_DNN_MNIST_randomlabels.py
	HW1_Part3_No_of_params_vs_generalization.py	
	HW1_Part3_DNN_MNIST_flattness_vs_generatlization_part1.py
```

## Code Breakdown

#### Part 1

* HW1_Part1_DNN_func.py: This code deploys 3 DNN models on two functions. The code must be run for each function separately by commenting one out.

```
#y1 = np.cos(x1*x1) + x1 + np.random.normal(0, 0.1, size=x1.shape)
y1 = np.sin(2*x1) + x1 + np.random.normal(0, 0.1, size=x1.shape)
```

* HW1_Part1_CNN_MNIST.py: This code deploys 2 CNN and 1 DNN models on the MNIST dataset and plots loss and accuracy.

#### Part 2

* HW1_Part2_DNN_MNIST_PCA.py: This code deploys a DNN model on the MNIST dataset. It collects the weights for the entire model and just the first layer for every 3 epochs and plots the PCA of the same. 

* HW1_Part2_DNN_func_pnorm.py: This code deploys a DNN on a function and plots the gradient norm and loss.

* HW1_Part2_DNN_MNIST_pnorm.py: This code deploys a DNN on the MNIST dataset and plots the gradient norm and loss.

#### Part 3

* HW1_Part3_DNN_MNIST_randomlabels.py: This code trains a DNN model by randomly changing the values of labels assigned to the training batch. 

* HW1_Part3_No_of_params_vs_generalization.py: This code trains a DNN model ten times, each time by assigning an increasing number of nodes to the layers. 

* HW1_Part3_DNN_MNIST_flattness_vs_generatlization_part1.py: This code records the interpolation between the weights of two models and evaluates a third model designed on the interpolated weights. 

* HW1_Part3_DNN_MNIST_flattness_vs_generatlization_part2.py: This code records the loss, accuracy and sensitivity of 5 models -  a mix of CNNs and DNNs on the MNIST database. 


## Built With

* Tensorflow 1.15
*  JupyterHub - Python 3


