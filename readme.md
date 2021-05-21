# Modified MNIST

The objective is not only predict the number from the MNIST dataset , but also predict the sum of the random number(0-9) added to it.

## Input Data prep-processing

Build a combined dataset using

- torchvision MNIST
- Add random numbers between 0-9 as the second input
- 2 outputs : predicted number and sum of predicted number and random number provided as input

# Network Design

- Convolution block using the image as input

- **After the convolution add the 2nd input : random number**

- Pass the ouput and random number (input2) through linear layers

- No activation function required during addition of two numbers as it is a linear function

  ```
   The model layers are: 
  Network(
    (input1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
    (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (oneconv1): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
    (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (conv5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (conv6): Conv2d(64, 10, kernel_size=(5, 5), stride=(1, 1))
    (input2): Linear(in_features=2, out_features=5, bias=True)
    (layer1): Linear(in_features=5, out_features=5, bias=True)
    (out2): Linear(in_features=5, out_features=1, bias=True)
  )
  
  Shape of parameters in each layer is: 
  input1.weight 		 torch.Size([16, 1, 3, 3])
  input1.bias 		 torch.Size([16])
  conv1.weight 		 torch.Size([32, 16, 3, 3])
  conv1.bias 		 torch.Size([32])
  conv2.weight 		 torch.Size([64, 32, 3, 3])
  conv2.bias 		 torch.Size([64])
  oneconv1.weight 		 torch.Size([16, 64, 1, 1])
  oneconv1.bias 		 torch.Size([16])
  conv3.weight 		 torch.Size([32, 16, 3, 3])
  conv3.bias 		 torch.Size([32])
  conv4.weight 		 torch.Size([64, 32, 3, 3])
  conv4.bias 		 torch.Size([64])
  conv5.weight 		 torch.Size([64, 64, 3, 3])
  conv5.bias 		 torch.Size([64])
  conv6.weight 		 torch.Size([10, 64, 5, 5])
  conv6.bias 		 torch.Size([10])
  input2.weight 		 torch.Size([5, 2])
  input2.bias 		 torch.Size([5])
  layer1.weight 		 torch.Size([5, 5])
  layer1.bias 		 torch.Size([5])
  out2.weight 		 torch.Size([1, 5])
  out2.bias 		 torch.Size([1])
  ```

## Training and Loss

- Number of epochs : 10

- Loss - as there are 2 components - image detection and sum - 2 loss functions are used

  - cross entropy and mean square error both with equal weights

  ```
  Epoch: 1, loss: 15730.5960277915, Classification Acc: 39.335, Addition Acc: 24.92
  Epoch: 2, loss: 1750.585687068291, Classification Acc: 95.72166666666668, Addition Acc: 88.07166666666667
  Epoch: 3, loss: 960.6274697096087, Classification Acc: 97.65, Addition Acc: 95.20166666666667
  Epoch: 4, loss: 815.7211702149361, Classification Acc: 98.13499999999999, Addition Acc: 96.22500000000001
  Epoch: 5, loss: 646.2290848720586, Classification Acc: 98.52666666666666, Addition Acc: 97.89333333333333
  Epoch: 6, loss: 484.78688704257365, Classification Acc: 98.84166666666667, Addition Acc: 98.215
  Epoch: 7, loss: 385.7182908653631, Classification Acc: 99.08166666666666, Addition Acc: 98.7
  Epoch: 8, loss: 363.83727842749795, Classification Acc: 99.14500000000001, Addition Acc: 98.895
  Epoch: 9, loss: 276.718324864225, Classification Acc: 99.33666666666666, Addition Acc: 99.28833333333333
  Epoch: 10, loss: 273.10551575793943, Classification Acc: 99.31833333333333, Addition Acc: 99.27833333333334
  Finished Training
  ```

## Evaluation

```
Accuracy of the network on the 10,000 test images:  98.44249201277955
Accuracy of the network on the 10,000 test images:  98.44249201277955
```

## Observations/ Learning

- The model is able to predict the number and the sum very well
- Equal weightages were given to both the losses - which helped
- No activation function was used during addition of two numbers as it was a linear function