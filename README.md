# Supervised Learning Neural Network
Customizable input, hidden layers, and output layer for a Python neural network.

## Description
This program allows you to create a supervised neural network simply by specifying the desired layers.

## Features

- custom input, output, and hidden layer.
- 3 option of activation function.
- custom epoch and learning step.
- accessable weight and bias to save.

## Requirements

- Python 3
- Numpy

## Installation

Clone the repository:
  ```sh
  git clone https://github.com/Machive5/VirusBreakout.git
  ```

## Usage

### 1. you can import initiate the class firs:
```py
    from neural_network import NeuralNetwork

    nn = NeuralNetwork([9,[6,4],3],1)
```

When initiate the class you need to pass 2 parameter.

The first parameters are indicate the layer of the neural network. 

- First index is input layer. The exaple means there is a input layer with 9 input. 

- Seccond index must be wraped on an array. This index indicated the hi with 3 output
dden layer you want. The example means that there is 2 hidden layer with 6 neourons on first layer and 4 neurons on second layer.

- Third index is output layer. The example means there is an output layer

### 2. Train Data
You can train your data using ```train``` method.

```py
nn.train(train_data, target, 20000, 0.5)
```

There is 4 parameter you need to pass.
- First parameter is your datasets
- Seccond parameter are the targets you want to achieve for each test
- Third parameter is how much epoch you want to learn that model
- Fourth parameter is learning steps

### 3. Test the model
After training, you can test your model by give it the data you want to predict

```py
nn.predict(test_data)
```

That function need a input you want to predict.
The result will be array of output between 0 to 1. 

```py
[0.00, 0.98, 0.00]
``` 
###  4.  Accessing the weight and bias
To accessing weight and bias you just need to call the method

```py
nn.weight
nn.bias
```

## License
This project is licensed under the GPL-3.0 License.