import numpy as np

class NeuralNetwork:
    _input_layer_size = 1
    _hidden_layer_size = [1]
    _output_layer_size = 1
    _activation_type = 1
    _z = [] # result of weight and bias calculation
    _a = [] # result of activation function
    
    weight = []
    bias = []
    
    
    def __init__(self,atribute=[1,[1],1], activation_type=1):
        self._input_layer_size = atribute[0]
        self._hidden_layer_size = atribute[1]
        self._output_layer_size = atribute[2]

        inputs = self._input_layer_size
    
        np.random.seed(2675)
        
        # Initialize weights and biases
        for layer in self._hidden_layer_size:
            self.weight.append(np.random.randn(inputs, layer) * 0.01)
            self.bias.append(np.random.randn(1,layer))
            
            inputs = layer

        self.weight.append(np.random.randn(self._hidden_layer_size[-1], self._output_layer_size) * 0.01)
        self.bias.append(np.random.randn(1,self._output_layer_size))
        
        self._activation_type = activation_type
        
    def _loss(self, target, output):
        output = np.array(output, ndmin=2)
        target = np.array(target, ndmin=2)
        return 1/2 * np.mean((output - target)**2)
        
    def _activation(self, z):
        if self._activation_type == 1: # Sigmoid
            return 1/(1+np.exp(-z)) 
        elif self._activation_type == 2: # ReLU
            return np.maximum(0,z)
        elif self._activation_type == 3: # Tanh
            return np.tanh(z)
        
    def _derivative(self, z):
        if self._activation_type == 1:  # Sigmoid
            sigmoid = self._activation(z)
            return sigmoid * (1 - sigmoid)
        elif self._activation_type == 2:  # ReLU
            return np.where(z > 0, 1, 0)
        elif self._activation_type == 3:  # Tanh
            return 1 - np.tanh(z)**2
    
    def _softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1,keepdims=True))
        return exp/np.sum(exp, axis=1, keepdims=True)
        
        
    def _passForward(self, inputs):
        self._z = []
        self._a = []
        
        inputs = np.array(inputs, ndmin=2)
        
        for i in range(len(self._hidden_layer_size)):
            self._z.append(np.dot(inputs, self.weight[i]) + np.repeat(self.bias[i], np.size(inputs,0), 0))
            self._a.append(self._activation(self._z[i]))
            
            inputs = self._a[i]
        
        self._z.append(np.dot(inputs, self.weight[-1]) + np.repeat(self.bias[-1], np.size(inputs,0), 0))
        self._a.append(self._softmax(self._z[-1]))
        
        return self._a[-1]
            
    def _passBackward(self, inputs, outputs, target, learning_rate): #backpropagation
        #calculating the derivative of the loss function on the output layer
        inputs = np.array(inputs, ndmin=2)
        target = np.array(target, ndmin=2)
        outputs = np.array(outputs, ndmin=2)
        
        output_dz = outputs - target
        output_dw = np.dot(self._a[-2].T, output_dz) / np.size(inputs,0)
        output_db = np.sum(output_dz, axis=0, keepdims=True) / np.size(inputs,0)
        output_da = np.dot(output_dz, self.weight[-1].T)
        
        self.weight[-1] -= learning_rate * output_dw
        self.bias[-1] -= learning_rate * output_db
        
        #calculating the derivative of the loss function on the hidden layer
        for i in range(-3,-len(self._hidden_layer_size)-2,-1):
            output_dz = output_da*self._derivative(self._z[i+1])
            output_dw = np.dot(self._a[i].T, output_dz) / np.size(inputs,0)
            output_db = np.sum(output_dz,axis=0,keepdims=True) / np.size(inputs,0)
            output_da = np.dot(output_dz, self.weight[i+1].T)
            
            self.weight[i+1] -= learning_rate * output_dw
            self.bias[i+1] -= learning_rate * output_db
        
        #calculating the derivative of the loss function on the input layer
        output_dz = output_da*self._derivative(self._z[0])
        output_dw = np.dot(inputs.T, output_dz) / np.size(inputs,0)
        output_db = np.sum(output_dz, axis=0, keepdims=True) / np.size(inputs,0)
        
        self.weight[0] -= learning_rate * output_dw
        self.bias[0] -= learning_rate * output_db
            
    def train (self, inputs, target, epoch, learning_rate):
        inputs = np.array(inputs)
        target = np.array(target)
        output = np.zeros_like(target)
        
        for i in range(epoch):
            output = self._passForward(inputs)
            self._passBackward(inputs, output, target, learning_rate)
            if i % 100 == 0:
                loss = self._loss(target, self._passForward(inputs))
                print(f"Epoch {i} Loss: {loss:.6f}")
    
    def predict(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        return self._passForward(inputs)