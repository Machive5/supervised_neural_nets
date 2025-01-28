import numpy as np
from neural_network import NeuralNetwork

train_data = [[
    1,0,0,
    0,1,0,
    0,0,1,
],[
    0,0,1,
    0,1,0,
    1,0,0,
],[
    1,0,0,
    1,0,0,
    1,0,0,
],[
    0,1,0,
    0,1,0,
    0,1,0,
],[
    0,0,1,
    0,0,1,
    0,0,1,
],[
    1,1,1,
    0,0,0,
    0,0,0,
],[
    0,0,0,
    1,1,1,
    0,0,0,
],[
    0,0,0,
    0,0,0,
    1,1,1,
]]

target = [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1]]

nn = NeuralNetwork([9,[6],3],1)

nn.train(train_data,target, 20000, 0.1)

test_data = [[0,1,0,
              0,1,0,
              0,1,0],  
               
             [0,0,0,
              1,1,1,
              0,0,0],
             
             [1,0,0,
              0,1,0,
              0,0,1]]

predict = nn.predict(test_data)

print("\n=============================================\n")

np.set_printoptions(precision=3, suppress=True)

for i in range(len(predict)):
    print(np.reshape(test_data[i],(3,3)), '\n')
    
    if np.argmax(predict[i]) == 0:
        print("diagonal")
    elif np.argmax(predict[i]) == 1:
        print("vertikal")
    elif np.argmax(predict[i]) == 2:
        print("horizontal")
    
    print("=========================================================")