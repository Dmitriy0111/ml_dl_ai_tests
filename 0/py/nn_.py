## imports
import numpy as np
import scipy.special
%matplotlib inline
import matplotlib.pyplot
## neural network class
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.lr = learningrate
        
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        hidden_errors = np.dot(self.who.T, output_errors)
        
        self.who += self.lr * np.dot((output_errors * final_outputs * \
        (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * \
        (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    def query(self,inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def save_coefs(self, path2folder):
        np.savetxt(path2folder+"wih.dat",self.wih)
        np.savetxt(path2folder+"who.dat",self.who)
## network parameters
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epoch = 3
## network
nn_mnist = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
## path to files with training and testing datas
path2train = ""
path2test = ""
## training network
for ep in range(epoch):
    training_data_file = open(path2train+"mnist_train.csv", 'r') 
    training_data_list = training_data_file.readlines () 
    training_data_file.close()

    for record in training_data_list:
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        targets = np.zeros(output_nodes) + 0.01

        targets[int(all_values[0])] =0.99
        nn_mnist.train(inputs, targets) 
        pass

    test_data_file = open(path2test+"mnist_test.csv", 'r') 
    test_data_list = test_data_file.readlines () 
    test_data_file.close()
## testing network
scorecard = []
for record in test_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0]) 

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = nn_mnist.query(inputs)

    label = np.argmax(outputs)

    if (label == correct_label) :
        scorecard.append(1)
    else:
        scorecard.append(0) 
        pass
    pass
## printing results
scorecard_array = np.asarray(scorecard)
print ("Total score = ", scorecard_array.sum() / scorecard_array.size)
