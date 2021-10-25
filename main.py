import numpy as np
import gzip
import matplotlib.pyplot as plt
import random
import pdb

random.seed(123)
np.random.seed(123)

#network component defining
class AffineComponent:
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = 0.01 * np.random.randn(input_dim, output_dim)
        self.bias = 0.01 * np.random.randn(1,output_dim)

    def propagate(self,input_data):
        assert input_data.shape[1] == self.input_dim
        self.input_data = input_data
        return input_data.dot(self.weights) + self.bias

    def back_propagate(self,derivative):
        assert derivative.shape[1] == self.output_dim
        propagate_derivative = derivative.dot(self.weights.T)
        self.__update(derivative)
        return propagate_derivative

    def __update(self,derivative):
        self.bias -= LR * derivative.sum(axis = 0, keepdims= True)
        self.weights -= LR * self.input_data.T.dot(derivative)


class NolinearComponent:
    def __init__(self,dim,nolinear_type):
        self.dim = dim
        self.nolinear_type = nolinear_type

    def propagate(self,input_data):
        assert input_data.shape[1] == self.dim
        self.input_data = input_data
        if(self.nolinear_type == "relu"):
            return self.__relu(input_data)
        else:
            #program is not expected to reach here
            assert False

    def __relu(self,input_data):
        #important! must use copy or the input data will be change through index
        output_data = input_data.copy()
        output_data[output_data < 0] = 0
        return output_data
    #----------------------------
    def back_propagate(self,derivative):
        assert derivative.shape[1] == self.dim
        if(self.nolinear_type == "relu"):
            return self.__back_relu(derivative)
        else:
            #program is not expected to reach here
            assert False
            
    def __back_relu(self,derivative):
        derivative[self.input_data < 0] = 0
        return derivative

def network_propagate(input_data):
    activate = dnn1_affine.propagate(input_data)
    activate = dnn1_relu.propagate(activate)
    activate = dnn2_affine.propagate(activate)
    activate = dnn2_relu.propagate(activate)
    activate = dnn3_affine.propagate(activate)
    return output.propagate(activate)

def network_backpropagate(probs,batch_label):
    derivative = output.back_propagate(probs,batch_label)
    derivative = dnn3_affine.back_propagate(derivative)
    derivative = dnn2_relu.back_propagate(derivative)
    derivative = dnn2_affine.back_propagate(derivative)
    derivative = dnn1_relu.back_propagate(derivative)
    derivative = dnn1_affine.back_propagate(derivative)

def caculate_loss(probs, batch_label):
    batch_size = probs.shape[0]
    loss_list = -np.log(probs[range(batch_size), batch_label])
    
    average_loss = loss_list.mean(axis=0)
    return average_loss

class SoftmaxOutputComponent:
    def __init__(self, dim):
        self.dim = dim

    def propagate(self,input_data):
        assert input_data.shape[1] == self.dim
        self.input_data = input_data
        e_x = np.exp(input_data)
        return e_x / e_x.sum(axis=1, keepdims=True)

    def back_propagate(self, probs, label):
        assert probs.shape[0] == label.shape[0]
        batch_size = probs.shape[0]
        delta = probs
        delta[range(batch_size), batch_labels] -= 1
        return delta / batch_size

def load_data(file_name, image_size, num_images):
    f = gzip.open(file_name, 'r')
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size**2)
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    for i in range(0, data.shape[0], 1):
        data[i] = (data[i]-min_data[i])/(max_data[i]-min_data[i])
    return data

def load_labels(file_name, num_images):
    f = gzip.open(file_name, 'r')
    f.read(8)  
    buf = f.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    '''
    labels_onehot = np.zeros((labels.shape[0], 10))
    for i in range(0, labels.shape[0], 1):
        labels_onehot[i][labels[i]] = 1
    '''
    return labels

def cal_acc(preds, labels):
    correct_cnt = 0
    for i in range(0, len(preds), 1):
        if preds[i] == labels[i]:
            correct_cnt += 1
    return round(correct_cnt*100/len(preds), 2)
    

if __name__ == '__main__': 
    BATCH_SIZE = 20
    EPOCH = 30
    LR = 0.01
    
    train_data = load_data('./MNIST/train-images-idx3-ubyte.gz', 28, 60000)
    train_labels = load_labels('./MNIST/train-labels-idx1-ubyte.gz', 60000)
    
    
    val_indices = random.sample(range(len(train_labels)), int(60000*0.3)) # get 18000 random indices
    val_data = train_data[val_indices]
    val_labels = train_labels[val_indices]
    
    train_data = np.delete(train_data, val_indices, axis=0)
    train_labels = np.delete(train_labels, val_indices, axis=0)
    
    test_data = load_data('./MNIST/t10k-images-idx3-ubyte.gz', 28, 10000)
    test_labels = load_labels('./MNIST/t10k-labels-idx1-ubyte.gz', 10000)
    
    #network defining
    dnn1_affine = AffineComponent(28*28, 100)
    dnn1_relu = NolinearComponent(100, "relu")
    dnn2_affine = AffineComponent(100, 20)
    dnn2_relu = NolinearComponent(20, "relu")
    dnn3_affine = AffineComponent(20, 10)
    output = SoftmaxOutputComponent(10)
    
    for i in range(0, EPOCH, 1):
        epoch_loss = 0
        
        # training
        for train_index in range(0, train_data.shape[0], BATCH_SIZE):
            batch_data = train_data[train_index:(train_index+BATCH_SIZE)]
            batch_labels = train_labels[train_index:(train_index+BATCH_SIZE)]
            
            probs = network_propagate(batch_data)
            loss = caculate_loss(probs, batch_labels)
            epoch_loss += loss
            network_backpropagate(probs, batch_labels)
        
        # valiation
        preds = []
        for val_index in range(0, val_data.shape[0], BATCH_SIZE):
            batch_data = val_data[val_index:(val_index+BATCH_SIZE)]
            batch_labels = val_labels[val_index:(val_index+BATCH_SIZE)]
            
            probs = network_propagate(batch_data)
            preds_batch = np.argmax(probs, axis=1).tolist()
            preds += preds_batch
        val_acc = cal_acc(preds, val_labels.tolist())
        
        print("EPOCH:", i, "train_loss:", epoch_loss/train_data.shape[0], "val_acc:", val_acc)