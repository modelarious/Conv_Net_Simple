#CIFAR-10 CITATION: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
#dataset retrieved from: http://www.cs.toronto.edu/%7Ekriz/cifar.html 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt, ceil
import numpy as np
from time import time, sleep
import sys
import shutil
import pickle
import urllib
import os 
import tarfile





def unPickle(file):
    #extracts data from a binary encoded file
    with open(file, 'rb') as fp:
        dataDict = pickle.load(fp, encoding='bytes')
    return dataDict

def file_exists(fileName):
    #checks the existence of a given file
    return os.path.isfile(fileName)

def directory_Exists(dirName):
    #checks the existence of a directory
    return os.path.exists(dirName)

def download_File(name, fileName, location):
    #retrieves file from "location" and saves it as "fileName"
    print("Downloading", name, "data")
    urllib.request.urlretrieve(location, fileName)

def get_MNIST_DATA():
    #retrieves MNIST DATA, separated as the method of retrieval is different
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    trainX, trainY = mnist.train.images, mnist.train.labels
    testX, testY = mnist.test.images, mnist.test.labels   
    return trainX, trainY, testX, testY, testX.shape[1], testY.shape[1]

def get_Cifar_100_Data(folderName, data="coarse"):
    #opens up the binary files from the archive, reads the relevant data, 
    #normalizes the input data, and returns training data, testing data,
    #number of inputs per data entry, number of output categories
    
    data = data.lower()
    if data not in ["coarse", "fine"]:
        print(data, "is not one of coarse or fine")
        sys.exit(1)
        
    if data == "coarse":
        data = b'coarse_labels'
    else:
        data = b'fine_labels'
        
    #training data
    dataDict = unPickle(folderName + "/train")
    
    trainY = np.array(dataDict[data])
    
    numLabels = np.size(np.unique(trainY))
    #one hot
    trainY = np.eye(numLabels)[trainY]
    
    
    
    #coarse = np.array(dataDict[b'coarse_labels'])
    #fine = np.array(dataDict[b'fine_labels'])
    
    #one hot
    #coarse = np.eye(20)[coarse]
    #fine = np.eye(100)[fine]
    
    #put them one beside each other, makes 120 categories with 2 firing
    #in every example
    #trainY = np.concatenate((coarse, fine), axis=1)
    
    #read data
    trainX = dataDict[b'data']
    
    #testing data
    dataDict = unPickle(folderName + "/test")
    '''
    coarse = np.array(dataDict[b'coarse_labels'])
    fine = np.array(dataDict[b'fine_labels'])
    
    #one hot
    coarse = np.eye(20)[coarse]
    fine = np.eye(100)[fine]
    
    testY = np.concatenate((coarse, fine), axis=1)
    '''
    
    testY = np.array(dataDict[data])
    
    #one hot
    testY = np.eye(numLabels)[testY]    
    #read data
    testX = dataDict[b'data']
    
    #regularization
    trainX = (trainX-np.mean(trainX))/np.std(trainX)
    testX = (testX-np.mean(testX))/np.std(testX)    
    
    return trainX, trainY, testX, testY, testX.shape[1], testY.shape[1]
    
    
def get_Cifar_10_Data(folderName):  
    #opens up the binary files from the archive, reads the relevant data, 
    #normalizes the input data, and returns training data, testing data,
    #number of inputs per data entry, number of output categories    
    
    dataDict = unPickle(folderName + "/data_batch_1")
    trainX, trainY = dataDict[b'data'],  np.array(dataDict[b'labels'])
    
    dataDict = unPickle(folderName + "/data_batch_2")
    
    trainX = np.concatenate((trainX, dataDict[b'data']))  
    trainY = np.concatenate((trainY, dataDict[b'labels']))   
    
    dataDict = unPickle(folderName + "/data_batch_3")
    trainX = np.concatenate((trainX, dataDict[b'data']))  
    trainY = np.concatenate((trainY, dataDict[b'labels']))  
    
    dataDict = unPickle(folderName + "/data_batch_4")
    trainX = np.concatenate((trainX, dataDict[b'data']))  
    trainY = np.concatenate((trainY, dataDict[b'labels']))  
    
    dataDict = unPickle(folderName + "/data_batch_5")
    trainX = np.concatenate((trainX, dataDict[b'data']))  
    trainY = np.concatenate((trainY, dataDict[b'labels']))      
    
    dataDict = unPickle(folderName + "/test_batch")
    testX, testY = dataDict[b'data'],  dataDict[b'labels']
    
    #regularization
    trainX = (trainX-np.mean(trainX))/np.std(trainX)
    testX = (testX-np.mean(testX))/np.std(testX)
    
    #One hot
    numLabels = np.size(np.unique(trainY))
    trainY = np.eye(numLabels)[trainY]
    testY = np.eye(numLabels)[testY]
    
    return trainX, trainY, testX, testY, testX.shape[1], testY.shape[1]

def get_Input_Data(name, mode="coarse"):
    
    if name.upper() == "MNIST":
        print("Acquiring MNIST dataset")
        return get_MNIST_DATA()
    
    mapDict = {"Cifar-10": 
               ["cifar-10-batches-py", 
                "cifar-10.tar.gz", 
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"],
               
               "Cifar-100": 
               ["cifar-100-python", 
                "cifar-100.tar.gz", 
                "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"]
               
               }
    
    if name not in mapDict.keys():
        print("didn't find", name, "in accepted datasets")
        return -1
    
    
    #for title, descriptors in mapDict.items():
    folderName, fileName, location = mapDict[name]
    if directory_Exists(folderName) is False:
        if file_exists(fileName) is False:
            download_File(name, fileName, location)
        print("Extracting", name, "data")
        tar = tarfile.open(fileName, "r:gz")
        tar.extractall()
        tar.close()            
    
    if name == "Cifar-10":
        return get_Cifar_10_Data(folderName)
    elif name == "Cifar-100":
        return get_Cifar_100_Data(folderName, mode)
    else:
        return -1, -1, -1, -1
    

def weight_variable(shape):
    #create a weight variable within tensorflow, callable from convolutional
    #layers as well as artificial layers
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="WEIGHT")

def bias_variable(shape):
    #create a bias variable within tensorflow, callable from convolutional
    #layers as well as artificial layers    
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="BIAS")

def conv2d(x, W):
    #convolve filters W over image x
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name="CONVOLVE")

def max_pool_nxn(x, s, k):
    #pool using a k by k grid, moving in s by s
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, s, s, 1], padding='SAME', name="POOL")  

def artificial_Network_Architecture(vector):
    #prints out the shape of the artificial layers
    length = len(vector)
    if length == 2:
        print("Input layer with", vector[0], "inputs.")
        print("Output layer with", vector[1], "outputs.")
    else:
        print("Input layer with", vector[0], "inputs.")
        for i in range(1, length-1):
            print("Hidden layer with", vector[i], "nodes")
        print("Output layer with", vector[-1], "outputs.")    


def get_Name_Space(convShape, word):
    #Used to name the layers of the network
    return [word + str(i+1) for i in range(len(convShape))]

def return_x_dimension(inp, color):
    #determines the root of the input size (ex: such as sqrt(121) = 11)
    #separate cases for if there are 3 inputs due to color, in which case the
    #formula is sqrt(size/3)
    
    if color == True:
        inp = inp/3
    #square images, this is the length of one side
    xdim = sqrt(inp)
    
    #if it's not a whole number
    if int(xdim) != xdim:
        print("dimension of image must be a square, not", inp, "\n"
              "If you are inputing a color image, set color=True")
        return -1
    
    xdim = int(xdim)
    return xdim

def correct_Fc_Shape(fcShape, fcInputSize, outputLayerSize):
    #default to 3 layers: 2 hidden, 1 output
    #second hidden is half the size of the first hidden, rounded up
    #1st hidden layer has "fcInputSize" units
    #output layer has "outputLayerSize" units
    if len(fcShape) <= 1:
        fcShape = [fcInputSize, ceil(fcInputSize/2), outputLayerSize]
    else:
        #ensure the correct number of outputs
        fcShape[-1] = outputLayerSize
        fcShape[0] = fcInputSize
    
    fcShape=np.array(fcShape)
    
    for entry in fcShape:
        if entry == 0:
            print("Cannot have 0 sized layers")
            return -1
        
    return fcShape

def parse_Conv_Shape(inputLayerSize, minActGrid, filterX, filterY, 
            startingFeatures, featureScaling, poolingSize, convShape, color):
    #translates ConvShape keywords into proper entries
    #creates a default ConvShape based on the input parameters
    #attempts to create conv layers until it finds the activations are less
    #than 4*4 grids
    
    #if conv net shape is not specified, create one
    if len(convShape) == 0:
        convShape = default_conv_net(
            inputLayerSize, minActGrid, filterX, filterY, 
            startingFeatures, featureScaling, poolingSize, color)
        
        if convShape == -1:
            return -1
    
    #replace keywords if there are any
    #
    #['bottleneck', numFeatures, poolingOnLast]
    #['bottleneck', 64, 3]
    #[[1, 64, 0], [3, 64, 0], [1, 64, 3]
    #becomes a 1x1 conv then 3x3 conv then 1x1 conv with optional pooling
    #on the last conv layer of the series
    
    #Want to add in some of the googLeNet structures
    #also option to make the convnet a residual net
    #resnet-50 structure
    ##http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
    
    #translate keyword
    convShapeCopy = []
    for x in range(len(convShape)):
        if convShape[x][0] == "bottleneck":
            
            numFeatures = convShape[x][1]
            poolingOnLast = convShape[x][2]
            
            convShapeCopy.append([1, numFeatures, 0])
            convShapeCopy.append([3, numFeatures, 0])
            if poolingOnLast != 1:
                #1x1 pooling is nothing but expensive
                convShapeCopy.append([1, numFeatures, poolingOnLast])
            else:
                convShapeCopy.append([1, numFeatures, 0])
        else:
            convShapeCopy.append(convShape[x])
    
    return convShapeCopy

def extract(layerAttr):
    '''
    extracts the following features on each layer from a variety of formats:
    filterX, filterY, numberOfFeaturesToProduce, ksize, strideSize
    
    filterX is the X dimension of the filters on this layer
    filterY is the Y dimension of the filters on this layer
    numberOfFeaturesToProduce is the number of filters to use on this layer
    ksize is the kernel size for pooling
    strideSize is the stride used for pooling
    '''
    if len(layerAttr) == 5:
#[filterX, filterY, numberOfFeaturesToProduce, ksize, strideSize (for pooling)]
        return layerAttr
                
    elif len(layerAttr) == 4:
#[filterSize, numberOfFeaturesToProduce, ksize, strideSize (for pooling)]
        filterSize, Features, kSize, strideSize = layerAttr
            
        #filter is a square of filterSize x filterSize
        return filterSize, filterSize, Features, kSize, strideSize
        
    elif len(layerAttr) == 3:
#[filterSize, numberOfFeaturesToProduce, poolingSize]
        filterSize, Features, poolingSize = layerAttr
        
        #filter is a square of filterSize x filterSize
        #pooling box size is the same amount as the stride,
        #if poolingSize is 2, a 2x2 box will move with 2x2 strides
        return filterSize, filterSize, Features, poolingSize, poolingSize
                    
    else:
        print("UNRECOGNIZED INPUT:", layerAttr, "ABORT")
        return -1, -1, -1, -1, -1

def def_conv_net_error_check(inputLayerSize, minActGrid, filterX, filterY, 
                     startingFeatures, featureScaling, poolingSize, color):
    #checks for errors in the input parameters before creating 
    #a default conv net
    if type(inputLayerSize) != int:
        print("inputLayerSize must be an int")
        return -1
    
    if type(minActGrid) != int:
        print("minActGrid must be an int")
        return -1
    
    if type(filterX) != int:
        print("filterX must be an int")
        return -1
    
    if type(filterY) != int:
        print("filterY must be an int")
        return -1  
    
    if type(startingFeatures) != int:
        print("startingFeatures must be an int")
        return -1
        
    if type(featureScaling) != int:
        print("featureScaling must be an int")
        return -1
    
    if type(poolingSize) != int:
        print("poolingSize must be an int")
        return -1
    
    if poolingSize <= 1:
        print("poolingSize must be greater than 1")
        return -1
    
    if color not in [True, False]:
        print("color must be boolean")
        return -1
    
    #passed all tests
    return 0
        
def default_conv_net(inputLayerSize, minActGrid=4, filterX=5, 
                     filterY=5, startingFeatures=32, featureScaling=2,
                     poolingSize=2, color=False):
    
    #design a simple convolutional network
    #default is 5x5 filters, 32 features in the first layer (2 times as
    #many in each subsequent layer), 2*2 pooling with a stride of 2*2
    #creates layers until it has activations that are 4x4 grids
    #all these hardcoded values can be changed however desired by modifying
    #the input parameters
    
    rc = def_conv_net_error_check(
        inputLayerSize, minActGrid, filterX, filterY, 
        startingFeatures, featureScaling, poolingSize, color)
    
    if rc == -1:
        return -1
    
    
    convShape = []
    
    
    if color == True:
        currSize = sqrt(inputLayerSize/3)
    else:
        #get the length of the square input
        currSize = sqrt(inputLayerSize)
        
    if int(currSize) != currSize:
        print("Input images must be squared if you want to "
              "create a default convnet, not", currSize, "\nIf you're working"
              " with Color inputs, don't forget to set color=True.")
        return -1
    
    currSize = int(currSize)
    

    if currSize <= minActGrid:
        print("Image size smaller than or equal"
              " to requested minimum of", minActGrid)
        return -1
    
    #default 5 by 5 conv filter, 32 features (mult by 2 each layer)
    Features = startingFeatures

    #loops until our activations are grids of at most (min_act_grid-1)^^2
    #default is looping until we have 4x4 activations or smaller 
    #from the convnet

    while currSize > minActGrid:
        # conv layer with default filter size, finding 32 features
        # on the first layer, then twice as many on each subsequent layer
        convShape.append([filterX, filterY, 
                          Features, poolingSize, poolingSize])
        
        #extract "feature_scaling" times the features on the next layer
        #default is doubling the features at the next layer
        Features *= featureScaling
        currSize = ceil(currSize/poolingSize)  

    return convShape

def add_Conv_Layers(IN, convShape, xdim, silent=False, color=False):
    #adds convolutional layers as specified by convShape
    #to understand the possible values, look at definition of 
    #extract function
    nameSpace = get_Name_Space(convShape, "conv")
    
    #if color is true, then we have 3 times the channels as we have RGB
    if color == False:
        conv_shape = tf.reshape(IN, [-1, xdim, xdim, 1])
        currFeatures = 1
    if color == True:
        #reformat the [[RRRR...BBBB...GGGG....],...] data into 
        #shape = (numSamples, 32 columns, 32 rows, 3 values per cell)
        conv_shape = tf.transpose(
            tf.reshape(IN, [-1, 3, xdim, xdim]), [0, 2, 3, 1])
        
        currFeatures = 3
    
    #keep track of a few inputs for tensorboard visuals
    tf.summary.image('Example Input', conv_shape, 3)
    conv = conv_shape
    
    counter = 0
    for layerAttr in convShape:
        with tf.name_scope(nameSpace[counter]):
        
            filterX, filterY, Features, kSize, strideSize = extract(layerAttr)
            
            conds = [
                filterX >= 1,
                filterY >= 1,
                Features >= 1,
                kSize >= 0,
                strideSize >= 0]
            
            if False in conds:
                print("filterX, filterY and number of features must be "
                      "greater than or equal to 1. kSize and strideSize must "
                      "be greater than or equal to 0")
                sys.exit(1)
            
            #currFeatures is the size of input to the layer
            #Features is the size of output of the layer
            W = weight_variable([filterX, filterY, currFeatures, Features])
            b = bias_variable([Features]) 
            
            #keep track of previous 
            currFeatures = Features
            
            
            #convolution 
            conv = tf.nn.relu(conv2d(conv, W) + b)
            
            #keep track of values over time
            tf.summary.histogram('Weight', W)
            tf.summary.histogram('Bias', b)
            tf.summary.histogram('Activations', conv)
            
            #no point if both are 1 or any are less than 1, 
            #anything else is fine
            if (kSize >= 1 and strideSize > 1) \
               or (kSize > 1 and strideSize >= 1):
                conv = max_pool_nxn(conv, strideSize, kSize)
                
                if silent == False:
                    print("Flow through conv layer,", filterX, "x", filterY, 
                          "filter applied, producing", Features, "features",
                          "with pooling using a", kSize, "x", kSize, 
                          "box moving by", strideSize, "x",strideSize, "steps")
                
            else:
                if silent == False:
                    print("Flow through conv layer,", filterX, "x", filterY,
                          "filter applied, producing", Features, "features",
                          "without pooling")  
            
            counter += 1
    
    #need to flatten, and we have lst[1]*lst[2]*lst[3] items we need to fit
    #in our output
    cOutShape = conv.get_shape().as_list()
    fcInputSize = cOutShape[1]*cOutShape[2]*cOutShape[3] 
    flattenConv = tf.reshape(conv, [-1, fcInputSize])    
    
    return flattenConv, fcInputSize

def add_Conv_Layers_DROP(IN, convShape, xdim, silent=False, color=False):
    #not added yet, but adds the ability to add dropout to the conv layers
    #https://arxiv.org/pdf/1506.02158v6.pdf
    
    nameSpace = get_Name_Space(convShape, "conv")
    
    if color == False:
        conv_shape = tf.reshape(IN, [-1, xdim, xdim, 1])
        currFeatures = 1
    if color == True:
        conv_shape = tf.transpose(
            tf.reshape(IN, [-1, 3, xdim, xdim]), [0, 2, 3, 1])
        currFeatures = 3
    
    tf.summary.image('Example Input', conv_shape, 3)
    conv = conv_shape
    
    counter = 0
    
    keepProbConv = tf.placeholder(tf.float32, name="Keep_Probability_Conv")
    
    for layerAttr in convShape:
        with tf.name_scope(nameSpace[counter]):
        
            filterX, filterY, Features, kSize, strideSize = extract(layerAttr)
            
            conds = [
                filterX >= 1,
                filterY >= 1,
                Features >= 1,
                kSize >= 0,
                strideSize >= 0]
            
            if False in conds:
                print("filterX, filterY and number of features must be "
                      "greater than or equal to 1. kSize and strideSize must "
                      "be greater than or equal to 0")
                sys.exit(1)
            
            #currFeatures is the size of input to the layer
            #Features is the size of output of the layer
            W = weight_variable([filterX, filterY, currFeatures, Features])
            b = bias_variable([Features]) 
            
            #keep track of previous 
            currFeatures = Features
            
            #dropout on convolution layer
            conv = tf.nn.dropout(conv, keepProbConv)
            
            #convolution
            conv = tf.nn.relu(conv2d(conv, W) + b)
            
            #keep track of values over time
            tf.summary.histogram('Weight', W)
            tf.summary.histogram('Bias', b)
            tf.summary.histogram('Activations', conv)
            
            #no point if both are 1 or any are less than 1, 
            #anything else is fine
            if (kSize >= 1 and strideSize > 1) \
               or (kSize > 1 and strideSize >= 1):
                conv = max_pool_nxn(conv, strideSize, kSize)
                
                if silent == False:
                    print("Flow through conv layer,", filterX, "x", filterY, 
                          "filter applied, producing", Features, "features",
                          "with pooling using a", kSize, "x", kSize, 
                          "box moving by", strideSize, "x",strideSize, "steps")
                
            else:
                if silent == False:
                    print("Flow through conv layer,", filterX, "x", filterY,
                          "filter applied, producing", Features, "features",
                          "without pooling")  
            
            counter += 1
    
    #need to flatten, and we have lst[1]*lst[2]*lst[3] items we need to fit
    #in our output
    cOutShape = conv.get_shape().as_list()
    fcInputSize = cOutShape[1]*cOutShape[2]*cOutShape[3] 
    flattenConv = tf.reshape(conv, [-1, fcInputSize])    
    
    return flattenConv, fcInputSize, keepProbConv


def add_Art_Layers(FC, fcShape=[2048, 1024, 10]):
    '''
    FC is the output of the convolutional layers
    it's now going to be passed through the artificial layers
    
    each entry in fcShape dictates the size of that layer
    
    #batch normalization improves the
    #speed of training
    #not implemented yet
    #scaling factor gamma and shifting factor beta
    #as well as the weights, bias is taken care of by the shifting factor
    activation(WEIGHTS*(SCALE*((x-mean)/(stdDev + epsilon)) + SHIFT))
    '''
    
    nameSpace = get_Name_Space(fcShape, "FC")
    
    keepProb = tf.placeholder(tf.float32, name="Keep_Probability")
    
    for i in range(len(fcShape) - 1):
        with tf.name_scope(nameSpace[i]):
                
            #dropout to reduce overfitting
            DROP = tf.nn.dropout(FC, keepProb)
            
            #weights and biases
            W = weight_variable([fcShape[i], fcShape[i+1]])
            b = bias_variable([fcShape[i+1]])
            
            #WX+b
            ACT = tf.matmul(DROP, W) + b 
            FC = tf.nn.relu(ACT)
            
            #track variables
            tf.summary.histogram('Weight', W)
            tf.summary.histogram('Bias', b)
            tf.summary.histogram('Activations', FC)
    
    return ACT, keepProb


def create_CNN(inputLayerSize, outputLayerSize, fcShape=[], convShape=[], 
               silent=False, minActGrid=4, filterX=5, filterY=5, 
               startingFeatures=32, featureScaling=2, poolingSize=2, 
               color=False):
    '''
    inputLayerSize is the size of the input, so in MNIST, that's 28*28=784
    outputLayerSize is the number of labels that are possible, for MNIST
    this is 10 possible labels
    
    fcShape dictates the shape of the artificial network after the 
    convolutional layers
    each entry is the number of nodes in the layer
    first and last entries of fcShape will self configure
    
    convShape dictates the shape of the convolutional layers
    each entry in convShape is 
    [filterX, filterY, FeaturesToProduce, ksize, strideSize (for pooling)]
    or
    [filterSize, numberOfFeaturesToProduce, ksize, strideSize (for pooling)]
    or
    [filterSize, numberOfFeaturesToProduce, poolingSize]
    or
    ["bottleneck", numberOfFeaturesToProduce, poolingSizeOnLast]
    
    produce equivalent nets
    convShape= [[5, 5, 32, 2, 2], [1, 3, 64, 3, 2], [5, 5, 256, 4, 3]], 
    fcShape = [2048, 1024, 10] 
    
    convShape=[[5, 32, 2], [1, 3, 64, 3, 2], [5, 256, 4, 3]],
    fcShape=[0, 1024, 0]
    
    set ksize or strideSize to 0 to have no pooling at this layer
    
    minActGrid, filterX, filterY, startingFeatures, featureScaling 
    and poolingSize are all for if you want a default network to be built
    
    minActGrid is the size the activation outputs need to get to for it to stop
    producing convolutional layers
    default is stopping once the layers have reduced the output images to 4*4
    
    filterX is the number of pixels wide the filters will be
    filterY is the number of pixels tall the filters will be
    
    startingFeatures is the number of features to extract on the first layer
    
    featureScaling is the magnitude of increase of features to extract 
    at each layer
    the default is to extract twice the number of features at each layer
    
    poolingSize is the size of the pooling window that will be passed over the
    output, it is also the stridesize so no pixels are ignored
    '''
    
    #modify values within convShape to ensure it is correct
    #OR build one from scratch if one is not provided
    convShape = parse_Conv_Shape(inputLayerSize, minActGrid, filterX, filterY, 
            startingFeatures, featureScaling, poolingSize, convShape, color)
    if type(convShape) == int:
        return -1, -1, -1, -1
    
    #input and output placeholder, feed data to IN, feed labels to LABEL_IN
    with tf.name_scope("IN"):
        IN = tf.placeholder(tf.float32, [None, inputLayerSize], 
                            name="InputData")
    with tf.name_scope("OUT"):
        LABEL_IN = tf.placeholder(tf.float32, [None, outputLayerSize], 
                                  name="OutputData") 
    
    #get length of a side of the square
    xdim = return_x_dimension(inputLayerSize, color)
    if xdim == -1:
        return -1, -1, -1, -1
    
    #CONVOLUTIONAL LAYERS
    conv, fcInputSize = add_Conv_Layers(IN, convShape, xdim, silent, color)
    if conv == -1:
        return -1, -1, -1, -1

    #set the output and input layers
    fcShape = correct_Fc_Shape(fcShape, fcInputSize, outputLayerSize)
    if type(fcShape) == int:
        return -1, -1, -1, -1
    
    #ARTIFICIAL LAYERS
    OUT, keepProb = add_Art_Layers(conv, fcShape)
    
    #print out the shape of the artificial network
    artificial_Network_Architecture(fcShape)   
    
    return IN, LABEL_IN, OUT, keepProb
    
 
def set_Display_Step(step):
    rv = int(step/500)
    if rv == 0:
        rv += 1
    return rv

def BATCH(to_batch_x, to_batch_y, batchSize=0):
    #create a batch out of the data of size batchSize
    length = to_batch_x.shape[0]
    # if batchSize is unspecified, or batchSize is greater than the number
    # of samples available
    if batchSize <= 0 or batchSize > length:
        batchSize = int(length/10)
        if batchSize == 0:
            batchSize += 1
    batch_indices = np.random.choice(length, batchSize, replace=False)
    return to_batch_x[batch_indices], to_batch_y[batch_indices]

def train_CNN(IN, LABELS_IN, LOGITS, keepProb, trainX, trainY, testX, testY, 
              keepPercent=0.5, batchSize=50, trainingEpochs=20000, alpha=1e-4,
              silent=False, dest='', modelDest='', modelExists=False, 
              opt="Adam"):
    '''
    IN is the placeholder for the input data
    LABELS_IN is the placeholder for the label data
    LOGITS is the output of the network before applying an activation function
    keepProb is the placeholder for dropout keep rate
    keepPercent is the dropout keep rate
    
    batchSize is the size of the sample to retrieve from the training data
    on each training step
    
    trainingEpochs is the number of episodes of training to be performed
    
    trainX, trainY, testX, testY are the training and testing datasets
    '''
    sess = tf.InteractiveSession()
    
    #calculate loss function, backpropogate using adam gradient descent
    #measure accuracy of output, initialize saver
    with tf.name_scope("EVAL"):
        with tf.name_scope("Cross_Entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=LABELS_IN, logits=LOGITS))

        if opt.lower() == "adam":
            with tf.name_scope("ADAM"):
                train_step = tf.train.AdamOptimizer(alpha).minimize(
                    cross_entropy)
                
        elif opt.lower() == "rms":
            with tf.name_scope("RMS"):
                train_step = tf.train.RMSPropOptimizer(alpha).minimize(
                    cross_entropy)
        else:
            print("Opt must be one of Adam or RMS")
            return -1, -1, -1, -1, -1
            
        with tf.name_scope("PRED"):
            correct_prediction = tf.equal(
                tf.argmax(LOGITS,1), tf.argmax(LABELS_IN,1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            
        with tf.name_scope("SAVER"):
            saver = tf.train.Saver()        
            
    #keep track for tensorboard
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Cross Entropy', cross_entropy)
    
    #get path to current folder and this OS's separator
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sep = os.sep
    
    
    #get destination for saving the tensorboard data
    if dest == '':
        dest = input("Please enter the directory you would like to save "
        "tensorboard data to, \nwill default to '" + dir_path + sep + 
        "tensorboard_data' if nothing is specified:")
    if dest == '':
        dest = dir_path + sep + 'tensorboard_data'
        
    #message on how to use tensorboard
    print("\nrun in terminal to get insight into training:\ntensorboard"
    " --logdir", dest,"\nThen go to http://0.0.0.0:6006 in a browser.\nTo"
    " compare multiple models/runs, store each in a separate folder then point"
    " logdir at that folder\nLike such:\ntensorboard --logdir parent_folder\n"
    "Where each run is stored such as parent_folder/run1/, parent_folder/run2/"
    ", etc\nFinally, if you are storing different models or runs, store them"
    " in separate \nfolders as the data will be overwritten otherwise\n\n")
    
    #get destination for saving model checkpoints
    if modelDest == '':
        modelDest = input("Please enter the directory you would like to save "
        "Model data to, \nwill default to '" + dir_path + sep + "model_data' " 
        "if nothing is specified:")
    if modelDest == '':
        modelDest = dir_path + sep + 'model_data' + sep
    
    #for some reason, the saver can't make it's own folder, but the summary
    #writer can
    if not os.path.exists(modelDest):
        os.makedirs(modelDest)    
    
    #create summary writer
    merged_summary = tf.summary.merge_all()
    writer_data = tf.summary.FileWriter(dest)
    writer_data.add_graph(sess.graph)
    
    displayStep = set_Display_Step(trainingEpochs)
    
    #last output until the test accuracy
    if silent==True:
        print("Training...")  
    
    #run training
    with tf.name_scope("EVALUATE"):
        sess.run(tf.global_variables_initializer())
        
        #restore model if it exists
        if modelExists == True:
            try:
                saver.restore(sess, modelDest)
                print("Restored model from:", modelDest)    
            except:
                print("Failed to restore model, training from start.")
        
        #train for trainingEpochs iterations
        for i in range(trainingEpochs):
            sampleX, sampleY = BATCH(trainX, trainY, batchSize)
            
            #print accuracy
            if i%displayStep == 0:
                if silent == False:
                    train_accuracy = accuracy.eval(feed_dict={
                        IN:sampleX, LABELS_IN:sampleY, keepProb:1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
            
            #take measurements for tensorboard        
            if i%5==0:
                s = sess.run(merged_summary, feed_dict={
                    IN:sampleX, LABELS_IN:sampleY, keepProb:1.0})
                writer_data.add_summary(s, i)
            
            #Save model
            if (i%1000 == 0 or i ==(trainingEpochs-1)) and i != 0:
                path = saver.save(sess, modelDest)
                print("Saved model at:", path)
            
            #execute training step with dropout
            train_step.run(feed_dict={
                IN:sampleX, LABELS_IN:sampleY, keepProb:keepPercent})
            
        #print final accuracy
        print("test accuracy %g"%accuracy.eval(feed_dict={
            IN:testX, LABELS_IN:testY, keepProb:1.0}))    
    
    return IN, LABELS_IN, LOGITS, keepProb, sess




def main():
    #99.5% using default settings
    trainX, trainY, testX, testY, inputSize, outputSize = get_Input_Data("mnist")
    
    #76.5% using default settings and alpha=0.001, batchsize=100
    #trainX, trainY, testX, testY, inputSize, outputSize = get_Input_Data("Cifar-10")
    
    #trainX, trainY, testX, testY, inputSize, outputSize = get_Input_Data("Cifar-100", "coarse")
    #trainX, trainY, testX, testY, inputSize, outputSize = get_Input_Data("Cifar-100", "fine")
    
    convShape = [[5, 32, 2], ['bottleneck', 64, 2], [5, 128, 3]]
    
    IN, LABELS_IN, LOGITS, keepProb = create_CNN(
        inputSize, outputSize, convShape=convShape)
    
    if type(IN) == int:
        print("FAILED")
    else:
        IN, LABELS_IN, OUTPUT, keepProb, session = train_CNN(IN, LABELS_IN, 
                                                             LOGITS, keepProb, 
                                                             trainX, trainY, 
                                                             testX, testY, 
                                                             batchSize=100, 
                                                             alpha=0.001) 
                                                             #modelExists=True)
main()
