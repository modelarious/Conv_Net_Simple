README

CIFAR-10 CITATION: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

INTRO



Implements a general form of a feed forward Convolutional neural network with dropout on 
each of the artificial layers and normalization of the inputs. Complete with Tensorboard
and periodic model parameter saving to easily start from the last checkpoint.
  
Three functions required to deploy network:
	get_Input_Data
	create_CNN
	train_CNN

The network is built as a computation graph within tensorflow.




FUNCTIONALITY



get_Input_Data
	
	Purpose:
		
		Retrieve 1 of 3 datasets for use in training.
		
	Definition:
	
		get_Input_Data(name, mode="coarse")
		
	Input:
	
		accepts a small range of inputs:
		
		get_Input_Data("mnist")
		get_Input_Data("Cifar-10")
		get_Input_Data("Cifar-100", "coarse")
		get_Input_Data("Cifar-100", "fine")
		
		Supplying keyword "mnist" or "Cifar-10" extracts the data as normal.
		Supplying "Cifar-100" adds the ability to choose if you want to predict
		the coarse labels (20 labels) or the fine labels (100 labels).
		

	Returns:
	
		first four return values are numpy arrays, last two are integers

		trainX, trainY, testX, testY, size_of_a_single_test_case, number_of_target_classes


	Method:
	
		First checks if the extracted data it wants to work with is in the working folder,
		if not it checks if the tar.gz of the data exists and extracts if it does.  If 
		the archive doesn't exist either, it downloads and extracts the datasets from 
		Toronto's Machine Learning data repo: 
		http://www.cs.toronto.edu/%7Ekriz/cifar.html (CIFAR-10 and CIFAR-100)

		After reading the data into python, it then one hot encodes the labels and 
		normalizes the testing and training data by subtracting the mean and dividing 
		by the std deviation.

		Then it returns the data in the form specified above.
		
	





create_CNN

	Purpose:
	
		Creates the computational graph that represents the network specified by the 
		input parameters.  Capable of creating it's own structure for the graph 
		dynamically based on the size of the input data if no parameters are specified.
		
		Uses tensorboard to visualize the distribution of the weights, biases and 
		activations over time, also takes a sample of what's being input to the network
		to highlight if there's any problems with the input images.
		
		When handling each of the layers, they are all given their own name space to 
		improve the readability of the tensorboard graph.

	Definition:
	
        create_CNN(inputLayerSize, outputLayerSize, fcShape=[], convShape=[], 
                   silent=False, minActGrid=4, filterX=5, filterY=5, 
                   startingFeatures=32, featureScaling=2, poolingSize=2, 
                   color=False)
    
    Input:
    
    	inputLayerSize is the size of one input example, so if you are using 28 x 28
    		grayscale images (1 color channel), this number would be 28*28*1=784.  If using
    		32 x 32 color images (3 color channels), this number would be 32*32*3=3072.
    		Takes an integer.
    	
    	outputLayerSize is the number of categories that the data could fall into. Takes
    		an integer.
    	
    	fcShape dictates the shape of the fully connected layers that come after the
    		convolution.  Each entry contains the number of nodes in each layer, as well
    		the first and last entries will correct themselves to the right values.
    		ex: [0, 240, 360, 0] specifies 4 layers, first one is the same size as the 
    		output from the convolutional layer, next has 240 nodes, next has 360 nodes 
    		then finally the output layer has the same number of nodes as there are 
    		categories.
    	
    	convShape dictates the shape of the convolutional layers, each entry
    		corresponds to a layer.  There are many ways to enter data for a layer:
    	
    		[filterX, filterY, FeaturesToProduce, ksize, strideSize (for pooling)]
    		or
    		[filterSize, FeaturesToProduce, ksize, strideSize (for pooling)]
    		or
    		[filterSize, FeaturesToProduce, poolingSize]
    		or
    		["bottleneck", FeaturesToProduce, poolingSizeOnLast]
    	
    		Where filterX and filterY dictate the shape of the kernel at the layer (if 
    		it's just filterSize then filterSize=filterX=filterY), FeaturesToProduce is 
    		an indicator of how many filters should be used at this layer, ksize is the 
    		maxpool size and strideSize dictates what size strides will be used in the 
    		pooling step (if only poolingSize is supplied, poolingSize=ksize=strideSize).
    	
    		Finally, ["bottleneck", FeaturesToProduce, poolingSizeOnLast] creates three 
    		layers, a 1x1 conv layer no pooling, 3x3 conv layer no pooling, 1x1 conv 
    		layer with "poolingSizeOnLast" size pooling, all three layers use 
    		"FeaturesToProduce" number of filters.
    		
    		ex: if supplied with:
    		[[5, 32, 2], 
    		 [1, 3, 64, 3, 2], 
    		 [5, 256, 4, 3]
    		]
    		first layer has a 5x5 filter, 32 features and 2x2 pooling with 2x2 stride
    		second layer has 1x3 filter, 64 features and 3x3 pooling with 2x2 stride
    		third layer has 5x5 filter, 256 features, 4x4 pooling and 3x3 stride
    	
    	silent is a boolean value, if set to true, will print out less information.
    	
    	minActGrid, filterX, filterY, startingFeatures, featureScaling and poolingSize
    		are all used by default_conv_net (defined below)
    	
    	color is a boolean value, if set to true, then it is assumed the input data has	
    		three color channels, and the samples are set up in the following format:
            [[image1],
             [image2],
              ...
            ]
    		
    		Where each image is stored as all red pixels, then all green pixels then all
    		blue pixels, starting with the first row of red pixels.
    		
    		ex:
    		
    		first image is 32x32 and stored as: 
    		[RRRRRRRRRRRRRRRRRRRR...GGGGGGGGGGGGG...BBBBBBBB]
    		Where the first 32 R's correspond to the R values for the first row of the
    		image.
    		
    		All of this must happen because of the reshaping and transposing function
    		to input the data into the convolutional layers assumes this format.
    		
    	
    	Returns:
    	
    		IN is the input placeholder tensor, where trainX or testX should be fed
    		LABEL_IN is the label placeholder tensor, where trainY or testY should be fed
    		OUT is a tensor of the predictions made by the net based on data in IN
    		keepProb is the probability of keeping the value of a node or dropping
    			it to reduce overfitting, default 50% of the nodes are kept and 50% 
    			dropped (takes a float value between 0 and 1)
    			
    		IN, LABEL_IN, OUT, keepProb
    		
    	Method:
    	
    		First parses the shape provided for convShape, changes any keywords into
    		multiple layers based on their function.  If a convShape hasn't been defined, 
    		it creates one using default_conv_net.  It then adds on the convolutional
    		layers to the placeholder it created (note: I wrote the conv layers and art 
    		layers functions in such a way that they append themselves to whatever 
    		tensor I provide (provided it can be reshaped correctly.  In this case, I'm
    		providing the input placeholder tensor).
    		
    		Then it parses the shape of the requested fully connected layers (setting the
    		size of the output of the conv layers as the first layer size).  Then it
    		appends the fully connected layers to the network.  These layers have dropout
    		between each of them to improve on overfitting (which I found required longer
    		training times, but produced 4% better results on my best architecture for
    		Cifar-10).
    		
    		After which, all the placeholders are returned as well as the method of
    		accessing the network's predictions based on chosen input.



default_conv_net

	Purpose:
	
		Define an adequate convolutional net structure for the input data when one is not
		provided by the user.
		
	Definition:
	
		default_conv_net(inputLayerSize, minActGrid=4, filterX=5, 
                         filterY=5, startingFeatures=32, featureScaling=2,
                         poolingSize=2, color=False)
    
    Input:
    	
    	inputLayerSize is used to determine when to stop making layers.  First the square 
    		root is taken to determine the length of one side of the square images, then 
    		it is divided by the poolingSize each layer as such: 
    		inputLayerSize = ceil(inputLayerSize/poolingSize)
    		
    		This is being used to track the size of the activation planes that are being
    		output by each layer of the convnet, once it finds that the size of the 
    		activations on a layer are smaller than "minActGrid", it stops making layers.
    		
    	minActGrid controls the minimum size of the activation grids that are being
    		output by the final layer of the convnet.  So for example, the default is 4,
    		so once pooling has reduced the output of the convolutional layer to 4x4
    		grids, no more layers are created
    		
    	filterX describes the size in the X dimension of the filters used at each layer
    	
    	filterY describes the size in the Y dimension of the filters used at each layer
    	
    	startingFeatures is the number of filters that will be used on the first layer.
    		At each subsequent layer the number of filters is multiplied by 
    		"featureScaling", so in the default, each layer has double the filters
    		of the previous
    	
    	featureScaling is a multiplier for the number of filters used in each layer,
    		each layer has "featureScaling" times as many filters as the previous
    	
    	poolingSize is the size of the pooling used at each layer, the stride size is
    		equal to the pooling size
    		
    	color is a boolean, if it is true, we assume that there are 3 color channels and
    		first divide inputLayerSize by 3 before taking the square root
    	
    	
    Returns:
    
    	the shape of the convnet it created (as a list)
    
    
    Method:
    
    	Determine the size of the inputs: sqrt(inputLayerSize) if working with grayscale
    	or sqrt(inputLayerSize/3) if working with color.  This variable keeps track of
    	the size of the current outputs (pooling reduces the size of the outputs).  Once
    	the size of the outputs of the current layer are equal or less than the desired
    	minimum output size (dictated by minActGrid), no more layers are produced.
    	
    	Each layer produced has filters of size filterX x filterY, pooling of size
    	"poolingSize" and "previousLayerFeatures" * "featureScaling" number of features.
    	
    	Then the shape it decided on is returned for use by create_CNN
    


train_CNN
    
    Purpose:
    
    	Appends the desired training algorithm, default is Adam optimizer, but RMSProp is
    	also available for second order optimization (helps to prevent problems with 
    	non-convex loss functions).  Saves the model periodically and uses tensorboard to 
    	track accuracy on current batch from training data as well as tracking the 
    	(hopeful) reduction of the cost function.  Saves the structure of the 
    	computational in a much more readable format in tensorboard.
    	
    	All the modules appended in this step are done so in the name space of "EVAL" to
    	improve the visualization of the graph in tensorboard.
    	
    Definition:
    
    	train_CNN(IN, LABELS_IN, LOGITS, keepProb, trainX, trainY, testX, testY, 
              keepPercent=0.5, batchSize=50, trainingEpochs=20000, alpha=1e-4,
              silent=False, dest='', modelDest='', modelExists=False, opt="Adam")
            
    Inputs:
    	
    	IN is the placeholder tensor for input data X.
    	
    	LABELS_IN is the placeholder tensor for the label data Y.
    	
    	LOGITS is the output tensor of the network
    	
    	keepProb is the placeholder tensor for dropout percent
    	
    	trainX is the input training data
    	
    	trainY is the input training labels
    	
    	testX is the final testing data to evaluate the network
    	
    	testY is the final testing labels
    	
    	keepPercent is the complement of the dropoutRate. DropoutRate dictates the
    		percentage of nodes on the previous layer to randomly ignore in order to
    		improve the generality of the predictions made. So keepPercent is
    		1 - dropoutRate
    	
    	batchSize is the size of the randomized batch fed to the optimizer for each epoch
    	
    	trainingEpochs is the number of iterations the training loop is executed
    	
    	alpha is the "learning rate", more intuitive to think of it as the step size
    		moved when using back propagation to adjust the weights and biases.
    	
    	silent is boolean, if true, will not give nearly as much information
    	
    	dest is the path that the tensorboard data will be saved.  If not specified, it
    		will prompt the user for a path
    	
    	modelDest is the path where the model checkpoints will be saved.  If not 
    		specified, it will prompt the user for a path
    		
    	modelExists is boolean, if it's true, train_CNN will attempt to reload the last
    		checkpoint into the current model (will only fail if the models have
    		different architectures)
    	
    	opt is a string that is either "Adam" or "RMS", specifies the optimizer that will
    		be used.  "Adam" specifies the adam optimizer, "RMS" specifies the RMSProp
    		optimizer.
    
    Returns:
    
    	IN: The input placeholder tensor
    	LABELS_IN: the label placeholder tensor
    	LOGITS: the output of the network
    	keepProb: the placeholder for dropout percent
    	sess: the session that was just trained
    	
    	IN, LABELS_IN, LOGITS, keepProb, sess
    	
    	
    	
    Method:
    
    	Uses a cross entropy loss function that is minimized by either Adam or RMS
    	versions of gradient descent.  Appends onto that a function that calculates
    	the accuracy of the output of the network based on the inputs and the correct
    	input labels.
    	
    	Sets up the summaries to log the training, graph saver to store a representation
    	of the graph in tensorboard and the saver to periodically save checkpoints of 
    	the model's training.
    	
    	Executes the training steps, saving tensorboard data every 5 steps and saving the
    	model checkpoints every 1000 steps (and on the last step of training).
    	
    	Once it executes all the training loops, it then evaluates the testing data to
    	check the quality of the internal representation of the data
    	
    	Returns all the parameters needed to work with the now trained model
    	
    	aside: I originally had it evaluate the testing data every 200 steps to make it 
    	easier to determine the best value for trainingEpochs, but it made the execution
    	run at 60% of the speed due to the large amount of data in the testing data (when
    	compared to the size of a batch of the training data) so I removed the feature.
    	
    
    	
FURTHER:

	Data augmentation would be a great addition: some rotations, reflexions, 
	translations, etc.	
	
	Batch Normalization looks very promising.
	
	experiment with dropout on conv layers
	https://arxiv.org/pdf/1506.02158v6.pdf
	https://www.reddit.com/r/MachineLearning/comments/42nnpe/why_do_i_never_see_dropout_applied_in/#bottom-comments
	If not dropout, then L1 or L2 regularization could be promising as well.
	
	Read about a good initialization method being the identity matrix. Also read about an initialization technique which uses
	RBMs to set the initial weights, then train them from that initialization.
	
	Bayesian hyperparameter optimization is very impressive, the evolutionary technique
	is a lot less efficient. Though evolutionary techniques are good for coming up with
	great optimizations of very complex systems (such as the designing the optimal
	shape for the antenna for two of NASA's expeditions)
