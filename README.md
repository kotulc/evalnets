#Evalnets

Evalnets is a set of TensorFlow scripts used for evaluation of the generalization performance of various neural network layer encodings against the MNIST dataset. This script borrows primarily from tensorflow mnist tutorials found in the tensorflow repository under tensorflow/tensorflow/examples/tutorials/mnist/ including fully_connected_feed.py and utilizes the functionality of input_data.py. control_eval_fc2.py generates a graph used to train a network with a conv-fc1-fc2-out architecture where conv is a convolutional layer followed by a max_pool operation, fc1 and fc2 are fully connected layers. The test performance of the network generated in control_eval_fc2.py is for comparison purposes against networks generated in exp_eval_fc2.py.

1/3/2016


1/2/2016
Update control_eval_fc2.py to solve for a single numeric class. This 'binary' classification approach targets a single label such as '0' and is trained against all other classes where input_data returns data sets with a m x 2 label matrix. label(i,1) is the binary value denoting a member of the 'null' set (all labels with the exception of target_label) and label (i,2) is a binary identifier of the target set.

Average accuracy for '0' classification with this architecture is 99.83% over 5 runs. 
