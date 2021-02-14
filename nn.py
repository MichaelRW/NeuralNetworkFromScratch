

#%% Modules

from data import get_mnist

import matplotlib
import matplotlib.pyplot as plt
import numpy as np



#%% Train Three Layer (Input-Hidden-Output) Neural Network

# Nomenclature,
#
# w -> connection/link weights
# b -> bias values
# i -> input layer
# h -> hidden layer
# o -> output layer
# l -> labels

images, labels = get_mnist()  # 60000-by-784, 60000-by-1, respectively.

# Initialize input-to-hidden layer weights and set bias values to zero.
np.random.seed(0)
w_i_h = np.random.uniform( -0.5, 0.5, (20, 784) )  # 20-by-784
b_i_h = np.zeros((20, 1))  # 20-by-1

# Initialize hidden-to-output layer weights and set bias values to zero.
np.random.seed(0)
w_h_o = np.random.uniform( -0.5, 0.5, (10, 20) )  # 10-by-20
b_h_o = np.zeros((10, 1))  # 20-by-1


print( "\n\n" )

learn_rate = 0.01;  epochs = 50

nr_correct = 0;  accuracy = list()

for epoch in range(epochs):  # From 0 to (epochs - 1)
    
    for img, l in zip(images, labels):  # Group a row from images and labels
        
        img.shape += (1,)  # Make the second dimension of the 1-Tuple (i.e. 1) explicit.
        l.shape += (1,)  # Make the second dimension of the 1-Tuple (i.e. 1) explicit.
        
        
        # Forward Propagation -> Input Layer to Hidden Layer
        h_pre = b_i_h  +  w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))  # Activation
        
        # Forward Propagation -> Hidden Layer to Output Layer
        o_pre = b_h_o  +  w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))  # Activation


        # Compute Cost/Error/Performance Metric
        e = 1 / len(o) * np.sum( (o - l) ** 2, axis=0 )
        #
        # e -> 1-by-1, a scalar -> Average squared difference between predicted and expected output.
        # o, l -> 10-by-1# 
        #
        nr_correct = nr_correct + int( np.argmax(o) == np.argmax(l) )


        # Backpropagation -> Output Layer to Hidden Layer -> The cost function derivative.
        delta_o = o - l  # 10-by-1
        #
        w_h_o = w_h_o  +  -1.0*learn_rate*delta_o @ np.transpose(h)  # 10-by-20 + (1-by-1 * 10-by-1 * 1-by-20) -> 10-by-20
        b_h_o = b_h_o  +  -1.0*learn_rate*delta_o  # 10-by-1 + (1-by-1 * 10-by-1) -> 10-by-1
        
        # Backpropagation -> Hidden Layer to Input Layer -> The activation function derivative.
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))  # 20-by-10 @ 10-by-1 * 20-by-1 -> 20-by-1 (@ -> matrix multiplication; * -> element-by-element multiplication)
        #
        w_i_h = w_i_h  +  -1.0*learn_rate*delta_h @ np.transpose(img)  # 20-by-784 + (1-by-1 * 20-by-1 @ 1-by-784) -> 20-by-784
        b_i_h = b_i_h  +  -1.0*learn_rate*delta_h  # 20-by-1 + (1-by-1 *20-by-1) -> 20-by-1


    # Show accuracy for this epoch.
    print( f"Epoch {epoch + 1} of {epochs} -> Accuracy: {round((nr_correct / images.shape[0]) * 100, 2)}% ({nr_correct}/{images.shape[0]})" )
    
    accuracy.append( nr_correct / images.shape[0] * 100 )
    nr_correct = 0  # Reset counter for next epoch.
    
    
    
#%% Plot Accuracy Versus Epoch
    
epochValues = list(range(epochs));
    
fig, ax = plt.subplots()
ax.plot( [x+1 for x in epochValues], accuracy)

ax.set(xlabel='Training Epoch', ylabel='Accuracy (%)', title='Accuracy Versus Training Epoch')
ax.grid()
fig.savefig('50 Epochs of Training MNIST February 14, 2021.png')
plt.show()


# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/simple_plot.html



#%% Predict Selected Digit
    
# index = int(input("Enter a number (0 - 59999): "))
index = 50  # Digit 3
    
img = images[index]
plt.imshow( img.reshape(28, 28), cmap="Blues" )

img.shape += (1,)  # Make the second dimension of the 1-Tuple (i.e. 1) explicit.


# Forward Popagation -> Input Layer to Hidden Layer
h_pre = b_i_h  +  w_i_h @ img.reshape(784, 1)
#
# b_i_h -> Array of type float64, 20-by-1
# w_i_h -> Array of type float64, 20-by-784
# img -> Array of float32, 784-by-1
#
# 20-by-1 + 20-by-784 * 784-by-1 -> 20-by-1 + 20-by-1 -> 20-by-1
#
h = 1 / ( 1 + np.exp(-h_pre) )  # Apply activation function.
    

# Forward Propagation -> Hidden Layer to Output Layer
o_pre = b_h_o  +  w_h_o @ h
#
# b_h_o -> Array of type float64, 10-by-1
# w_h_o -> Array of type float64, 10-by-20
# h -> Array of type float64, 20-by-1
#
# 10-by-1 + 10-by-20 * 20-by-1 -> 10-by-1 + 10-by-1 -> 10-by-1
#
o = 1 / (1 + np.exp(-o_pre))  # Apply activation function.


plt.title( f"Predicted Digit: {o.argmax()}" ); plt.show()


