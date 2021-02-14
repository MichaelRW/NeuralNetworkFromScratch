

#%% Modules

import numpy as np
import pathlib  # Object-oriented Filesystem Paths



#$$ Function Definition -> get_mnist

def get_mnist():
    
    with np.load( f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz" ) as f:
        images, labels = f["x_train"], f["y_train"]
        
    images = images.astype("float32") / 255  # Cast a copy of the images array to 32-bit floating point representation.
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    
    labels = np.eye(10)[labels]
    
    return images, labels



#%% Validate returned variables -> images, labels
    
images, labels = get_mnist()
    
# print( type(images) )
# print( images.shape )  # 60000-by-784 -> Each row contains a 1-Tuple of 784 values (i.e. 28-by-28).

# print( type(labels) )
# print( labels.shape )  # 60000-by-1 -> Each row contains an array of zeros/ones that indicate the labeled digit (i.e. 0-9).


