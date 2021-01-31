

==================================================
Sunday, January 31, 2021

https://www.youtube.com/watch?v=9RN2Wr8xvro&feature=share&ab_channel=BotAcademy

https://www.datasciencecentral.com/m/blogpost?id=6448529%3ABlogPost%3A489568

Additional Notes:
1. You might’ve seen that we haven’t used the variable e at all. 
This is for two reasons. First, normally we would’ve used it to calculate ‘delta_o’, 
but due to some tricks, it is not needed here. Second, it is sometimes helpful to print the average error during training to see if it decreases.

2. To see how it performs on images not seen during training, you could only use just the first 50000 images for training and then analyze the results on the remaining 10000 samples. I haven’t done it in this video for simplicity. The accuracy, however, shouldn’t change that much.

3. It seems like some people have a hard time understanding the shape lines [e.g. x.shape += (1,)]. So let me try to explain:

To create a 1-tuple in python we need to write x=(1,). If we would just write x=(1), it gets converted to the integer 1 in Python.

Numpy introduces the shape attribute for arrays. Because the shape of a matrix has to be represented by a tuple like (2, 5) or (2, 4, 7), it is a good idea to represent a vector as a 1-tuple instead of an integer for consistency. So it is (X,).

If we want to use this vector in a matrix multiplication with a matrix, it doesn't work because you can't matrix multiply a vector with a matrix in numpy. So we need to add this 'invisible' second dimension of size 1. The line basically adds a (1,) vector to the shape of the (X,) vector which results in a matrix of size (1, X). That's also why it doesn't work with (2,) because that would require more values. For example (5,) and (1,5) both contain 5 values while (2,5) would contain 10 values.

I should've shown the shapes in the shape information box as (X,) instead of just X. I think that also made it more confusing.
 
 
