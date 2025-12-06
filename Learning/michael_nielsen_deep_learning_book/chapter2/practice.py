import numpy as np


# Creates an array later wise, 2 items in the outer most layer, then followed by 3 in each of the element in the 
# outermost layer, followed by 4 inside the innermost layer.
print (np.zeros((2, 3, 4)))

A = [[1, 2], [2, 2]]
W = [[3, 3 ], [2, 1], [1, 3]]
print("A before it's converted to a numpy array: ")
print(A)
print("W before it's converted to a numpy array: ")
print(W)

A = np.array(A)
W = np.array(W)
print("A as a numpy array is: ")
print(A)
print("W is a numpy array is : ")
print(W)
print (np.dot(W, A))

random_np = np.array([[1, 2], [2, 3], [3, 4]])
print(np.sum(random_np, axis = 1))




