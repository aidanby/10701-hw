from ANN import *
import numpy as np

Model = ANN(6, 4, 3, learning_rate=1, regularization=0.01)

# x = np.array([[1,0,1,0,1,0]])
# y = np.array([[0, 1, 0]])
# Model.alpha = np.array([[2,1,-1,-1,0,-2], [0,1,0,-1,1,3], [-1,2,1,3,1,-1], [1,3,4,2,-1,2]],
#                        dtype=float)
# Model.beta = np.array([[2,-2, 2, 1], [3, -1, 1, 2], [0, -1, 0, 1]], dtype=float)


x = np.array([[1,0,1,0,1,1]])
y = np.array([[0, 0, 1]])
Model.alpha = np.array([[1,2,-3,0,1,-3], [2,1,1,1,0,2], [3,2,2,2,2,1], [2,0,3,1,-2,2]],
                       dtype=float)
Model.beta = np.array([[1,2,-2,3], [2,-1,3,1], [3,1,-1,1]], dtype=float)


a = Model.alpha
b = a[ :,1:]


Model.fit(x, y, epoch=1)
print("Finished")