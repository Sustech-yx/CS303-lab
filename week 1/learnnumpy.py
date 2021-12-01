## numpy is an extended python library including
# a lot of operations on matrix and mathematical function library
# rename numpy as np for convenience

import numpy as np

## Array/Matrix Creation

arr=np.array([0,1,2,3,4,5]) # use a list to initialize a one-dimentional numpy array
type(arr)  # <class 'numpy.ndarray'>

arr.itemsize ## number of Bytes for each element, floating

arr = np.array([0,1,2,3,4,5], dtype='int16')  ## initialize with int type
arr.itemsize ## int type

arr.dtype    ## 'int16'

arr.astype('int16')
arr.astype('float32')

arr = np.array([[0,1,2],[2,3,4]]) # initialize a two-dimensional numpy array

arr = np.zeros((10,10))      # initialize a two-dimensional array with all elements equal to 0

arr = np.full((10,10),1)   # initialize a two-dimensional array with all elements equal to 1

arr.size   ## number of elements

arr.ndim   ## number of dimensions

arr.shape  ## size for each dimension

arr = np.arange(100) # initialize a one-dimensional numpy array with numbers from 0 to 99

arr2=arr.reshape(10,10)   # reshape arr to 10*10 two-dimensional array arr2

a = np.identity(3)   ## identity matrix

b = np.diag([1, 2, 3])  ##

c = np.diag([3, 4], 1)  ##

## Indexing and Slicing

arr = np.arange(12)

arr[0]  ## get the first element

slice1 = arr[:4]  # get from the first to the forth elements

slice2 = arr[7:10] # get from the eighth to the tenth elements

slice3 = arr[0:12:4] ## get from the first and the last elements with stepsize = 2

arr2[0][0]

arr2[0,:]   ## one row

arr2[0,0:9:2]  ## start, stop, step

arr2[-1,:]  ## last row

arr2[[0,2],:] ## first and third row

arr2[:,0]   ## first column

arr2[:,-1]  ## last colum

arr2[0:6,0:6]    ## submatrix

arr2.diagonal()   # get the diagonal elements of arr2

arr2.diagonal(offset=1) #

arr2.diagonal(offset=-1) #

np.diag(arr2)  ## alternative method

i = np.arange(10)

arr2[i,i]  # alternative method

a = np.arange(10)

np.vstack([arr2,a])  ## append a to arr2 as the last row

b = a.reshape(10,1)
np.hstack([arr2, b]) ## append a to arr2 as the last column

## mathematical operators

arr3 = arr2 + 1  # mathematical operation

arr4 = arr2 + arr3 #

arr5 = arr2*2     #

arr2**2           # exponential operation

arr2**(1/2)

arr2[arr2>50]    # get elements that are bigger than 50

arr2[arr2>50] = 10  # reset the value

np.max(arr2,axis=0) # the maximum value in each column

np.max(arr2,axis=1) # the maximum value in each row

np.sum(arr2,axis=0) # the sum of each column

np.sum(arr2,axis=1) # the sum of each row

np.mean(arr2)  ## mean value

np.var(arr2)   ## variance value

np.std(arr2)   ## standard deviation value

## reading from file and writing to file

np.genfromtxt('1.txt',delimiter=',')    ## get an array from 1.txt

np.savetxt('2.txt', arr2, fmt='%d', delimiter=',')  ## save an array to 2.txt with format %d


## practice
# 1. Create a 10*10 ndarray object, and the matrix boundary value is 1, and the rest are 0.
#
# 2. Create a 5*5 matrix with each row being 0 to 4
#
# 3. Create an 8*8 matrix and set it with 0,1 in a staggered form, like [[0,1],[1,0]])
#
# 4. in:    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#    out:   array([1, 3, 5, 7, 9])   (output all the odd numbers in the array)






























