import numpy as np
a = np.array([1, 2, 3]) # Create a rank 1 array 
print("One dimensional array a = ",a)
b = np.array([[1,2,3],[4,5,6]])
print("Two dimensional array b = ",b)
print("Size of the array: ",a.size)
print("Element at indices 0,1,2 : ",a[0], a[1], a[2]) 
a[0] = 5 # Change an element of the array 
print("Array after changing the element at index 0 : ",a) 
a = np.zeros((2,2)) # Create an array of all zeros 
print("An array of all zeros : ",a)
b = np.ones((1,2)) # Create an array of all ones 
print("An array of all ones : ",b)
c = np.full((2,2), 7) # Create a constant array 
print("A constant array : ",c)
d = np.eye (2) # Create a 2x2 identity matrix
print("A 2*2 identity matrix : ",d)
e = np.random.random((2,2)) # Create an array filled with random values 
print("An array with random values : ",e)