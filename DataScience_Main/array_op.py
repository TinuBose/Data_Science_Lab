import numpy as np

arr1=np.array([1,2,3,4,5])
arr2=np.array([6,7,8,9,10])
result_add=arr1+arr2
print("sum = ",result_add)
result_multiply=arr1*arr2
print("product = ",result_multiply)
print("mean = ",np.mean(result_add))
print("max = ",np.max(result_multiply))