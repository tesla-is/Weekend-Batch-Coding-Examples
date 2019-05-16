[1, 2, 3] * 3
a = np.array([1, 2, 3])
a * 3


import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))

print("Time elapsed by Numpy array : ")
%time for a in range(10): my_arr2 = my_arr * 2
print("Time elapsed by Python list : ")
%time for a in range(10): my_list2 = my_list * 2

import time
    
for a in range(10):
    start = time.time()
    my_arr2 = my_arr * 2
    end = time.time()
    
print(end - start)

data = np.random.randn(2, 3)
data


data * 10
data + data

data.shape
data.dtype

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

arr1


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

arr2
arr2.ndim
arr2.shape

arr1.dtype
arr2.dtype

np.zeros(10)
np.zeros((2, 3))
np.empty((2, 3, 2))
np.arange(10, step = 0.01)



arr1 = np.array([1, 2, 3], dtype = np.float64)
arr1.dtype

arr2 = np.array([1, 2, 3], dtype = np.int32)
arr2.dtype

arr = np.array([1, 2, 3, 4, 5])
arr.dtype

float_arr = arr.astype(np.float64)
float_arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr.dtype

arr.astype(np.int32)

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype = np.string_)
numeric_strings.dtype

numeric_strings.astype(np.float64)
numeric_strings.dtype
numeric_strings

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50])

int_array.astype(calibers.dtype)

empty_uint32 = np.empty(8, dtype = 'u4')
empty_uint32

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr

arr * arr
arr - arr

1 / arr

arr ** 0.5

arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
arr

arr2 > arr

arr = np.arange(10)
arr

arr[5]

arr[5:8]

arr[5:8] = 12

arr

z = [0, 1, 2, 3, 4, 5]
z[2:4] = list([10, 10, 10])

arr_slice = arr[5:8]
arr_slice
arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr

test = arr[5:8].copy()
del(test)

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr2d[1]
arr2d[1, 1]
arr2d[1][1]


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d.shape
arr3d[0]
arr3d[1]

old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d

arr3d[1, 0]
x = arr3d[1]
x
x[0]
arr
arr[1:6]


arr2d
arr2d[:2, 1:]
arr2d[1, :2]
arr2d[:2, 2]
arr2d[:, :1]
arr2d[:2, 1:] = 0
arr2d

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)

names
data

names == "Bob"
data[names == "Bob"]
data[names == "Bob", 2:]
data[names == "Bob", 3]
names != "Bob"
data[~(names == "Bob")]
cond = names == "Bob"
data[~cond]

mask = (names == "Bob") | (names == "Will")
test_data = data[mask]
test_data
test_data = 0
data

data[data <0] = 0
data
data[names != "Joe"] = 7
data

arr = np.empty((8, 4))
arr

for i in range(8):
    arr[i] = i

arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8, 4))
arr

arr[[1, 5, 7, 2], [0, 3, 1, 2]]

arr[[1, 7]][:, [3, 2, 1, 0]]

# Boolean comparisons and Fancy Indexing always result in new arrays

arr = np.arange(15).reshape((3, 5))
arr

arr.T

arr = np.random.randn(6, 3)
arr

np.dot(arr.T, arr)

arr = np.arange(16).reshape(2, 2, 4)
arr
arr.T
arr.transpose((2, 0, 1))
arr.strides
arr
arr.swapaxes(1, 2)

arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x = np.random.randn(8)
y = np.random.randn(8)
x
y
np.maximum(x, y)

arr = np.random.randn(7) * 5
remainder, whole_part = np.modf(arr)
remainder
whole_part

arr
np.sqrt(arr)
np.sqrt(arr, arr)
arr

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
xs
ys

z = np.sqrt(xs ** 2 + ys ** 2)
z

import matplotlib.pyplot as plt
plt.imshow(z, cmap = plt.cm.gray)
plt.colorbar()
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

cond = np.array([True, False, True, True, False])

result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result

result = np.where(cond, xarr, yarr)
result

arr = np.random.randn(4, 4)
arr

arr > 0

np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)

arr = np.random.randn(5, 4)
arr
arr.mean()
arr.sum()
np.mean(arr)
np.sum(arr)
arr.mean(axis = 0)
arr.sum(axis = 0)

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr
arr.cumsum()

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr

arr.cumsum(axis = 0)
arr.cumsum(axis = 1)

arr.cumprod(axis = 0)
arr.cumprod(axis = 1)

arr = np.random.randn(100)
arr

(arr > 0).sum()

bools = np.array([False, False, True, False])
bools.any()
bools.all()

arr = np.random.rand(6)
arr
arr.sort()
arr

arr = np.random.randn(5, 3)
arr.sort(axis = 0)
arr.sort(axis = 1)
arr

arr[4].sort()
arr

large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')
np.savez('array_archive.npz', a = arr, b = arr)
arch = np.load('array_archive.npz')
arch['a']

np.savez_compressed('arrays_compressed.npz', a = arr, b = arr)
