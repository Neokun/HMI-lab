import numpy as np

a = np.arange(6).reshape(3,2)
b = np.array(('a', 'b', 'c', 'd','e','f', 'g','h','i','k','l','m','A', 'B', 'C', 'D','E','F', 'G','H','I','K','L','M'), dtype=object)
b = b.reshape(2,4,3)
print b
print a
print a.shape, b.shape

c1 = np.tensordot(a,b,(1,0))

print c1.shape
print c1

a = np.arange(6).reshape(3,2)
b = np.array(('a', 'b', 'c', 'd','e','f', 'g','h','i','k','l','m','A', 'B', 'C', 'D','E','F', 'G','H','I','K','L','M'), dtype=object)
b = b.reshape(2,12)
print b
print a
print a.shape, b.shape

c2 = np.dot(a,b)

print c2.shape
print c2

print (10*'-')
c3 = c2.reshape(3,4,3)
print c3
print c1.reshape(3,12)
