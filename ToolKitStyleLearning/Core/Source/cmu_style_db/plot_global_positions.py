#Plot global positions
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ReadStylesDataset import read_cmu_style


clips, labels = read_cmu_style()
print (clips.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fr_no = 1 #clips.shape[1]
for i in range(100, 101):
	print (clips[i, :, :])
	fr = clips[i,:,:]
	ax.scatter(fr[:,0], fr[:,1], fr[:,2], c='b', marker='o')
plt.show()
