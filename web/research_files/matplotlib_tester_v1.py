# matplotlib tester
# follows http://stackoverflow.com/questions/4981815/how-to-remove-lines-in-a-matplotlib-plot


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
d = np.random.random((100,100))
e = range(10) * 10
e2 = np.reshape(e, (1,100))
d = d + e2
fig = plt.figure(figsize=(3.5,3.5), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')
lines = ax.plot(d[:,0],d[:,1],d[:,2],'o', markersize=7, color='red', alpha=0.5)
lines2 = ax.plot(d[:,3],d[:,4],d[:,5],'o', markersize=7, color='blue', alpha=0.5)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_zticks([])
#zz = ax.plot(d[d[:,3]==num][:,0],d[d[:,3]==num][:,1],d[d[:,3]==num][:,2],'o', markersize=7, color=colors[num], alpha=0.3)
print "lines[0]:", lines[0]
#lines.pop(0)
print 'lines:', lines
print 'ax:', ax
print 'ax.lines:', ax.lines[0]
ax.lines.remove(lines[0])
#ax.lines.pop(0)
plt.show()
#print zz[0]
'''

a = np.random.random((3,100))
b = np.array(range(10)*10)
b = b.reshape(1,100)
c = np.concatenate((a,b))
d = c[3,:]==2

fig = plt.figure(figsize=(3.5,3.5), tight_layout=True)
ax = fig.add_subplot(111, projection='3d')
lines = ax.plot(c[0,:], c[1,:], c[2,:], 'o', markersize=7, color='grey', alpha=0.5)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_zticks([])

fig.savefig('fig-1.png')
print 'figure 1 saved'

for num in range(3):
	d = c[3,:]==num	
	lines2 = ax.plot(c[0,:][d], c[1,:][d], c[2,:][d], 'o', markersize=7, color='red', alpha=0.5)
	#print "lines[0]:", lines[0]
	#print "lines2[0]:", lines2[0]
	#print 'ax:', ax
	#print 'ax.lines:', ax.lines
	#print 'ax.lines[0]:', ax.lines[0]
	#print 'ax.lines[1]:', ax.lines[1]
	
	filename = 'fig-' + str(num)
	fig.savefig(filename + '_before.png')	
	print 'figure ', num, '_before saved'
	print 'ax.lines:', ax.lines

	ax.lines.pop(1)

	fig.savefig(filename + '_after.png')
	print 'figure ', str(num), '_after saved'
	print 'ax.lines:', ax.lines
	# plt.show()

