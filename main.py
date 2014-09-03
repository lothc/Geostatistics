from computeVario import *


fid = open("pointData3Structures.dat","rb")
xyz = scipy.fromfile(file=fid, dtype=scipy.float32).reshape((50,4))
fid.close()
fid = open("FullRaster3Structures.dat","rb")
rasterReference = scipy.fromfile(file=fid, dtype=scipy.float32).reshape((50,50,2))
fid.close()

[y,x] = scipy.mgrid[0:50,0:50]
xyGrid = scipy.vstack((x.flatten(), y.flatten())).transpose()
cov = ExponentialCovariance([10.,10.],1.)
[sk,skvar] = KrigingEstimation(xyz[:,[0,1,2]], xyGrid, cov,["SK", xyz[:,2].mean()])
[ok,okvar] = KrigingEstimation(xyz[:,[0,1,2]], xyGrid, cov,["OK"])
[uk,ukvar] = KrigingEstimation(xyz[:,[0,1,2]], xyGrid, cov,["UK", XTrend])

pylab.figure()
pylab.subplot(221)
pylab.title('Reference')
pylab.imshow(rasterReference[:,:,0],interpolation='nearest')
pylab.colorbar()
pylab.subplot(222)
pylab.title('SK estimation')
pylab.imshow(sk.reshape((50,50)),interpolation='nearest')
pylab.colorbar()
pylab.subplot(223)
pylab.title('OK estimation')
pylab.imshow(ok.reshape((50,50)),interpolation='nearest')
pylab.colorbar()
pylab.subplot(224)
pylab.title('UK estimation')
pylab.imshow(uk.reshape((50,50)),interpolation='nearest')
pylab.colorbar()
pylab.show()