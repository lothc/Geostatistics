import scipy
import numpy
import pylab
#from scipy import linalg
from numpy import linalg
from math import cos,sin

class NuggetCovariance :
  ''' Nugger Effect Covariance model'''  
  def __init__(self,sill):
    self.c=sill
    
  def __call__(self,loc1,loc2):
    if loc1==loc2:
        return self.c
    else:
        return 0
    


class NuggetVariogram :
  ''' Nugger Effect Variogram model'''
  def __init__(self,sill):
    self.cov=NuggetCovariance(sill)
    
  def __call__(self,loc1,loc2):
    return self.cov.c - self.cov(loc1,loc2)


class SphericalCovariance : 
  ''' Spherical Covariance model'''  
  def __init__(self,range, sill, angle = 0.):
    ''' range is a list [maxRange, minRange], angle is from N clockwise '''
    self.a = range
    self.c = sill
    angle = (angle-90.)/180.0*scipy.pi
    scaling = range[0]/range[1]
    self.rot = scipy.array([[cos(angle), -sin(angle)],\
                           [scaling*sin(angle), scaling*cos(angle)]])
        
  def __call__(self, loc1, loc2):
    dist = scipy.dot(self.rot,(loc1-loc2))
    dist = scipy.sqrt(scipy.sum(dist*dist))
    if dist > self.a[0] : return 0.
    h  = dist/self.a[0]
    return  self.c * (1. -  1.5*h + 0.5*scipy.power(h, 3))

class SphericalVariogram :
  ''' Spherical variogram model''' 
  def __init__(self,range,sill, angle = 0.):
    self.cov = SphericalCovariance(range,sill,angle)
    
  def __call__(self, loc1, loc2):
    return  self.cov.c - self.cov(loc1,loc2)


class ExponentialCovariance :
  ''' Exponential Covariance model''' 
  def __init__(self,range, sill, angle = 0.):
    ''' range is a list [maxRange, minRange], angle is from N clockwise '''
    self.a = range
    self.c = sill
    angle = (angle-90.)/180.0*scipy.pi
    scaling = range[0]/range[1]
    self.rot = scipy.array([[cos(angle), -sin(angle)],
                           [scaling*sin(angle), scaling*cos(angle)]])
    #print self.rot
        
  def __call__(self, loc1, loc2):
    dist = scipy.dot(self.rot,(loc1-loc2))
    dist = scipy.sqrt(scipy.sum(dist*dist))
    h  = dist/self.a[0]
    return  self.c * scipy.exp(-3*h)


class ExponentialVariogram :
  ''' Exponential variogram model'''
  def __init__(self,range,sill, angle = 0.):
    self.cov = ExponentialCovariance(range,sill,angle)
    
  def __call__(self, loc1, loc2):
    return  self.cov.c - self.cov(loc1,loc2)

class GaussianCovariance :
  ''' Exponential Covariance model''' 
  def __init__(self,range, sill, angle = 0.):
    ''' range is a list [maxRange, minRange], angle is from N clockwise '''
    self.a = range
    self.c = sill
    angle = (angle-90.)/180.0*scipy.pi
    scaling = range[0]/range[1]
    print cos(angle),sin(angle)
    self.rot = scipy.array([[cos(angle), -sin(angle)],
                           [scaling*sin(angle), scaling*cos(angle)]])
        
  def __call__(self, loc1, loc2):
    dist = scipy.dot(self.rot,(loc1-loc2))
    dist = scipy.sqrt(scipy.sum(dist*dist))
    h  = dist/self.a[0]
    return  self.c * scipy.exp(-3*h*h)
  
  
class GaussianVariogram :
  ''' Gaussian Covariance model'''
  def __init__(self,range,sill, angle = 0.):
    self.cov = GaussianCovariance(range,sill,angle)
    
  def __call__(self, loc1, loc2):
    return  self.cov.c - self.cov(loc1,loc2)  
  
  
class NestedStructure :
  ''' Nested Covariance or Variogram model'''  
  def __init__(self,varioList):
    #varioList is a list of previously defined variogram models
    self.varioList=varioList
  def __call__(self,loc1,loc2):    
    v=0
    for vario in self.varioList:
      v+=vario(loc1,loc2)
    return v





class DualKrigingEstimation :
    ''' Dual Kriging Estimation'''  
    def __init__(self,data,covarianceModel,krigingType):
        self.data=data
        self.covarianceModel=covarianceModel
        self.krigingType=krigingType
        
    
        #Define functionalMean from previous KrigingEstimation
        functionalMean=[]
        c0=covarianceModel(scipy.zeros(2),scipy.zeros(2))
        if krigingType[0]=="SK":      
            skmean=krigingType[1]   
        elif krigingType[0]=="OK":      
            functionalMean.append(Constant)
        elif krigingType[0]=="UK":
            functionalMean.append(Constant)
            for f in krigingType[1:]:
                functionalMean.append(f)
        else:
            print "Kriging type not recognized"
            return None
        self.functionalMean=functionalMean
               
        #Build LHS matrix     
        size=data.shape[0]+len(functionalMean)
        lhs=scipy.zeros([size,size])  

        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                lhs[i,j]=covarianceModel(data[i,:],data[j,:])
            for j in range(data.shape[0],size):
                lhs[i,j]=functionalMean[j-data.shape[0]](data[i,0],data[i,1])
    
        for i in range(data.shape[0],size):
            for j in range(data.shape[0]):
                lhs[i,j]=functionalMean[i-data.shape[0]](data[j,0],data[j,1])
            
    
        self.lhs=lhs
          
        #Build RHS Dual vector 
        rhsDual=scipy.zeros(size)
        rhsDual[0:data.shape[0]]=data[:,2]

        self.rhsDual=rhsDual
    
        #Solve for the weights  
        self.weights=numpy.linalg.solve(self.lhs,self.rhsDual)
        self.b=self.weights[0:data.shape[0]]
        self.d=self.weights[data.shape[0]:size]
        
    
    def __call__(self,pointsToBeEstimated):


        for point in pointsToBeEstimated: #################


                
        #Build RHS vector (copy from previous KrigingEstimation)
        c0=self.covarianceModel(scipy.zeros(2),scipy.zeros(2))
        size=data.shape[0]+len(functionalMean)
        rhs=scipy.zeros(size)  

        for i in range(self.data.shape[0]):
            locData=self.data[i,:]
            rhs[i]=c0-self.covarianceModel(point,locData)###########################
        
        for i in range(data.shape[0],size):
            rhs[i]=self.functionalMean[i-self.data.shape[0]](point[0],point[1])
        
        






def Constant(x,y):
  return 1. 
      
def XTrend(x,y):
  return x 

def YTrend(x,y):
  return y

def XYTrend(x,y):
  return x*y  

def X2Trend(x,y):
  return x*x  

def Y2Trend(x,y):
  return y*y  



# EXTERNAL DRIFT

class DriftTrend:
    ''' Drift Trend'''  
    def __init__(self,driftArray):
        self.driftArray=driftArray #nx3 array (x,y,value)
    
    def __call__(self,x,y):       
        indX=self.driftArray[:,0]==x
        indY=self.driftArray[:,1]==y
        flag=0
        for i in range(driftArray.shape[0]):
            if indX[i]:
                if indY[i]:
                    flag=1
                    return self.driftArray[i,3]

        if flag==0:
            return "This point is not in the drift array"
        
        
        
        
        

def build_rhs_kriging_matrix(point,data,covarianceModel,functionalMean):
    ''' Calculate the right hand side vector from the kriging system.  The input is a scipy array.
    point: 1x2 array (x,y) (cf. pointsToBeEstimated)
    data : nx2 array (x,y, values)  (cf. dataPoint first 2 columns)
    covarianceModel : function or fucntor to compute the covariance (call: covarianceModel(c1,c2) )
    FunctionalMean : list of trend functions for the UK option (begins with 1 for OK/UK, empty for SK).
    '''
    c0=covarianceModel(scipy.zeros(2),scipy.zeros(2))
    size=data.shape[0]+len(functionalMean)
    output=scipy.zeros(size)  

    for i in range(data.shape[0]):
        locData=data[i,:]
        output[i]=c0-covarianceModel(point,locData)
    
    for i in range(data.shape[0],size):
        output[i]=functionalMean[i-data.shape[0]](point[0],point[1])
    
    return output
    
        

def build_lhs_kriging_matrix(data,covarianceModel,functionalMean):
    ''' Calculate the left hand side matrix of the kriging system.  The input is a scipy array.
    data : nx2 array (x,y, values)  (cf. dataPoint first 2 columns)
    covarianceModel : function or fucntor to compute the covariance (call: covarianceModel(c1,c2) )
    functionalMean : list of trend functions for the UK option (begins with 1 for OK/UK, empty for SK).
    '''
    c0=covarianceModel(scipy.zeros(2),scipy.zeros(2))
    size=data.shape[0]+len(functionalMean)
    output=scipy.zeros([size,size])  

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            output[i,j]=c0-covarianceModel(data[i,:],data[j,:])
        for j in range(data.shape[0],size):
            output[i,j]=functionalMean[j-data.shape[0]](data[i,0],data[i,1])
    
    for i in range(data.shape[0],size):
        for j in range(data.shape[0]):
            output[i,j]=functionalMean[i-data.shape[0]](data[j,0],data[j,1])
            
    
    return output



      
def KrigingEstimation(dataPoint, pointsToBeEstimated, covarianceModel, krigingType = ["OK"]) :
    ''' Calculate Simple, Ordinary or Universal Kriging.  The inputs are scipy array.
    dataPoint : nx3 array (x,y, values)  
    pointsToBeEstimated : mx2 array (x,y)
    covarianceModel : function or fucntor to compute the covariance (call: covarianceModel(c1,c2) )
    meanFunctionals : list with kriging type (SK, OK, UK) 
                      the UK option needs a list of functions to calculate the trend
    '''
    # CHECK INPUTS (ex: if UK, check if lenOfFunc,.........)
    functionalMean=[]
    c0=covarianceModel(scipy.zeros(2),scipy.zeros(2))
    
    if krigingType[0]=="SK":      
        skmean=krigingType[1]   
    elif krigingType[0]=="OK":      
        functionalMean.append(Constant)
    elif krigingType[0]=="UK":
        functionalMean.append(Constant)
        for f in krigingType[1:]:
            functionalMean.append(f)
    else:
        print "Kriging type not recognized"
        return None
    
    
    nData=dataPoint.shape[0]
    nUnknown=pointsToBeEstimated.shape[0]
    lhs=build_lhs_kriging_matrix(dataPoint[:,[0,1]],covarianceModel,functionalMean)  
    
    
    z=scipy.empty(nUnknown)
    kvar=scipy.empty(nUnknown)  
    
    
    for i in range(nUnknown):
        
        rhs=build_rhs_kriging_matrix(pointsToBeEstimated[i,:],dataPoint[:,[0,1]],covarianceModel,\
                               functionalMean)
        
        w=numpy.linalg.solve(lhs,rhs) #scipy.linalg.solve(lhs,rhs)
        
        
        if krigingType[0]=="SK":
            z[i]=scipy.dot(w,dataPoint[:,2]-skmean)+skmean
        else:
            z[i]=scipy.dot(w[0:nData],dataPoint[:,2])
        
        
        kvar[i]=c0-scipy.dot(w,rhs)

    
    
    return [z,kvar]
      
      
  
  
  
  
  
  
  
  
  
  
