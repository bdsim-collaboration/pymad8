import pylab as pl 
import numpy as np 
import Mad8  as m8

def testTrack(rmatFile, nparticle = 10) :
    o = m8.OutputReader()     
    [c,r] = o.readFile(rmatFile,"rmat") 
    c.subline('IEX',-1) 
    r.subline('IEX',-1)
    s = Track(c,r)
    s.trackParticles(nparticle)
    return s

class Track : 
    def __init__(self, common, rmat) : 
        self.common     = common
        self.rmat       = rmat
        self.nelement   = rmat.data.shape[0]
        # Turn rmat data into matrix
        self.rmatMatrix = np.reshape(rmat.data[:,0:36],(self.nelement,6,6))

    def trackParticles(self, nparticle) : 
        # store the number of particles tracked 
        self.nparticle = nparticle
        
        # Nparticle, Nelement phase space storage
        self.psVector = np.zeros((nparticle,self.nelement,6))

        # loop over particles
        for i in range(0,nparticle) : 
            print 'Track.Simple.track> particle ',i
            p = self.generate() 
            psv = self.trackParticle(p)
            self.psVector[i] = psv
    
    def generate(self) : 
        return np.array([0.0,
                         0.0,
                         pl.normal(0,1e-6),
                         0.0,
                         0.0,
                         0.0],dtype=np.double)

    def trackParticle(self,p) :

        # Tracking vector 
        psVector   = np.zeros((self.rmatMatrix.shape[0],6))

        # Set first element
        i = 1 
        psVector[0] = p 
                
        # loop over all elements from start to end 
        for m in self.rmatMatrix[1:] :
#            psVector[i] = np.dot(m,psVector[i-1])
            psVector[i] = np.dot(m,p)
            i=i+1

        return psVector

