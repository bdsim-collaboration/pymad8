import pylab as _pl
import numpy as _np
import pymad8 as _m8


def testTrack(rmatFile, nparticle=10):
    r = _m8.Output(rmatFile, "rmat")
    r.subline('IEX', -1)
    s = Track(r)
    s.trackParticles(nparticle)
    return s


class Track:
    def __init__(self, rmat):
        self.rmat       = rmat
        self.nelement   = rmat.nrec
        # Turn rmat data into matrix
        self.rmatMatrix = _np.reshape(rmat.data[:, 0:36], (self.nelement, 6, 6))

    def trackParticles(self, nparticle):
        # store the number of particles tracked 
        self.nparticle = nparticle
        
        # Nparticle, Nelement phase space storage
        self.psVector = _np.zeros((nparticle, self.nelement, 6))

        # loop over particles
        for i in range(0, nparticle):
            print('Track.Simple.track> particle ', i)
            p = self.generate() 
            psv = self.trackParticle(p)
            self.psVector[i] = psv
    
    def generate(self):
        return _np.array([0.0,
                         0.0,
                         _pl.normal(0, 1e-6),
                         0.0,
                         0.0,
                         0.0], dtype=_np.double)

    def trackParticle(self, p):
        # Tracking vector 
        psVector   = _np.zeros((self.rmatMatrix.shape[0], 6))

        # Set first element
        i = 1 
        psVector[0] = p 
                
        # loop over all elements from start to end 
        for m in self.rmatMatrix[1:]:
            # psVector[i] = np.dot(m,psVector[i-1])
            psVector[i] = _np.dot(m, p)
            i = i+1

        return psVector

