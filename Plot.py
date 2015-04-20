import Mad8 as _Mad8

import pylab as _pl

def linearOptics(name = "ebds1") : 
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")
    [c,e] = r.readFile(name+".envelope","envel")    

def phaseAdvance(name = "ebds1") :
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")
    [c,e] = r.readFile(name+".envelope","envel")    

    suml = t.getColumn("suml")
    muX  = t.getColumn("mux")
    muY  = t.getColumn("muy")
    
    _pl.subplot(2,1,1)
    _pl.plot(suml,muX % _pl.pi)
    
    _pl.subplot(2,1,2)
    _pl.plot(suml,muY % _pl.pi)

def apertures(name = "ebds1") : 
    # read mad8 data
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")
    [c,e] = r.readFile(name+".envelope","envel")

    # calculate beam sizes
    sigmaX = _pl.sqrt(e.getColumn('s11'))
    sigmaY = _pl.sqrt(e.getColumn('s33'))
    
    # get apertures 
    aper   = _pl.array(map(float,c.getApertures(raw=False)))/10.0
    
    # suml 
    suml = t.getColumn('suml')


    _pl.plot(suml,sigmaX,label="$\sigma_{x}$")
    _pl.plot(suml,sigmaY,label="$\sigma_{y}$")
    _pl.plot(suml,aper, label="Aperture/10")
    _pl.xlabel("s [m]")
    _pl.ylabel("$\sigma_{x,y}$ [m]")
    _pl.legend(loc=0)

    _pl.savefig(name+"_apertures.pdf")



    

