import Mad8 as _Mad8

import pylab as _pl
import numpy as _np
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches

class _My_Axes(_matplotlib.axes.Axes):
    """
    Inherit matplotlib.axes.Axes but override pan action for mouse.
    Only allow horizontal panning - useful for lattice axes.
    """
    name = "_My_Axes"
    def drag_pan(self, button, key, x, y):
        _matplotlib.axes.Axes.drag_pan(self, button, 'x', x, y) # pretend key=='x'

#register the new class of axes
_matplotlib.projections.register_projection(_My_Axes)

def setCallbacks(figure, axm, axplot) :
    #put callbacks for linked scrolling
    def MachineXlim(axm): 
        axm.set_autoscale_on(False)
        axplot.set_xlim(axm.get_xlim())

    def Click(a) : 
        if a.button == 3 : 
            print 'Closest element: ',tfs.NameFromNearestS(a.xdata)

    axm.callbacks.connect('xlim_changed', MachineXlim)
    figure.canvas.mpl_connect('button_press_event', Click) 

def drawMachineLattice(mad8c, mad8t) : 
    ax = _plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #NOTE madx defines S as the end of the element by default
    #define temporary functions to draw individual objects
    #Not sure about mad8
    def DrawBend(e,color='b',alpha=1.0):
        br = _patches.Rectangle((e['suml']-e['l'],-0.1),e['l'],0.2,color=color,alpha=alpha)
        ax.add_patch(br)
    def DrawQuad(e,color='r',alpha=1.0):
        if e['k1'] > 0 :
            qr = _patches.Rectangle((e['suml']-e['l'],0),e['l'],0.2,color=color,alpha=alpha)
        elif e['k1'] < 0: 
            qr = _patches.Rectangle((e['suml']-e['l'],-0.2),e['l'],0.2,color=color,alpha=alpha)
        else:
            #quadrupole off
            qr = _patches.Rectangle((e['suml']-e['l'],-0.1),e['l'],0.2,color='#B2B2B2',alpha=0.5) #a nice grey in hex
        ax.add_patch(qr)
    def DrawHex(e,color,alpha=1.0):
        s = e['suml']-e['l']
        l = e['l']
        edges = _np.array([[s,-0.1],[s,0.1],[s+l/2.,0.13],[s+l,0.1],[s+l,-0.1],[s+l/2.,-0.13]])
        sr = _patches.Polygon(edges,color=color,fill=True,alpha=alpha)
        ax.add_patch(sr)
    def DrawRect(e,color,alpha=1.0):
        rect = _patches.Rectangle((e['suml']-e['l'],-0.1),e['l'],0.2,color=color,alpha=alpha)
        ax.add_patch(rect)
    def DrawLine(e,color,alpha=1.0):
        ax.plot([e['suml']-e['l'],e['suml']-e['l']],[-0.2,0.2],'-',color=color,alpha=alpha)

    # plot beam line 
    ax.plot([0,mad8t.getRowByIndex(-1)['suml']],[0,0],'k--',lw=1)
    ax.set_ylim(-0.5,0.5)

    # loop over elements and Draw on beamline
    for i in range(0,mad8c.getNElements(),1) :
        element = mad8c.getRowByIndex(i)
        element['suml'] = mad8t.getRowByIndex(i)['suml']

        kw = element['type']
        if kw == 'quad': 
            DrawQuad(element)
        elif kw == 'rbend': 
            DrawBend(element)
        elif kw == 'sben': 
            DrawBend(element)
        elif kw == 'rcol': 
            DrawRect(element,'k')
        elif kw == 'ecol': 
            DrawRect(element,'k')
        elif kw == 'sext':
            DrawHex(element,'#ffcf17') #yellow
        elif kw == 'octu':
            DrawHex(element,'g')
        elif kw == 'drif':
            pass
        elif kw == 'mult':
            element['l'] = 0.0
            DrawLine(element,'grey',alpha=0.5)
        elif kw == 'mark' :
            pass
        elif kw == '' :
            pass
        else:
            #unknown so make light in alpha
            if element['l'] > 1e-1:
                DrawRect(element,'#cccccc',alpha=0.1) #light grey
            else:
                #relatively short element - just draw a line
                DrawLine(element,'#cccccc',alpha=0.1)

    
def linearOptics(name = "ebds1") : 
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")
#    [c,e] = r.readFile(name+".envelope","envel")    

    figure = _plt.figure(1)

    gs  = _plt.GridSpec(3,1,height_ratios=[1,3,3])
    ax0 = figure.add_subplot(gs[0],projection="_My_Axes")
    drawMachineLattice(c,t)    

    ax1 = _plt.subplot(gs[1]) 
    sqrtBetX = _pl.sqrt(t.getColumn("betx"))
    _plt.plot(t.getColumn("suml"),sqrtBetX,"b",label="$\\beta_{x}^{1/2}$")
    sqrtBetY = _pl.sqrt(t.getColumn("bety"))
    _plt.plot(t.getColumn("suml"),sqrtBetY,"g--",label="$\\beta_{y}^{1/2}$")
    _plt.ylabel("$\\beta_{x,y}^{1/2}$ $[{\mathrm m}]^{1/2}$")
    _plt.legend(loc=2)

    ax2 = _plt.subplot(gs[2]) 
    ax2.set_autoscale_on(True)
    ax2.autoscale_view(True,False,True)
    sqrtDisX = t.getColumn("dx")
    _plt.plot(t.getColumn("suml"),sqrtDisX,"b",label="$\\eta_{x}$")
    sqrtDisY = t.getColumn("dy")
    _plt.plot(t.getColumn("suml"),sqrtDisY,"g--",label="$\\eta_{y}$")
    _plt.xlabel("S [m]")
    _plt.ylabel("$\eta_{x,y}$ [m]")
    _plt.legend(loc=2)


    _plt.savefig(name+"_linear.pdf")
    
    setCallbacks(figure,ax0,ax1)
    setCallbacks(figure,ax0,ax2)

def phaseAdvance(name = "ebds1") :
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")
    [c,e] = r.readFile(name+".envelope","envel")    

    suml = t.getColumn("suml")
    muX  = t.getColumn("mux")
    muY  = t.getColumn("muy")
    
    figure = _plt.figure(1)
    gs  = _plt.GridSpec(3,1,height_ratios=[1,3,3])
    ax0 = figure.add_subplot(gs[0],projection="_My_Axes")
    drawMachineLattice(c,t)    

    ax1 = _plt.subplot(gs[1]) 
    _pl.plot(suml,muX % _pl.pi)
    _pl.ylabel("$\mu_{x}$")

    ax2 = _plt.subplot(gs[2]) 
    _pl.plot(suml,muY % _pl.pi)
    _pl.xlabel("S [m]")
    _pl.ylabel("$\mu_{y}$")

    setCallbacks(figure,ax0,ax1)
    setCallbacks(figure,ax0,ax2)

    _pl.savefig(name+"_phase.pdf")

def apertures(name = "ebds1") : 
    # read mad8 data
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")
    [c,e] = r.readFile(name+".envelope","envel")

    # calculate beam sizes
    sigmaX = _pl.sqrt(e.getColumn('s11'))
    sigmaY = _pl.sqrt(e.getColumn('s33'))
    
    # get apertures 
    aper    = _pl.array(map(float,c.getApertures(raw=False)))
    aper    = aper/aper.max()*max(sigmaX.max(),sigmaX.max())
    aperMax = aper.max()*_pl.ones(len(aper))
        
    # suml 
    suml = t.getColumn('suml')

    figure = _pl.figure(1) 
    figure.subplots_adjust(left=0.15)
    gs  = _plt.GridSpec(3,1,height_ratios=[1,3,3])

    ax0 = figure.add_subplot(gs[0],projection="_My_Axes")
    drawMachineLattice(c,t)          

    ax1 = _plt.subplot(gs[1]) 
    _plt.plot(suml,sigmaX,"b",label="$\sigma_{x}$")
    _plt.plot(suml,-sigmaX,"b")
    _plt.fill_between(suml, sigmaX, -sigmaX, color="b", alpha=0.2)
    _plt.fill_between(suml, aper, aperMax, color="k", alpha=0.2)
    _plt.fill_between(suml, -aper, -aperMax, color="k", alpha=0.2)
    _plt.ylim(-sigmaX.max(),sigmaX.max())
    _plt.ylabel("$\sigma_{x}$ [m]")


    ax2 = _plt.subplot(gs[2]) 
    _plt.plot(suml,sigmaY,"b",label="$\sigma_{y}$")
    _plt.plot(suml,-sigmaY,"b")
    _plt.fill_between(suml, sigmaY, -sigmaY, color="b", alpha=0.2)
    _plt.fill_between(suml, aper, aperMax, color="k", alpha=0.2)
    _plt.fill_between(suml, -aper, -aperMax, color="k", alpha=0.2)
    _plt.xlabel("s [m]")
    _plt.ylabel("$\sigma_{y}$ [m]")
    _plt.ylim(-sigmaX.max(),sigmaX.max())

    setCallbacks(figure,ax0,ax1)
    setCallbacks(figure,ax0,ax2)

    _plt.savefig(name+"_apertures.pdf")


def energy(name = "ebds1") : 
    # read mad8 data
    r = _Mad8.OutputReader() 
    [c,t] = r.readFile(name+".twiss","twiss")

    # get suml 
    suml = t.getColumn('suml')[1:]
    
    figure = _pl.figure(1)
    figure.subplots_adjust(left=0.15)
    gs  = _plt.GridSpec(2,1,height_ratios=[1,6])
    
    ax0 = figure.add_subplot(gs[0],projection="_My_Axes")
    drawMachineLattice(c,t)      

    ax1 = _plt.subplot(gs[1]) 
    e   = c.getColumn("E")
    _plt.plot(suml,e,"b",label="$E$")
    _plt.xlabel("$s$ [m]")
    _plt.ylabel("$E$ [GeV]")
    _plt.legend()
    
    setCallbacks(figure,ax0,ax1)
    
def survey(name = "ebds1") : 
    # read mad8 data
    r = _Mad8.OutputReader()
    [c,s] = r.readFile(name+".survey","survey") 
    
    # get suml
    suml = s.getColumn('suml')
    
    print suml
