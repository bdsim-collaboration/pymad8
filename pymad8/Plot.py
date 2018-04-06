import Output as _Output

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

_matplotlib.projections.register_projection(_My_Axes)

def AddMachineLatticeToFigure(figure, mad8opt, tightLayout=True):
    """
    Add a diagram above the current graph in the figure that represents the
    accelerator based on a madx twiss file in tfs format.

    Note you can use matplotlib's gcf() 'get current figure' as an argument.

    >>> pymadx.Plot.AddMachineLatticeToFigure(gcf(), 'afile.tfs')

    """    

    axs = figure.get_axes() # get existing graph

    axoptics = figure.get_axes()[0]
    _AdjustExistingAxes(figure, tightLayout=tightLayout)
    axmachine = _PrepareMachineAxes(figure)
    
    _DrawMachineLattice(axmachine,mad8opt)

    # put callbacks for linked scrolling 
    def MachineXlim(ax):
        axmachine.set_autoscale_on(False)
        axoptics.set_xlim(axmachine.get_xlim())

    def Click(a):
        if a.button == 3:
            try:
                print a.xdata,mad8opt['twiss'].nameFromNearestS(a.xdata)
            except ValueError:
                pass # don't complain if the S is out of bounds        

    MachineXlim(axmachine)
    axmachine.callbacks.connect('xlim_changed', MachineXlim)
    figure.canvas.mpl_connect('button_press_event', Click)

def _PrepareMachineAxes(figure):
    # create new machine axis with proportions 6 : 1
    axmachine = figure.add_subplot(911, projection="_My_Axes")
    axmachine.set_facecolor('none') # make background transparent to allow scientific notation
    _SetMachineAxesStyle(axmachine)
    return axmachine
    
def _AdjustExistingAxes(figure, fraction=0.9, tightLayout=True):
    """
    Fraction is fraction of height all subplots will be after adjustment.
    Default is 0.9 for 90% of height. 
    """
    # we have to set tight layout before adjustment otherwise if called
    # later it will cause an overlap with the machine diagram
    if (tightLayout):
        _plt.tight_layout()

    axs = figure.get_axes()

    for ax in axs:
        bbox = ax.get_position()
        bbox.y0 = bbox.y0 * fraction
        bbox.y1 = bbox.y1 * fraction
        ax.set_position(bbox)

def _SetMachineAxesStyle(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _DrawMachineLattice(axesinstance, mad8opt):
    ax = axesinstance
    m8 = mad8opt
    
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
        ax.plot([e['suml']-e['l'],e['l']-e['l']],[-0.2,0.2],'-',color=color,alpha=alpha)

    ax.plot([m8['twiss'].smin,m8['twiss'].smax],[0,0],'k-',lw=1)
    ax.set_ylim(-0.2,0.2)

    # loop over elements 
    for i in range(0,m8['comm'].nele) : 
        c = m8['comm']
        t = m8['twiss']

        # need to add s to element
        e = c.getRowByIndex(i)
        e['suml'] = t.getRowByIndex(i)['suml']

        if e['type'] == 'quad':
            DrawQuad(e, u'#d10000') #red
        elif e['type'] == 'rben':
            DrawBend(e, u'#0066cc') #blue
        elif e['type'] == 'sben':
            DrawBend(e, u'#0066cc') #blue
        elif e['type'] == 'kick':
            DrawRect(e, u'#4c33b2')
        elif e['type'] == 'hkic':
            DrawRect(e, u'#4c33b2') #purple
        elif e['type'] == 'vkic':
            DrawRect(e, u'#ba55d3')            
        elif e['type'] == 'rcol':
            DrawRect(e,'k')
        elif e['type'] == 'ecol':
            DrawRect(e,'k')
        elif e['type'] == 'sext':
            DrawHex(e, u'#ffcc00')
        elif e['type'] == 'octu':
            DrawHex(e, u'#00994c') #green
        elif e['type'] == 'lcav':
            DrawRect(e, u'#000000',0.1)
        elif e['type'] == 'sole':
            DrawRect(e, u'#000000',0.1)
        elif e['type'] == 'drif':
            pass
        elif e['type'] == 'moni':
            pass
        elif e['type'] == 'mark':
            pass
        else :
            pass
            #print 'not drawn',e['type']
        
# ============================================================================
# Below is old
# ============================================================================

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

def setCallbacks(figure, axm, axplot,twiss) :
    #put callbacks for linked scrolling
    def MachineXlim(axm): 
        axm.set_autoscale_on(False)
        for ax in axplot :
            ax.set_xlim(axm.get_xlim())

    def Click(a) : 
        if a.button == 3 : 
            print 'Closest element: ',twiss.nameFromNearestS(a.xdata)

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

    
def linearOptics(twissfile = "ebds1") : 
    r = _Output.OutputReader() 
    [c,t] = r.readFile(twissfile,"twiss")
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
    ax2.relim()
    ax2.autoscale_view(False,False,True)
    sqrtDisX = t.getColumn("dx")
    _plt.plot(t.getColumn("suml"),sqrtDisX,"b",label="$\\eta_{x}$")
    sqrtDisY = t.getColumn("dy")
    _plt.plot(t.getColumn("suml"),sqrtDisY,"g--",label="$\\eta_{y}$")
    _plt.xlabel("S [m]")
    _plt.ylabel("$\eta_{x,y}$ [m]")
    _plt.legend(loc=2)


    setCallbacks(figure,ax0,[ax1,ax2],t)

#    _plt.savefig(name+"_linear.pdf")
    


def phaseAdvance(twissfile = "ebds1") :
    r = _Output.OutputReader() 
    [c,t] = r.readFile(twissfile,"twiss")

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

    setCallbacks(figure,ax0,[ax1,ax2],t)

#    _pl.savefig(name+"_phase.pdf")

def dispersion(twissfile = "ebds1") : 
    r = _Output.OutputReader() 
    [c,t] = r.readFile(twissfile,"twiss")

    suml = t.getColumn("suml")
    etaX  = t.getColumn("dx")
    etaY  = t.getColumn("dy")
    
    figure = _plt.figure(1)
    gs  = _plt.GridSpec(3,1,height_ratios=[1,3,3])
    ax0 = figure.add_subplot(gs[0],projection="_My_Axes")
    drawMachineLattice(c,t)     

    ax1 = _plt.subplot(gs[1]) 
    _pl.plot(suml,etaX)
    _pl.ylabel("$\eta_{x}$")

    ax2 = _plt.subplot(gs[2]) 
    _pl.plot(suml,etaY)
    _pl.xlabel("S [m]")
    _pl.ylabel("$\eta_{y}$")

    setCallbacks(figure,ax0,[ax1,ax2],t)

def dispersionPrime(twissfile = "ebds1") : 
    r = _Output.OutputReader() 
    [c,t] = r.readFile(twissfile,"twiss")

    suml = t.getColumn("suml")
    etapX  = t.getColumn("dpx")
    etapY  = t.getColumn("dpy")
    
    figure = _plt.figure(1)
    gs  = _plt.GridSpec(3,1,height_ratios=[1,3,3])
    ax0 = figure.add_subplot(gs[0],projection="_My_Axes")
    drawMachineLattice(c,t)     

    ax1 = _plt.subplot(gs[1]) 
    _pl.plot(suml,etapX)
    _pl.ylabel("$\eta_{x}\prime$")

    ax2 = _plt.subplot(gs[2]) 
    _pl.plot(suml,etapY)
    _pl.xlabel("S [m]")
    _pl.ylabel("$\eta_{y}\prime$")

    setCallbacks(figure,ax0,[ax1,ax2],t)

def apertures(twissfile="ebds1", envelfile="ebds1") : 
    # read mad8 data
    r = _Output.OutputReader() 
    [c,t] = r.readFile(twissfile,"twiss")
    [c,e] = r.readFile(envelfile,"envel")

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

    setCallbacks(figure,ax0,[ax1,ax2],t)

#    _plt.savefig(name+"_apertures.pdf")


def energy(twissfile = "ebds1") : 
    # read mad8 data
    r = _Output.OutputReader() 
    [c,t] = r.readFile(twissfile,"twiss")

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
    
    setCallbacks(figure,ax0,[ax1],t)
    
def survey(surveyfile = "ebds1") : 
    # read mad8 data
    r = _Output.OutputReader()
    [c,s] = r.readFile(surveyfile,"survey") 
    
    # get suml
    suml = s.getColumn('suml')
    
    print suml
