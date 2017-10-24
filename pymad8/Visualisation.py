import pylab as pl 
import pymad8 as m8 
import matplotlib as mpl
import matplotlib.patches as mpt
import matplotlib.pyplot  as plt

def testOneDim() : 
    o = m8.Output.OutputReader()
    [c,s] = o.readFile("./test/atf_v5.1/survey.tape","survey")
    od = OneDim(c,s,False)
    od.plot()
    return od    

def testTwoDim() : 
    o = m8.Output.OutputReader()
    [c,s] = o.readFile("./test/atf_v5.1/survey.tape","survey")
    td = TwoDim(c,s,False,False,True)
    td.plot()
    return td

def transformedRect(xyc, dx, dy, theta) : 
    x = xyc[0]
    y = xyc[1]

    # basic rectangle
    xy = pl.array([[-dx/2.0, -dy/2.0],
                   [-dx/2.0, +dy/2.0],
                   [+dx/2.0, +dy/2.0],
                   [+dx/2.0, -dy/2.0]])

    # transform to correct location 
    p  = transformedPoly(xy,xyc,theta)
    
    return p

def transformedPoly(xy, xyc, theta) :
    # Rotate in place
    r = pl.array([[ pl.cos(theta),-pl.sin(theta)],
                  [ pl.sin(theta), pl.cos(theta)]])
    xy = xy.dot(r)

    # Translate to new centre 
    xy = xy + xyc 
    p = mpt.Polygon(xy)

    # Return transformed poly
    return p

class OneDim :
    def __init__(self, common, survey, debug) :
        self.common = common
        self.survey = survey 

        self.x    =  self.survey.data[:,self.survey.keys['x']]
        self.y    =  self.survey.data[:,self.survey.keys['y']]
        self.z    = -self.survey.data[:,self.survey.keys['z']]
        self.suml =  self.survey.data[:,self.survey.keys['suml']]

        self.debug    = debug 
        #self.annotate = annotate 
        self.quadWidth = 0.1
        self.bendWidth = 0.1
        self.sextWidth = 0.1

        self._offcolour = '0.9'
        self._no_colour = '0.2'

    def plot(self,colour=True) : 
        s = self.survey.data[:,self.survey.keys['suml']]
        z = pl.zeros(s.shape)

        # plot beam line
        pl.plot(s,z,'k-')
        pl.ylim(-1.5,1.5)

        # Draw specific types of element
        self.drawElements("QUAD",colour) 
        self.drawElements("SBEN",colour) 
        self.drawElements("RBEN",colour)
        self.drawElements("SEXT",colour)

    def drawElements(self,type,colour=True) : 
        if self.debug : 
            print 'pymad8.Visualisation.OneDim.drawElements>'

        ilist = self.common.findByType(type) 
        for i in ilist : 
            self.drawElement(i,colour)
    
    def drawElement(self,elem,colour=True) : 
        if self.debug : 
            print 'pymad8.Visualisation.OneDim.drawElement>',elem
            print 'pymad8.Visualisation.OneDim.drawElement> Use Colour -> ',colour

        # find element if string
        if type(elem) == str : 
            i = self.common.findByName(name)[0]
        else :
            i = elem

        t    = self.common.type[i] 
        n    = self.common.name[i]
        c    = self.common.data[i]
        s    = self.survey.data[i]
        suml = self.suml[i]

        if t == 'QUAD' : 
            self.drawQuad(c,s,suml,colour)
        elif t == 'MULT' : 
            self.drawMult(c,s,suml,colour)
        elif t == 'SBEN' : 
            self.drawBend(c,s,suml,colour)
        elif t == 'RBEN' : 
            self.drawBend(c,s,suml,colour)
        elif t == 'SEXT' : 
            self.drawSext(c,s,suml,colour)
        elif t == 'HKIC' :
            self.drawHkic(c,s,suml,colour)
        elif t == 'VKIC' : 
            self.drawVkic(c,s,suml,colour)
        elif t == 'MONI' : 
            self.drawMoni(c,s,suml,colour)
        elif t == 'WIRE' : 
            self.drawWire(c,s,suml,colour)
        elif t == 'PROF' : 
            self.drawProf(c,s,suml,colour)
        elif t == 'INST' : 
            self.drawInst(c,s,suml,colour)
        elif t == 'MARK' : 
            self.drawMark(c,s,suml,colour)
        else :
            print 'pymad8.Visualisation.OneDim> Type not known'

        # Annotate element

    def drawQuad(self,c,s,suml,colour=True) : 
        ql = c[m8.Output.Common.keys['quad']['l']]
        qk = c[m8.Output.Common.keys['quad']['k1']]
        poscolour = 'r'
        negcolour = 'b'
        if colour == False:
                poscolour,negcolour = self._no_colour,self._no_colour
        
        if qk > 0 :
            qr = mpt.Rectangle((suml-ql/2.0,0),
                               ql,self.quadWidth,color=poscolour)
        elif qk < 0 : 
            qr = mpt.Rectangle((suml-ql/2.0,-self.quadWidth),
                               ql,self.quadWidth,color=negcolour)            
        elif qk == 0 : 
            qr = mpt.Rectangle((suml-ql/2.0,-self.quadWidth/2.0),
                               ql,self.quadWidth,color=self._offcolour)            
            
        ax = plt.gca()            
        ax.add_patch(qr)

    def drawMult(self,c,s,suml,colour=True) :
        pass

    def drawBend(self,c,s,suml,colour=True) : 
        #colour argument unused but added for compliance
        bl = c[m8.Output.Common.keys['sben']['l']]
        br = mpt.Rectangle((suml-bl/2.0,-self.bendWidth/2.0),
                           bl,self.bendWidth,color='k')
        ax = plt.gca()            
        ax.add_patch(br)

    def drawSext(self,c,s,suml,colour=True) : 
        sl = c[m8.Output.Common.keys['sextupole']['l']]
        ax = plt.gca()
        #original but looks like thin lens for most lattices rather
        #than hexagon
        #sh = mpt.RegularPolygon((suml-sl/2.0,0),6,
        #                        sl,color='g')
        if colour == True:
            sext_colour = 'g'
        else:
            sext_colour = 'grey'
        
        sh = mpt.Rectangle((suml-sl/2.0,-self.bendWidth/2.0),
                           sl,self.bendWidth,color=sext_colour) 
        ax.add_patch(sh)        

    def drawHkic(self,c,s,suml,colour=True) :
        pass

    def drawVkic(self,c,s,suml,colour=True) :
        pass
        
    def drawMoni(self,c,s,suml,colour=True) :
        pass
    
    def drawWire(self,c,s,suml,colour=True) :
        pass

    def drawProf(self,c,s,suml,colour=True) :
        pass

    def drawInst(self,c,s,suml,colour=True) :
        pass
    
    def drawMark(self,c,s,suml,colour=True) :
        pass


class TwoDim : 
    def __init__(self,common,survey, debug = False, annotate = False, fancy = False) :
        self.debug    = debug
        self.annotate = annotate
        self.fancy    = fancy
        self.common   = common
        self.survey   = survey
        self.quadWidth= 0.75

    def plot(self,event=None) : 
        print 'Visualisation.TwoDim.plot>'
        self.x = self.survey.data[:,self.survey.keys['x']]
        self.y = self.survey.data[:,self.survey.keys['y']]
        self.z = -self.survey.data[:,self.survey.keys['z']]

        self.f  = plt.figure()
#        self.rec = self.f.canvas.mpl_connect('draw_event',self.plotUpdate)        

        xmin = self.x.min()
        xmax = self.x.max()
        zmin = self.z.min()
        zmax = self.z.max()

        self.ax = plt.gca()
        self.ax.clear()        

        pl.plot(self.z,self.x,'--')        
        pl.xlim(zmin-10,zmax+10)
        pl.ylim(xmin-10,xmax+10)        

        self.drawElements("QUAD") 

    def plotUpdate(self,event) : 
        print 'Visualisation.TwoDim.plotUpdate>'
        self.drawElements("QUAD")

    def drawElements(self,type) : 
        if self.debug : 
            print 'pymad8.Visualisation.TwoDim.drawElements>'

        ilist = self.common.findByType(type) 
        for i in ilist : 
            self.drawElement(i)

    def drawElement(self,elem) : 
        if self.debug : 
            print 'pymad8.Visualisation.TwoDim.drawElement>',elem

        # find element if string
        if type(elem) == str : 
            i = self.common.findByName(name)[0]
        else :
            i = elem

        t = self.common.type[i] 
        n = self.common.name[i]
        c = self.common.data[i]
        s = self.survey.data[i]

        # plot marker
        ex = self.x[i]
        ey = self.y[i]
        ez = self.z[i]

        if t == 'QUAD' : 
            self.drawQuad(c,s,ex,ey,ez)
        elif t == 'SBEN' : 
            self.drawBend(c,s,ex,ey,ez) 
        elif t == 'RBEN' : 
            self.drawBend(c,s,ex,ey,ez)
        elif t == 'MONI' : 
            self.drawMoni(c,s,ex,ey,ez)            
        elif t == 'MARK' : 
            self.drawMark(c,s,ex,ey,ez)
        else :
            print 'pymad8.Visualisation.TwoDim> Type not known'
        # pl.plot([z[i]],[x[i]],"+")

        # Annotate element

    def drawQuad(self,c,s,x,y,z) : 
        if self.debug : 
            print 'Visualisation.TwoDim.drawQuad>'
            print '>',c
            print '>',s
        if self.fancy : 
            # get data
            ql = c[m8.Output.Common.keys['quad']['l']]
            qk = c[m8.Output.Common.keys['quad']['k1']]
            qt = s[m8.Output.Survey.keys['theta']]

            # make patch
            qr = transformedRect([z,x],ql,self.quadWidth,qt)
            qr.set_color('r')
            qr.set_alpha(0.6)

            # add patch
            ax = plt.gca()
            ax.add_patch(qr)
        else : 
            pl.plot([z],[x],'r+')

    def drawBend(self,c,s,x,y,z) : 
        if self.debug : 
            print 'Visualisation.TwoDim.drawDipole>'

        if self.fancy : 
            # get data
            bl = c[m8.Output.Common.keys['sben']['l']]
            bt = s[m8.Output.Survey.keys['theta']]
            
        else : 
            pl.plot([z],[x],'b+')        

    def drawMoni(self,c,s,x,y,z) :
        if self.debug : 
            print 'Visualisation.TwoDim.drawMoni>'

        pl.plot([z],[x],'g+')                

    def drawMark(self,c,s,x,y,z) : 
        if self.debug : 
            print 'Visualisation.TwoDim.drawMark>'

        pl.plot([z],[x],'b+')        
    
