import pylab as _pl 
import numpy as _np
import sys as _sys
import copy as _copy
import os as _os
from   collections import defaultdict
import pymad8 as _pymad8
import fortranformat as _ff

import Input as _Input

###############################################################################################################
def getValueByName(name, key, common, table) : 
    ind1 = common.findByName(name)
    ind2 = table.keys[key]
    return table.data[ind1,ind2]
###############################################################################################################
class General : 
    '''General list of accelerator component infomation'''
    def __init__(self) : 
        '''No arguments, sets up internal variables for general tables 
        type     : type of element
        name     : name of element
        dataList : parameters associated with element'''

        self.type     = [] 
        self.name     = []
        self.dataList = [] 
        self.ind      = []

    def addElement(self,type,name,data) :
        # just in case strip the type and name of trailing spaces
        type = type.strip(' ')
        name = name.strip(' ')

        if type == '' :
            type == 'init'

        self.type.append(type) 
        self.name.append(name) 
        self.dataList.append(data)

    def makeArray(self) :
        self.data = _np.array(self.dataList)
        self.ind  = _np.arange(0,len(self.data),1)

    def getIndex(self,name) : 
        ind = self.ind[_np.array(self.name) == name]
        return ind
        
    def findByName(self,name) :
        ind = self.getIndex(name)
        if len(ind) == 1 : 
            return ind[0]
        else : 
            return ind

    def findByType(self,type) :
        return self.ind[_np.array(self.type) == type]

    def getNElements(self) : 
        return len(self.name)

    def getNames(self, ind) : 
        return _np.array(self.name)[ind]

    def getRowByName(self, name) : 
        return self.getRowByIndex(self.findByName(name))

    def getRowByIndex(self, index) : # was getData
        d = {}
        d['name'] = self.name[index].strip().lower()
        d['type'] = self.type[index].strip().lower()
        dKeys = self.keys
        
        for k in dKeys :             
            d[k] = self.data[index,dKeys[k]]

        return d

    def getColumn(self, key) : 
        return self.data[:,self.keys[key]]

    def subline(self, start, end) : 
        if type(start) == str : 
            start = self.findByName(start) 
        if type(end) == str : 
            end = self.findByName(end)

        self.type = self.type[start:end]
        self.name = self.name[start:end]
        self.data = self.data[start:end]
        self.ind  = _pl.arange(0,len(self.data),1)
#        self.ind  = self.ind[start:end]


    def plotXY(self, xkey, ykey) :        
        _pl.plot(self.data[:,self.keys[xkey]],
                self.data[:,self.keys[ykey]])

###############################################################################################################
class Common(General) : 

    keys = {
        'drif'       :{'l':0                                                                              ,'aper':9,'note':10,'E':11},
        'rben'       :{'l':0, 'angle':1, 'k1':2 , 'k2':3, 'tilt':4 ,'e1':5    , 'e2':6   , 'h1':7 , 'h2':8,'aper':9,'note':10,'E':11},
        'sben'       :{'l':0, 'angle':1, 'k1':2 , 'k2':3, 'tilt':4 ,'e1':5    , 'e2':6   , 'h1':7 , 'h2':8,'aper':9,'note':10,'E':11},        
        'quad'       :{'l':0,            'k1':2 ,         'tilt':4                                        ,'aper':9,'note':10,'E':11},
        'sext'       :{'l':0,                    'k2':3 , 'tilt':4                                        ,'aper':9,'note':10,'E':11},
        'octu'       :{'l':0,                             'tilt':4 ,'k3':5                                ,'aper':9,'note':10,'E':11},     
        'mult'       :{       'k0l':1  , 'k1l':2,'k2l':3, 't0':4   ,'k3l':5   , 't1':6   , 't2':7 , 't3':8,'aper':9,'note':10,'E':11},
        'sole'       :{'l':0,                                       'ks':5                                ,'aper':9,'note':10,'E':11},
        'rfcavity'   :{'l':0,                                       'freq':5  , 'volt':6 , 'lag':7        ,'aper':9,'note':10,'E':11},
        'elseparator':{'l':0,                             'tilt':4 ,'efield':5                            ,'aper':9,'note':10,'E':11},
        'kick'       :{'l':0,                             'hkick':4,'vkick':5                             ,'aper':9,'note':10,'E':11},    
        'hkic'       :{'l':0,                             'hkick':4                                       ,'aper':9,'note':10,'E':11},    
        'vkic'       :{'l':0,                                       'vkick':5                             ,'aper':9,'note':10,'E':11}, 
        'srot'       :{'l':0,                                       'angle':5                             ,'aper':9,'note':10,'E':11}, 
        'yrot'       :{'l':0,                                       'angle':5                             ,'aper':9,'note':10,'E':11}, 
        'moni'       :{'l':0                                                                              ,'aper':9,'note':10,'E':11},
        'hmonitor'   :{'l':0                                                                              ,'aper':9,'note':10,'E':11},
        'vmonitor'   :{'l':0                                                                              ,'aper':9,'note':10,'E':11},
        'mark'       :{'l':0                                                                              ,'aper':9,'note':10,'E':11},
        'ecol'       :{'l':0,                             'xsize':4,'ysize':5                             ,'aper':9,'note':10,'E':11},   
        'rcol'       :{'l':0,                             'xsize':4,'ysize':5                             ,'aper':9,'note':10,'E':11},
        'mark'       :{'l':0,                                                                                       'note':10,'E':11},
        'inst'       :{'l':0,                                                                                       'note':10,'E':11},
        'wire'       :{'l':0,                                                                                       'note':10,'E':11},
        'imon'       :{'l':0,                                                                                       'note':10,'E':11},
        'prof'       :{'l':0,                                                                                       'note':10,'E':11},
        'blmo'       :{'l':0,                                                                                       'note':10,'E':11},
        'lcav'       :{'l':0,                                       'freq':5  , 'volt':6 , 'lag':7        ,'aper':9,'note':10,'E':11},
        'matr'       :{'l':0,                                                                              'aper':9,          'E':11}
    }

    def __init__(self) :
        General.__init__(self)

    def getData(self, index) :
        d = {}
        d['name'] = self.name[index].strip().lower()
        d['type'] = self.type[index].strip().lower()
        dKeys = self.keys[self.type[index].lower()]
        
        for k in dKeys : 
            d[k] = self.data[index,dKeys[k]]

        return d

    def getRowByName(self, name) : 
        return self.getRowByIndex(self.findByName(name))

    def getRowByIndex(self, index) : # was getData
        d = {}
        d['name'] = self.name[index].strip().lower()
        d['type'] = self.type[index].strip().lower()
        dKeys = self.keys

        if d['type'] == '' : 
            return d

        for k in dKeys[d['type']] :
            d[k] = self.data[index,dKeys[d['type']][k]]
        
        try : 
            self.keys.keys().index(d['type'])
            for k in dKeys[d['type']] : 
                if k != 'note' : 
                    d[k] = float(d[k])
            return d
        except ValueError : 
            return d


    def getColumn(self,colName) : 
        d  = [] 
        if colName == "E" : 
            for i in range(0,len(self.data),1) : 
                r = self.getRowByIndex(i)
                if r['name'] == 'initial' :
                    d.append(0)
                if r['name'] != 'initial' : 
                    d.append(self.getRowByIndex(i)['E'])                 
            d[0] = d[1]
        else : 
            print "Common.getColumn does not exist for ", colName
            return _np.array([])
        return _np.array(d)

    def containsEnergyVariation(self) : 
        '''Method to determine if the energy is constant in the lattice
        Required if there is 
        1) RfCavities
        ''' 
        for i in range(0,len(self.data),1) : 
            if self.type[i] == 'LCAV' : 
                return True
        return False 

    def getApertures(self, raw = True) : 
        aper = [] 

        # see if there is any aperture information 
        for i in range(0,len(self.data),1) :             
            aper.append(float(self.data[i][self.keys['drif']['aper']]))
        if sum(aper)== 0 :
            aper = []
            for i in range(0,len(self.data),1) :             
                aper.append(0.1)            
            return aper

        if raw : 
            for i in range(0,len(self.data),1) :             
                aper.append(self.data[i][self.keys['drif']['aper']])
            return aper
        else : 
            # find first non-zero aperture 
            for i in range(0,len(self.data),1) : 
                a = float(self.data[i][self.keys['drif']['aper']])
                if a != 0 : 
                    firstA = a
                    break 

            lastA = firstA 
            for i in range(0,len(self.data),1) : 
                a = float(self.data[i][self.keys['drif']['aper']])
                if a == 0 : 
                    a = lastA
                else : 
                    lastA = a 
                aper.append(a)

            return aper
                    
    def makeLocationList(self, elementNames = []) : 
        pass

###############################################################################################################
class Twiss(General) : 
    '''Twiss data structure
    data : numpy array of data 
    keys : key to data''' 

    keys = {'alfx':0, 'betx':1, 'mux':2, 'dx':3, 'dpx':4,
            'alfy':5, 'bety':6, 'muy':7, 'dy':8, 'dpy':9,
            'x':10,'px':11,'y':12,'py':13,'suml':14}

    def __init__(self) : 
        General.__init__(self)

    def plotBeta(self) : 
        _pl.figure(1)
        self.plotXY('suml','betx')
        self.plotXY('suml','bety')

    def plotEta(self) :
        _pl.figure(2)
        self.plotXY('suml','dx');
        self.plotXY('suml','dy');
        
    def plotEtaPrime(self) : 
        _pl.figure(3)
        self.plotXY('suml','dpx');
        self.plotXY('suml','dpy');        

    def plotAlf(self) : 
        _pl.figure(3)
        self.plotXY('suml','alfx');
        self.plotXY('suml','alfy');                

    def plotMu(self) : 
        _pl.figure(3)
        self.plotXY('suml','mux');
        self.plotXY('suml','muy');                

    def nameFromNearestS(self,s) :
        suml = self.getColumn('suml')
        for i in range(0,len(suml)-1) :
            if s > suml[i-1] and s < suml[i] :
                return self.name[i]
        return "Not found"

###############################################################################################################
class Survey(General) :
    '''Survey data structure
    data : numpy array of data 
    keys : key to data''' 
    keys = {'x':0,'y':1,'z':2,'suml':3,
            'theta':4,'phi':5,'psi':6}    

    def __init__(self) : 
        General.__init__(self)

###############################################################################################################
class Rmat(General) : 
    '''Rmatrix data structure
    data : numpy array of data 
    keys : key to data''' 

    keys = {'r11':0 ,'r12':1 ,'r13':2 ,'r14':3 ,'r15':4 ,'r16':5 ,
            'r21':6 ,'r22':7 ,'r23':8 ,'r24':9 ,'r25':10,'r26':11,
            'r31':12,'r32':13,'r33':14,'r34':15,'r35':16,'r36':17,
            'r41':18,'r42':19,'r43':20,'r44':21,'r45':22,'r46':23,
            'r51':24,'r52':25,'r53':26,'r54':27,'r55':28,'r56':29,
            'r61':30,'r62':31,'r63':32,'r64':33,'r65':34,'r66':35,
            'suml':36}    

    def __init__(self) : 
        General.__init__(self)

    def getData(self, index) :
        return self.data[index]

###############################################################################################################
class Chrom(General) : 
    '''Chromaticity data structure
    data : numpy array of data 
    keys : key to data''' 

    def __init__(self) :
        General.__init__(self)

    def getData(self, index) :
        return self.data[index]
     
###############################################################################################################           
class Envelope(General) : 
    '''Beam envelope data structure
    data : numpy array of data 
    keys : key to data'''
 
    keys = {'s11':0 ,'s12':1 ,'s13':2 ,'s14':3 ,'s15':4 ,'s16':5 ,
            's21':6 ,'s22':7 ,'s23':8 ,'s24':9 ,'s25':10,'s26':11,
            's31':12,'s32':13,'s33':14,'s34':15,'s35':16,'s36':17,
            's41':18,'s42':19,'s43':20,'s44':21,'s45':22,'s46':23,
            's51':24,'s52':25,'s53':26,'s54':27,'s55':28,'s56':29,
            's61':30,'s62':31,'s63':32,'s64':33,'s65':34,'s66':35,
            'suml':36}    

    def __init__(self) :
        General.__init__(self) 

    def getData(self,index) :
        return self.data[index]

###############################################################################################################
class OutputReader :

    '''Class to load different Mad8 output files
    Usage : 
    o = Mad8.OutputReader()
    [c, s] = o.readFile('./survey.tape','survey')    
    [c, r] = o.readFile('./rmat.tape','rmat')
    [c, t] = o.readFile('./twiss.tape','twiss')
    [c, c] = o.readFile('./chrom.tape','chrom')
    [c, e] = o.readFile('./envelope.tape','envel')

    c : Common data
    r : Rmat object 
    t : Twiss object 
    c : Chrom object 
    e : Envelope object
    '''

    def __init__(self) : 
        pass
#        import fortranformat as ff

    def readFile(self, fileName = '', type = 'twiss') :
        '''read mad8 output file
        '''
        
        self.fileName = fileName
        self.type     = type.lower() 

        if self.type == 'twiss' :
            r = self.readTwissFile()
        elif self.type == 'rmat' :
            r = self.readRmatFile()
        elif self.type == 'survey' : 
            r = self.readSurveyFile()
        elif self.type == 'chrom' :
            r = self.readChromFile() 
        elif self.type == 'envel' : 
            r = self.readEnvelopeFile()
        else : 
            print 'Mad8.OutputReader.readFile> Unknown file'
            return None 
        return r 

    def readTwissFile(self, f= None) :
        if f == None :
            f = open(self.fileName,'r')
        
        ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
        ffhr2 = _ff.FortranRecordReader('(A80)')

#        print '"'+l+'"'                
        # read header
        h1 = ffhr1.read(f.readline())
        h2 = ffhr2.read(f.readline())
        nrec = h1[7]

        print 'Mad8.readTwissFile > nrec='+str(nrec)
        
        ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
        ffe2 = _ff.FortranRecordReader('(5E16.9)')

        common  = Common() 
        twiss   = Twiss()

        # loop over entries
        for i in range(0,nrec,1) :
            l1 = ffe1.read(f.readline())
            l2 = ffe2.read(f.readline())
            l3 = ffe2.read(f.readline())
            l4 = ffe2.read(f.readline())
            l5 = ffe2.read(f.readline())  
            common.addElement(l1[0],l1[1],l1[2:6]+l2+[l1[6],l1[7],l1[8]])          
            twiss.addElement(l1[0],l1[1].strip(),l3+l4+l5)
        common.makeArray()
        twiss.makeArray()

        # minimum and maximum s
        twiss.smin = twiss.getColumn('suml')[0]
        twiss.smax = twiss.getColumn('suml')[-1]

        # number of elements 
        common.nele = len(common.name)
        twiss.nele  = len(twiss.getColumn('suml'))

        return [common,twiss]

    def readRmatFile(self, f=None) :
        if f == None :
            f = open(self.fileName,'r')

        # Standard header definition 
        ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
        ffhr2 = _ff.FortranRecordReader('(A80)')

        # read header
        h1 = ffhr1.read(f.readline())
        h2 = ffhr2.read(f.readline())
        # number of records
        nrec = h1[7]
        print 'Mad8.readRmatFile  > nrec='+str(nrec)

        ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
        ffe2 = _ff.FortranRecordReader('(5E16.9)')
        ffe3 = _ff.FortranRecordReader('(6E16.9)')
        ffe4 = _ff.FortranRecordReader('(7E16.9)')

        common  = Common() 
        rmat    = Rmat() 

        for i in range(0,nrec,1) :
            l1 = ffe1.read(f.readline())
            l2 = ffe2.read(f.readline())
            l3 = ffe3.read(f.readline())
            l4 = ffe3.read(f.readline())
            l5 = ffe3.read(f.readline())
            l6 = ffe3.read(f.readline())
            l7 = ffe3.read(f.readline())
            l8 = ffe4.read(f.readline())
            common.addElement(l1[0],l1[1],l1[2:6]+l2+[l1[6],l1[7],l1[8]])          
            rmat.addElement(l1[0],l1[1].strip(),l3+l4+l5+l6+l7+l8)
        f.close()    

        common.makeArray()
        rmat.makeArray()

        # minimum and maximum s
        rmat.smin = rmat.getColumn('suml')[0]
        rmat.smax = rmat.getColumn('suml')[-1]

        # number of elements 
        common.nele = len(common.name)
        rmat.nele  = len(rmat.getColumn('suml'))

        return [common,rmat]

    def readChromFile(self, f=None ) :
        if f==None : 
            f = open(self.fileName,'r') 

        # First read twiss 
        [c,twiss] = self.readTwissFile(f)
        
        # 3 random lines at end of chrom tape
        f.readline()
        f.readline()
        f.readline()

        # read header
        ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
        ffhr2 = _ff.FortranRecordReader('(A80)')
        h1 = ffhr1.read(f.readline())
        h2 = ffhr2.read(f.readline())
        nrec = h1[7]

        print 'Mad8.readChromFile > nrec='+str(nrec)

        ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
        ffe2 = _ff.FortranRecordReader('(5E16.9)')
        ffe3 = _ff.FortranRecordReader('(6E16.9)')
        
        common = Common() 
        chrom  = Chrom() 

        for i in range(0,nrec,1) : 
            l1 = ffe1.read(f.readline())
            l2 = ffe2.read(f.readline())
            l3 = ffe3.read(f.readline())
            l4 = ffe3.read(f.readline())
            l5 = ffe3.read(f.readline())

            common.addElement(l1[0],l1[1],l1[2:6]+l2+[l1[6],l1[7],l1[8]])           
            chrom.addElement(l1[0],l1[1].strip(),l3+l4+l5)
            
        common.makeArray()
        chrom.makeArray()

        # minimum and maximum s
        chrom.smin = chrom.getColumn('suml')[0]
        chrom.smax = chrom.getColumn('suml')[-1]

        # number of elements 
        common.nele = len(common.name)
        chrom.nele  = len(chrom.getColumn('suml'))

        return [common,twiss,chrom]
    
    def readEnvelopeFile(self, f=None) : 
        if f==None : 
            f = open (self.fileName,'r') 

        ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
        ffhr2 = _ff.FortranRecordReader('(A80)')

        # read header
        h1 = ffhr1.read(f.readline())
        h2 = ffhr2.read(f.readline())
        nrec = h1[7]

        print 'Mad8.readEnvelopeFile > nrec='+str(nrec)

        ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
        ffe2 = _ff.FortranRecordReader('(5E16.9)')
        ffe3 = _ff.FortranRecordReader('(6E16.9)')
        ffe4 = _ff.FortranRecordReader('(7E16.9)')
        
        common = Common()
        envel  = Envelope() 

        for i in range(0,nrec,1) : 
            l1 = ffe1.read(f.readline())
            l2 = ffe2.read(f.readline())
            l3 = ffe3.read(f.readline())
            l4 = ffe3.read(f.readline())
            l5 = ffe3.read(f.readline())
            l6 = ffe3.read(f.readline())
            l7 = ffe3.read(f.readline())
            l8 = ffe4.read(f.readline())

            common.addElement(l1[0],l1[1],l1[2:6]+l2+[l1[6],l1[7],l1[8]])          
            envel.addElement(l1[0],l1[1].strip(),l3+l4+l5+l6+l7+l8)
        f.close()

        common.makeArray()
        envel.makeArray()

        # minimum and maximum s
        envel.smin = envel.getColumn('suml')[0]
        envel.smax = envel.getColumn('suml')[-1]

        # number of elements 
        common.nele = len(common.name)
        envel.nele  = len(envel.getColumn('suml'))

        return [common,envel] 

    def readSurveyFile(self) :
        f = open(self.fileName,'r') 
        
        # Standard header definition 
        ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
        ffhr2 = _ff.FortranRecordReader('(A80)')

        # read header
        h1 = ffhr1.read(f.readline())
        h2 = ffhr2.read(f.readline())
        # number of records
        nrec = h1[7]

        print 'Mad8.readSurveyFile> nrec='+str(nrec)
        ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
        ffe2 = _ff.FortranRecordReader('(5E16.9)')
        ffe3 = _ff.FortranRecordReader('(4E16.9)')
        ffe4 = _ff.FortranRecordReader('(3E16.9)')
        
        common  = Common() 
        survey  = Survey() 
                
        for i in range(0,nrec,1) :
            l1 = ffe1.read(f.readline())
            l2 = ffe2.read(f.readline())
            l3 = ffe3.read(f.readline())
            l4 = ffe4.read(f.readline())
            common.addElement(l1[0],l1[1],l1[2:6]+l2+[l1[6],l1[7],l1[8]])          
            survey.addElement(l1[0],l1[1].strip(),l3+l4)            

        f.close()
        
        common.makeArray()
        survey.makeArray()

        return [common,survey]
    
###############################################################################################################
class Mad8 :
    def __init__(self, filename) : 
        self.particle = ''
        self.subroutines = []
        self.readFile(filename) 

    def readFile(self,filename) :
        f = open(filename)
        self.file = []

        # load file
        for l in f : 
            ls = l.strip()
            if len(ls) > 0 :
                self.file.append(l.strip())
        f.close()

        # determine particle
        for l in self.file : 
            if l.find("PARTICLE") != -1:
                if l.find("ELECTRON") != -1 : 
                    self.particle = "ELECTRON"
                elif l.find("POSITRON") != -1 :
                    self.particle = "POSITRON"

        # determine subroutines
        for l in self.file :
            if l.find("SUBROUTINE") != -1: 
                sl = l.split()
                if len(sl) > 1 :                     
                    self.subroutines.append(sl[0])

###############################################################################################################
class EchoValue :
    def __init__(self,echoFileName) : 
        self.echoFileName = echoFileName
        self.valueDict = {}
        
    def loadValues(self) : 
        f = open(self.echoFileName) 

        for l in f : 
            if l.find("Value") != -1 :
                sl = l.split()
                k  = sl[3].strip('"')
                v  = float(sl[5])
                self.valueDict[k] = v

###############################################################################################################
class Saveline :
    def __init__(self, fileName, lineName = 'EBDS') : 
        self.fileName = fileName
        self.lineName = lineName

        self.dictFile            = []
        self.expandedLine        = []
        self.dictFileRenamed     = []
        self.expandedLineRenamed = []
        self.readFile(self.fileName)
        self.parseFile()
        
    def readFile(self,fileName) : 

        f = open(fileName) 
        self.file = [] 
        for l in f : 
            self.file.append(l)
        f.close()

        # tidy file
        self.file = _Input.tidy(self.file)

        # remove comments
        self.file = _Input.removeComments(self.file)

        # remove continuation symbols
        self.file = _Input.removeContinuationSymbols(self.file)
                
    def parseFile(self) :
        for l in self.file : 
            d = _Input.decodeFileLine(l)
            self.dictFile.append(d)
            
    def expandLine(self) : 
        line = self.findNamedDict(self.lineName)['LINE']
        self.expandedLine = []
        for l in line : 
            self.expandedLine = self.expandedLine + self.findNamedDict(l)['LINE']

    def findNamedIndex(self,name) :
        for i in range(0,len(self.dictFile)) :
            if self.dictFile[i]['name'] == name : 
                return i

        return -1

    def findNamedDict(self, name) : 
        idx = self.findNamedIndex(name)
        if idx != -1 : 
            return self.dictFile[idx]
        else :
            return dict()

    def findRenamedNamedIndex(self,name) :
        for i in range(0,len(self.dictFileRenamed)) :
            if self.dictFileRenamed[i]['name'] == name : 
                return i

        return -1

    def findRenamedNamedDict(self, name) : 
        idx = self.findRenamedNamedIndex(name)
        if idx != -1 : 
            return self.dictFileRenamed[idx]
        else :
            return dict()

    def removeReplacements(self) : 
        if len(self.expandedLine) == 0 :
            self.expandLine()

    def removeDuplicates(self) : 
        if len(self.expandedLine) == 0 :
            self.expandLine()

        self.nameCount = {}
    

        # find template elements (only need to write one)
        self.templates = []
        for i in range(0,len(self.dictFile)) : 
            d   = _copy.deepcopy(self.findNamedDict(self.dictFile[i]['name']))
            idx = self.findNamedIndex(d['type']) 
            if idx != -1 :                 
                try : 
                    self.templates.index(self.dictFile[idx]['name'])
                except: 
                    self.dictFileRenamed.append(self.dictFile[idx])
                    self.templates.append(self.dictFile[idx]['name'])
                        
        # loop over elements in beamline and append name 
        for i in range(0,len(self.expandedLine)) :
            oldName = self.expandedLine[i]
            d = _copy.deepcopy(self.findNamedDict(oldName))
            
            try : 
                self.nameCount[self.expandedLine[i]] = self.nameCount[self.expandedLine[i]]+1 
            except KeyError :
                self.nameCount[self.expandedLine[i]] = 0            
            
            newName = d['name']+'_'+str(self.nameCount[d['name']])
            
            d['name'] = newName
            self.dictFileRenamed.append(d)
            self.expandedLineRenamed.append(newName)
            
    def makeSubLines(self) :
        
        outputLines = []
        # split elements into 100 element chunks
        self.splitLine = [self.expandedLineRenamed[i:i+100] for i in range(0, len(self.expandedLineRenamed), 100)]
        
        # loop over split lines
        self.subLines = [] 
        for i in range(0,len(self.splitLine)) : 
            line = self.splitLine[i]
            # loop over elements in line
            lineName = 'L'+'%06i' % i
            l = lineName+': LINE = ('
            self.subLines.append(lineName)
            for le in line : 
                l = l + le
                if line.index(le) < len(line)-1 : 
                    l = l +', '
            l = l+')'
            outputLines.append(l)

        # geneate final line 
        l = self.lineName+': LINE = ('
        for sl in self.subLines : 
            l = l + sl 
            if self.subLines.index(sl) < len(self.subLines)-1 : 
                l = l +', '
            
        l = l+')'
        outputLines.append(l)

        return outputLines

    def writeRenamed(self,filename) : 
        # open file
        f = open(filename,'w')

        # write lines 
        lines = self.makeSubLines();
        for l in lines : 
            writeContinuation(f,l+'\n')

        # write components
        for i in range(0,len(self.dictFileRenamed)) :
            l = self.dictFileRenamed[i]['name']+': '+self.dictFileRenamed[i]['type']

            for k in self.dictFileRenamed[i].keys() :
                if k != 'name' and k != 'type' :
                    l = l+', '+k+'='+str(self.dictFileRenamed[i][k])
            l = l + '\n'
            writeContinuation(f,l)

        # write rturn 
        f.write('RETURN\n')

        # close file
        f.close()

def writeContinuation(f, l) : 

    iContinuation = 0

    # loop over all characters in string 
    for i in range(0,len(l)) : 
        r = _np.mod(i,60) # 60 characters before searching for a space

        if r>40 and l[i]==' ' and i > iContinuation+30: 
            iContinuation = i
            f.write('&\n')

        f.write(l[i])


class Track : 
    def __init__(self,folderpath, filemapname, twissname) :
        self.folderpath  = folderpath
        self.filemapname = filemapname
        self.twissname   = twissname
        self.trackdata = defaultdict(list)

        if not (self.folderpath.endswith("/")):
            self.folderpath = self.folderpath+"/"
        
        cwd = _os.getcwd()
        print "pymad8.Output.Track >> Initialised in directory:"
        if (self.folderpath.startswith("/")):
            print self.folderpath
        else:
            print cwd+"/"+self.folderpath

        print "pymad8.Output.Track >> Filemap file:"
        if (self.filemapname.startswith("/")):
            print self.filemapname
        else:
            print cwd+"/"+self.filemapname

        print "pymad8.Output.Track >> Twiss file:"
        if (self.twissname.startswith("/")):
            print self.twissname
        else:
            print cwd+"/"+self.twissname

              
    def readDir(self) : 
        """
        Loop over all mad8 track output files in the target directory and 
        build a dictionary of the data. File map is used to match data
        from track files to observation plane in the twiss file.
        """

        print "pymad8.Output.Track >> Loading twiss file"
        if(_os.path.isfile(self.twissname)):
            reader           = _pymad8.Output.OutputReader()
            [common , twiss] = reader.readFile(self.twissname, 'twiss')
        else:
            print "No such file:", self.twissname
            print "Terminating.."
            _sys.exit(1)
            
        print "pymad8.Output.Track >> Loading element map"
        if(_os.path.isfile(self.filemapname)):
            with open(self.filemapname, 'r') as filemap:
                fmap={line.split()[1] : int(line.split()[0]) for line in filemap.readlines()}
        else:
            print "No such file:", self.filemapname
            print "Terminating.."
            _sys.exit(1)

        print "pymad8.Output.Track >> Loading track files.."
        for fn in _os.listdir(self.folderpath):
            if _os.path.isfile(self.folderpath+fn):
                #print "Loading file ", fn
                data = _np.loadtxt(self.folderpath+fn, skiprows=51, unpack=True)
                self.trackdata[fn].append(data)

        #Get the s p_ositions from the twiss file and match them to the data using the filemap
        for name in self.trackdata.keys():
            idx  = fmap[name]
            S    = twiss.getRowByIndex(idx)["suml"]
            self.trackdata[name]=[self.trackdata[name],S]

        print "pymad8.Output.Track >> Done"
            

    def appendDir(self, folderpath):
        """
        Loop over all mad8 track output files in the target directory and 
        append the data to the existing data structure.  
        """
        if not (folderpath.endswith("/")):
            folderpath = folderpath+"/"
        
        cwd = _os.getcwd()
        print "pymad8.Output.Track >> Appending directory:"
        if (folderpath.startswith("/")):
            print self.folderpath
        else:
            print cwd+folderpath

        "pymad8.Output.Track >> Loading files...:"
        for fn in _os.listdir(folderpath):
            if _os.path.isfile(folderpath+fn):
                if(fn in self.trackdata.keys()):
                    data = _np.loadtxt(folderpath+fn, skiprows=51, unpack=True)
                    self.trackdata[fn][0][0] = _np.concatenate((self.trackdata[fn][0][0], data), axis=1)
                else:
                    print "pymad8.Output.Track >>Sampler name not found in orignal set, skip: ", fn 
