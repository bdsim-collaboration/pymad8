import xml.dom.minidom

class Mad8 :
    def __init__(self, fileName) :
        # Keys to search for
        self.quadrupole_keys = ['L','K1','TILT','APERTURE']
        self.sbend_keys      = ['L','ANGLE','K1','E1','E2','TILT','HGAP','FINT','FINTX']
        self.sextupole_keys  = ['L','K2','TILT']
        self.kicker_keys     = ['L']
        self.drift_keys      = ['L']
        
        # list of types found
        self.elemTypes = []

        self.fileName = fileName
        self.readFile(fileName)
        x = self.parseFile()

        
        f = open(fileName[0:fileName.rfind('.')]+'.aml','w')
        f.write(x)
        f.close()


    def readFile(self,fileName) :
        '''Readfile into long uncontinued lines'''
        
        f = open(fileName,'r')

        self.fl = [] # file long lines
        
        lls = [] # stored long line
        for l in f :
            ssl = l.strip()
            if l[0] == '!' :
                pass
            elif len(ssl) == 0 :
                pass
            elif ssl[-1] == '&' :
                ll = ssl.rstrip('&')
                lls.append(ll)
            else :
                lls.append(ssl)                
                self.fl.append(''.join(lls))
                lls = []

    def parseFile(self, mom = 1.3e9, line = 'ATF2') :
        # create docuement
        self.doc = xml.dom.minidom.Document()        
        self.le= self.doc.createElement('laboratory')
        self.doc.appendChild(self.le)

        e    = self.doc.createElement('lattice')
        epcd = self.doc.createElement('pc')
        epcd.setAttribute('design',str(mom))
        e.appendChild(epcd)
        self.le.appendChild(e)
        
        for l in self.fl :
            self.parseLine(l)
        
        return self.doc.toprettyxml()
        
    def parseLine(self,l) :
        t0 = l.split(' ')
        name = t0[0].rstrip(':')
        type = []

        if name != 'RETURN' :
            type = t0[1].rstrip(',')    

            # store types
            try :
                ind = self.elemTypes.index(type) 
            except ValueError :
                self.elemTypes.append(type)
        #################################################################        
        # LINE
        #################################################################        
        if type == 'LINE' :
            # decode line
            elist = l[l.rfind('(')+1:l.rfind(')')].split(',')

            # build xml
            e     = self.doc.createElement('sector')
            e.setAttribute('name',name)

            for elem in elist :
                el= self.doc.createElement('element')
                el.setAttribute('ref',elem.lstrip().rstrip())
                e.appendChild(el)

            self.le.appendChild(e)            
        #################################################################        
        # SBEND
        #################################################################        
        elif type == 'SBEND' :
            # decode line
            kvals = self.getKeys(self.sbend_keys,l)
            
            # build xml
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            eb    = self.doc.createElement('bend')
            e.appendChild(eb)
            eg    = self.doc.createElement('g')
            if kvals['ANGLE'] == 'NONE' :
                eg.setAttribute('design','0')
            else :
                g = float(kvals['ANGLE'])/float(kvals['L'])
                eg.setAttribute('design',str(g))
            eb.appendChild(eg)            

            if kvals['K1'] != 'NONE' :
                esm   = self.doc.createElement('scaled_multipole')
                esmk  = self.doc.createElement('k_coef')
                esmk.setAttribute('n','2')
                esmk.setAttribute('design',str(float(kvals['K1'])/float(eg.getAttribute('design'))))
                eb.appendChild(esm)
                esm.appendChild(esmk)                
            ee1   = self.doc.createElement('e1')
            if kvals['E1'] == 'NONE' :
                ee1.setAttribute('design','0')
            else :
                ee1.setAttribute('design',kvals['E1'])
            eb.appendChild(ee1)
            ee2   = self.doc.createElement('e2')
            if kvals['E2'] == 'NONE' :
                ee2.setAttribute('design','0') 
            else :
                ee2.setAttribute('design',kvals['E2'])            
            eb.appendChild(ee2)            
            ehg1  = self.doc.createElement('h_gap1')
            if kvals['HGAP'] != 'NONE' :
                ehg1.setAttribute('design',kvals['HGAP'])
            else :
                ehg1.setAttribute('design','0')
            eb.appendChild(ehg1)
            ehg2  = self.doc.createElement('h_gap2')
            if kvals['HGAP'] != 'NONE' :
                ehg2.setAttribute('design',kvals['HGAP'])
            else :
                ehg2.setAttribute('design','0')                
            eb.appendChild(ehg2)
            efi1  = self.doc.createElement('f_int1')
            if kvals['FINT'] != 'NONE' :
                efi1.setAttribute('design',kvals['FINT'])
            else :
                efi1.setAttribute('design',0)                
            eb.appendChild(efi1)
            efi2  = self.doc.createElement('f_int2')
            if kvals['FINTX'] != 'NONE' :
                efi2.setAttribute('design',kvals['FINTX'])                
            else :
                efi2.setAttribute('design','0')
            eb.appendChild(efi2)            
            eh1  = self.doc.createElement('h1')
            eh1.setAttribute('design','0')
            eb.appendChild(eh1)
            eh2  = self.doc.createElement('h2')
            eh2.setAttribute('design','0')
            eb.appendChild(eh2)                        
            el   = self.doc.createElement('length')            
            el.setAttribute('design',kvals['L'])
            e.appendChild(el)
            self.le.appendChild(e)
        #################################################################        
        # QUADRUPOLE
        #################################################################        
        elif type == 'QUADRUPOLE' :
            # decode line
            kvals = self.getKeys(self.quadrupole_keys,l)

            # build xml
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            eq    = self.doc.createElement('quadrupole')
            ek    = self.doc.createElement('k')
            ek.setAttribute('design',kvals['K1'])
            ek.setAttribute('err','0')
            el    = self.doc.createElement('length')
            el.setAttribute('design',kvals['L'])
            e.appendChild(eq)            
            e.appendChild(el)
            if kvals['TILT'] != 'NONE' :
                et    = self.doc.createElement('tilt')
                et.setAttribute('design',kvals['TILT'])
                e.appendChild(et)
            eq.appendChild(ek)
            self.le.appendChild(e)            
        #################################################################        
        # SEXTUPOLE
        #################################################################        
        elif type == 'SEXTUPOLE' :
            # decode line
            kvals = self.getKeys(self.sextupole_keys,l)

            # build xml
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            es    = self.doc.createElement('sextupole')
            e.appendChild(es)
            ek    = self.doc.createElement('k')
            ek.setAttribute('design',kvals['K2'])
            ek.setAttribute('err','0')
            es.appendChild(ek)            
            el    = self.doc.createElement('length')
            el.setAttribute('design',kvals['L'])
            e.appendChild(el)
            if kvals['TILT'] != 'NONE' :
                et    = self.doc.createElement('tilt')
                et.setAttribute('design',kvals['TILT'])
                e.appendChild(et)                
            self.le.appendChild(e)
        #################################################################        
        # MULTIPOLE
        #################################################################        
        elif type == 'MULTIPOLE' :
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            el    = self.doc.createElement('length')
            el.setAttribute('design','0')
            e.appendChild(el)
            self.le.appendChild(e)
        #################################################################        
        # RFCAVITY
        #################################################################        
        elif type == 'RFCAVITY' :
            pass
        #################################################################        
        # DRIFT
        #################################################################        
        elif type == 'DRIFT' :
            # decode line
            kvals = self.getKeys(self.drift_keys,l)            

            # build xml
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            el    = self.doc.createElement('length')
            el.setAttribute('design',kvals['L'])
            e.appendChild(el)
            self.le.appendChild(e)
        #################################################################        
        # HKICKER
        #################################################################        
        elif type == 'HKICKER' :
            # decode line
            kvals = self.getKeys(self.kicker_keys,l)            
            
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            ek    = self.doc.createElement('kicker')
            e.appendChild(ek)
            ekk = self.doc.createElement('x_kick')
            ekk.setAttribute('design','0.0')
            ek.appendChild(ekk)
            el    = self.doc.createElement('length')
            if kvals['L'] != 'NONE' :
                el.setAttribute('design',kvals['L'])
            else :
                el.setAttribute('design','0')            
            e.appendChild(el)
            self.le.appendChild(e)
        #################################################################        
        # VKICKER
        #################################################################        
        elif type == 'VKICKER' :
            # decode line
            kvals = self.getKeys(self.kicker_keys,l)            

            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            ek    = self.doc.createElement('kicker')
            e.appendChild(ek)
            ekk = self.doc.createElement('y_kick')
            ekk.setAttribute('design','0.0')
            ek.appendChild(ekk)
            el    = self.doc.createElement('length')
            if kvals['L'] != 'NONE' :
                el.setAttribute('design',kvals['L'])
            else :
                el.setAttribute('design','0')            
            e.appendChild(el)
            self.le.appendChild(e)
        #################################################################        
        # MONITOR
        #################################################################        
        elif type == 'MONITOR' :
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            eit   = self.doc.createElement('instrument')
            eit.setAttribute('type','MONI')
            e.appendChild(eit)
            self.le.appendChild(e)                        
        #################################################################        
        # INSTRUMENT
        #################################################################        
        elif type == 'INSTRUMENT' :
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            eit   = self.doc.createElement('instrument')
            eit.setAttribute('type','INST')
            e.appendChild(eit)
            self.le.appendChild(e)                        
        #################################################################        
        # MARKER
        #################################################################        
        elif type == 'MARKER' :
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            if 'REF' in name:
                etype = self.doc.createElement('instrument')
                etype.setAttribute('type','IMON')
            else:
                etype = self.doc.createElement('marker')
            e.appendChild(etype)
            self.le.appendChild(e)
        #################################################################        
        # PROFILE
        #################################################################        
        elif type == 'PROFILE' :
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            eit   = self.doc.createElement('instrument')
            eit.setAttribute('type','PROF')
            e.appendChild(eit)
            self.le.appendChild(e)            
        #################################################################        
        # WIRE
        #################################################################        
        elif type == 'WIRE' :
            e     = self.doc.createElement('element')
            e.setAttribute('name',name)
            eit   = self.doc.createElement('instrument')
            eit.setAttribute('type','WIRE')
            e.appendChild(eit)
            self.le.appendChild(e)                        
        #################################################################        
        # UNKNOWN
        #################################################################        
        else :
            print('Converter element not processed',type)
            
    def getKeys(self, keylist, l) :
#        print l

        # list of key values
        kvallist = dict()

        # look over keys
        for k in keylist :
            # find value region of text 
            start = l.rfind(k+'=')+len(k)+1
            end   = l.find(',',start)

            # check that key is found
            if l.rfind(k+'=') == -1 :
                kval = 'NONE'
            # found so get the value
            else :
                if end == -1 :
                    end = len(l)                
                kval = l[start:end]

            # append to list of values
            kvallist[k] = kval
            
        return kvallist
