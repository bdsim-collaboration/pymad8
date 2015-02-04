import re as _re
import numpy as _np

class Loader :
    def __init__(self, fileName) :
        self.mad8Types = ['LINE',
                          'DRIFT',
                          'VKICKER',
                          'HKICKER',
                          'MARKER',
                          'SBEND',
                          'RBEND',
                          'QUADRUPOLE',
                          'SEXTUPOLE',
                          'OCTUPOLE',
                          'MULTIPOLE',
                          'ECOLLIMATOR',
                          'RCOLLIMATOR',
                          'WIRE',
                          'INSTRUMENT',
                          'MONITOR',
                          'LCAVITY']
        self.elementDict = {}
        self.mad8ElementClasses = []
        self.elementList = []
        # New types.  Think more on type is dict more useful
        self.components = []
        self.sequences = []
        self.samplers = []

        self.loadFile(fileName) # Does some basic file formatting
        self.fileAnalysis()
        self.expandFile()
        # exists in a converter?
        #self.saveFile('ILC.saveline') # Saves file doesn't update element definitions

    def loadFile(self, fileName) :
        f = open(fileName)

        # load entire file and remove continued lines
        checkNext = False
        # new list of complete lines
        self.ff = []
        # line to add
        lta = ''
        # loop over lines in original file
        for l in f :
            sl = l.strip('\n ')   # remove trailing characters
            t  = sl.split()       # split
            if len(sl) < 1 :      # remove empty lines
                self.ff.append(sl)
                continue
            if sl[-1] == '&' :    # check for continuations
                lta = lta+sl[0:-1]# append line
            else :
                if len(lta) == 0 :# no continuation
                    lta = l       # line is complete
                    self.ff.append(lta)
                    lta = ''
                else :
                    lta = lta+sl  # there was some line to add to
                    self.ff.append(lta+'\n')# append
                    lta = ''      # clear line to add

    def fileAnalysis(self) :
        # loop over lines
        for l in self.ff :
            l = l.replace(' ','')
            sl = _re.split('[:,\n]', l)
            sl = [s.strip() for s in sl]

            # skip empty lines
            if len(sl) <= 1  :
                continue
            # skip comment lines
            if not sl[0] == '' :
                if sl[0][0] == '!' :
                    continue

            name = sl[0]
            temp = sl[1] # Can you use type (reserved keyword)? Changed just to be sure, better name!

            # store all elements in dictionary
            if name == 'RETURN' :
                continue
            elif temp.find('LINE') != -1 : # find() returns -1 if it doesn't find anything
                # do lines a little differently find ( and )
                self.elementDict[name] = ['LINE', l[l.find('(') + 1 : l.find(')')].split(',')]
            else :
                self.elementDict[name] = [temp, sl[2:-1]]

            # store element name to recreate file
            self.elementList.append(name)

            # find element classes
            try :
                self.mad8Types.index(temp)
#                print name, type
            except ValueError :
#                print name, type, " <<<< element class"
                self.mad8ElementClasses.append(temp)

    def flattenElements(self, elementName) :
        if not elementName in self.mad8Types :

            for i in self.elementList :

                if i == self.elementDict[elementName][0] and not i == '' : # How to handle if they're empty?
                    self.elementDict[elementName][0] = self.elementDict[i][0]

                    if self.elementDict[i][1] == [] :
                        continue # No properties to update

                    if self.elementDict[i][1] == self.elementDict[elementName][1] : # properties are identicle so skip.
                        continue

                    for prop in self.elementDict[i][1] :

                        if prop in self.elementDict[elementName][1] :
                            continue # Already in the list

                        self.elementDict[elementName][1].append(prop) #property isn't empty, the same, or diff val, so add it.
                        self.flattenElements(self.elementDict[elementName][0])


    def expandFile(self) :
        # loop through all elements

        for e in self.elementList :
            self.flattenElements(e)

def SavelineTest():
    loader = Loader('ebds.saveline')
