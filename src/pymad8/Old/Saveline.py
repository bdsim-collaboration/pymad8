# maybe this should be in Converter.py

import re as _re
import numpy as _np


class Loader:
    def __init__(self, fileName):
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
        self.components = []
        self.sequences = []
        self.samplers = []

        self.loadFile(fileName)  # Does some basic file formatting
        self.fileAnalysis()
        self.expand_file()


    def loadFile(self, fileName):
        f = open(fileName)

        # load entire file and remove continued lines
        # new list of complete lines
        self.ff = []
        # line to add
        lta = ''
        # loop over lines in original file
        for l in f:
            sl = l.strip('\n ')  # remove trailing characters
            t = sl.split()  # split
            if len(sl) < 1:  # remove empty lines
                self.ff.append(sl)
                continue
            if sl[-1] == '&':  # check for continuations
                lta = lta + sl[0:-1]  # append line
            else:
                if len(lta) == 0:  # no continuation
                    lta = l  # line is complete
                    self.ff.append(lta)
                    lta = ''
                else:
                    lta = lta + sl  #  there was some line to add to
                    self.ff.append(lta + '\n')  # append
                    lta = ''  # clear line to add

    def fileAnalysis(self):
        # loop over lines
        for l in self.ff:
            l = l.replace(' ', '')
            sl = _re.split('[:,\n]', l)
            sl = [s.strip() for s in sl]
            # skip empty lines
            if len(sl) <= 1:
                continue
            # skip comment lines
            if not sl[0] == '':
                if sl[0][0] == '!':
                    continue

            element_name = sl[0]
            element_type = sl[1]

            # store all elements in dictionary
            if element_name == 'RETURN':
                continue
            elif element_type.find('LINE') != -1:  # find() returns -1 if it doesn't find anything
                # do lines a little differently find ( and )
                self.elementDict[element_name] = {'LINE': l[l.find('(') + 1: l.find(')')].split(',')}
            else:
                element_properties = {}
                for element_property in sl[2:-1]:
                    properties = element_property.split('=')
                    key = properties[0]
                    value = properties[1]
                    element_properties[key] = value
                self.elementDict[element_name] = {element_type: element_properties}
            # store element name to recreate file
            self.elementList.append(element_name)

            # find element classes
            try:
                self.mad8Types.index(element_type)
            # print name, type
            except ValueError:
                # print name, type, " <<<< element class"
                self.mad8ElementClasses.append(element_type)

    def flatten_elements(self, element_name):
        if element_name not in self.mad8Types:
            for i in self.elementList:
                if i == list(self.elementDict[element_name].keys())[0] and not i == '':
                    type_key = list(self.elementDict[i].keys())[0]
                    element_type_key = list(self.elementDict[element_name].keys())[0]

                    self.elementDict[element_name][type_key] = self.elementDict[element_name][element_type_key]
                    del self.elementDict[element_name][element_type_key]

                    if self.elementDict[element_name][type_key] == {}:
                        continue  # No properties to update

                    if list(self.elementDict[i].values())[0] == list(self.elementDict[element_name].values())[0]:  #  properties are identical so skip.
                        continue

                    for properties in self.elementDict[i].values():
                        for property_name, property_value in properties.items():
                            if property_name in list(self.elementDict[element_name].keys()) and property_value in list(self.elementDict[element_name].values()):
                                continue  # Already in the list

                        list(self.elementDict[element_name].values())[0].update(properties)  #  property isn't empty, the same, or diff val, so add it.
                        self.flatten_elements(list(self.elementDict[element_name].keys())[0])

    def expand_file(self):
        # loop through all elements
        for e in self.elementList:
            self.flatten_elements(e)


def SavelineTest(file_name):
    loader = Loader(file_name)
