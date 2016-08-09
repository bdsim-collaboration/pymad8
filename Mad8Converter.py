import pymad8.Saveline as _Saveline

class Mad8Converter :

    def __init__(self, fileName, ngenerate = 1000) :

        #Import .saveline
        self.saveline = _Saveline.Loader(fileName)
        self.fileName = fileName
        self.ngenerate = ngenerate

        #Create a new .gmad that includes the three converted files, how to handle beam options?

    def convert (self) :
        #Split the saveline into three files
        self._saveFile()

        #Convert the files into a gmad format
        components = 'components_' + self.fileName.replace('saveline', 'gmad')
        sequences = 'sequences_' + self.fileName.replace('saveline', 'gmad')
        samplers = 'samplers_' + self.fileName.replace('saveline', 'gmad')

        self._formatMad82Gmad('components_' + self.fileName, components)
        self._formatMad82Gmad('sequences_' + self.fileName, sequences)
        self._formatMad82Gmad('samplers_' + self.fileName, samplers)

        container = file(self.fileName.replace('saveline', 'gmad'), 'w')

        #Check how options and beam should be handled?
        container.write('include ' + components + ';\n'
                        'include ' + sequences + ';\n'
                        'include options.gmad;\n'
                        'include beam.gmad;\n'
                        'include ' + samplers + ';\n'
                        'option ngenerate=' + str(self.ngenerate) + ';')

        container.close()

    def _saveFile(self) :

        sequences = file('sequences_' + self.fileName, 'w')
        components = file('components_' + self.fileName, 'w')
        samplers = file('samplers_' + self.fileName, 'w')

        for e in self.saveline.elementList :
            self._removeConflictingGMADKeywords(self.saveline.elementDict[e], e)

        # if file exists overwrite, how?
        for i in self.saveline.sequences :
            sequences.write(i + '\n')
        for c in self.saveline.components :
            components.write(c + '\n')
        for s in self.saveline.samplers :
            samplers.write(s + '\n')

        sequences.close()
        components.close()
        samplers.close()

    def _removeConflictingGMADKeywords(self, elements, e) :

        if e == 'DUMP' :
            e = 'PMUD'
        temp = ''
        wasLine = False

        for element in elements :
            if element == 'LINE' :
                wasLine = True
                if type(element) == str :
                    temp += element + '=('
            elif element == 'MULTIPOLE' :
                temp = 'MARKER; #'

            else :
                if type(element) == list :
                    for prop in element :
                        if 'APERTURE=' in prop : #Need to remove APERTURE
                            continue
                        if prop == 'DUMP' : #Need to rename DUMP as conflicting keyword.
                            prop = 'PMUD'
                        self.saveline.samplers.append('sample, range = ' + prop)
                        temp += prop + ', '

                    temp = temp[:-2]

                    if wasLine :
                        temp += ')'
                        self.saveline.sequences.append(e + ': ' + temp)


                if type(element) == str :
                        temp += element + ', '

                elif not wasLine :
                    self.saveline.components.append(e + ': ' + temp)
                else :
                    wasLine = False

    def _formatMad82Gmad(self, mad8FileName, gmadFileName, removeAperture = True, removeMultipole = True) :
        i = open(mad8FileName,'r')
        o = open(gmadFileName,'w')

        # line for analysis
        lta = str()
        # boolian analysis or not
        ba  = False

        for l in i :
            # check if continued line
            l = l.strip('\n ')
            t = l.split()
            # if last token is an "&" then join the line
            if len(t) > 0 :
                if t[0] == '!' and t[-1] != '&' :
                    lta = l
                    ba  = False
                    o.write(lta+'\n')
                    continue
                elif l[-1] == '&' :
                    lta = lta+l[0:-1]
                    ba  = False
                elif len(lta) != 0 :
                    lta = lta+l[0:]
                    ba  = True
                else :
                    lta = l
                    ba  = True
            else :
                lta = l
                o.write(lta+'\n')
                continue

            if ba :
    #            print 'l  >',l
    #            print 'lta>',lta

                # lower case
                lta = lta.lower()
                # append semicolon
                lta = lta+';'

                # remove mad specific commands (should be saveline so no
                # problems but...)
                if lta.find('plot')     != -1 : lta = ''
                if lta.find('assign')   != -1 : lta = ''
                if lta.find('envelope') != -1 : lta = ''
                if lta.find('twiss')    != -1 : lta = ''
                if lta.find('survey')   != -1 : lta = ''
                if lta.find('title')    != -1 : lta = ''
                if lta.find('option')   != -1 : lta = ''
                if lta.find('optics')   != -1 : lta = ''
                if lta.find('btrns')    != -1 : lta = ''
                if lta.find('print')    != -1 : lta = ''
                if lta.find('return')   != -1 : lta = ''
                if lta.find('saveline') != -1 : lta = ''
                if lta.find('#') != -1 : lta = '!'

                # check for 4 letter commands
                if lta.find('sben') != -1 and lta.find('sbend') == -1 :
                    lta = lta.replace('sben','sbend')
                if lta.find('drif') != -1 and lta.find('drift') == -1 :
                    lta = lta.replace('drif','drift')
                if lta.find('sext') != -1 and lta.find('sextupole') == -1 :
                    lta = lta.replace('sext','sextupole')
                if lta.find('quad') != -1 and lta.find('quadrupole') == -1 :
                    lta = lta.replace('quad','quadrupole')
                if lta.find('octu') != -1 and lta.find('octupole') == -1 :
                    lta = lta.replace('octu','octupole')
                if lta.find('mark') != -1 and lta.find('marker') == -1 :
                    lta = lta.replace('mark','marker')
                if lta.find('moni') != -1 and lta.find('monitor') == -1 :
                    lta = lta.replace('moni','monitor')

                # change unsuported types
                lta = lta.replace('monitor','marker');
                lta = lta.replace('wire','marker');
                lta = lta.replace('prof','marker');

                # remove aperture parameters
                iap = lta.find('aperture')
                if removeAperture and (iap != -1) :
                    lta = lta[0:iap-2]+'; ! removed aperture'

                # remove multipoles
                if removeMultipole and (lta.find('multipole') != -1) :
                    lta = '! removed multipole'+lta
                o.write(lta+'\n')
                # clean up
                ba = False
                lta = str()

        i.close()
        o.close()
