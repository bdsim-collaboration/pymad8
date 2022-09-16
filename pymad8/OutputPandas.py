import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd
import fortranformat as _ff

##########################################################################################
class OutputPandas : 
	'''Class to load different Mad8 output files in a Pandas DataFrame
	twiss = pymad8.OutputPandas('/twiss.tape','twiss')
	rmat = pymad8.OutputPandas('/rmat.tape','rmat')
	chrom = pymad8.OutputPandas('/chrom.tape','chrom')
	envel = pymad8.OutputPandas('/envel.tape','envel')
	survey = pymad8.OutputPandas('/survey.tape','survey')

	By default the filetype is twiss'''

	def __init__(self, filename, filetype = 'twiss'):
		'''Take filename for argument, filetype if specified and save them as internal variables
		Then test for filetype and call the corresponding subfunction'''
		self.filename = filename
		self.filetype = filetype

		self.keys_dict = {'    '       :{},
        'DRIF'       :{'L':2,                                                                      'APER':11,'NOTE':12,'E':13},
        'RBEN'       :{'L':2,'ANGLE':3,'K1':4,'K2':5,'TILT' :6,'E1'    :7,'E2'  :8,'H1' :9,'H2':10,'APER':11,'NOTE':12,'E':13},
        'SBEN'       :{'L':2,'ANGLE':3,'K1':4,'K2':5,'TILT' :6,'E1'    :7,'E2'  :8,'H1' :9,'H2':10,'APER':11,'NOTE':12,'E':13},
        'QUAD'       :{'L':2,          'K1':4,       'TILT' :6,                                    'APER':11,'NOTE':12,'E':13},
        'SEXT'       :{'L':2,                 'K2':5,'TILT' :6,                                    'APER':11,'NOTE':12,'E':13},
        'OCTU'       :{'L':2,                        'TILT' :6,'K3'    :7,                         'APER':11,'NOTE':12,'E':13},
        'MULT'       :{      'K0'   :3,'K1':4,'K2':5,'T0'   :6,'K3'    :7,'T1'  :8,'T2' :9,'T3':10,'APER':11,'NOTE':12,'E':13},
        'SOLE'       :{'L':2,                                  'KS'    :7,                         'APER':11,'NOTE':12,'E':13},
        'RFCAVITY'   :{'L':2,                                  'FREQ'  :7,'VOLT':8,'LAG':9,        'APER':11,'NOTE':12,'E':13},
        'ELSEPARATOR':{'L':2,                        'TILT' :6,'EFIELD':7,                         'APER':11,'NOTE':12,'E':13},
        'KICK'       :{'L':2,                        'HKIC' :6,'VKIC'  :7,                         'APER':11,'NOTE':12,'E':13},
        'HKIC'       :{'L':2,                        'HKIC' :6,                                    'APER':11,'NOTE':12,'E':13},
        'VKIC'       :{'L':2,                                  'VKIC'  :7,                         'APER':11,'NOTE':12,'E':13},
        'SROT'       :{'L':2,                                  'ANGLE' :7,                         'APER':11,'NOTE':12,'E':13},
        'YROT'       :{'L':2,                                  'ANGLE' :7,                         'APER':11,'NOTE':12,'E':13},
        'MONI'       :{'L':2,                                                                      'APER':11,'NOTE':12,'E':13},
        'HMONITOR'   :{'L':2,                                                                      'APER':11,'NOTE':12,'E':13},
        'VMONITOR'   :{'L':2,                                                                      'APER':11,'NOTE':12,'E':13},
        'MARK'       :{'L':2,                                                                      'APER':11,'NOTE':12,'E':13},
        'ECOL'       :{'L':2,                        'XSIZE':6,'YSIZE' :7,                         'APER':11,'NOTE':12,'E':13},
        'RCOL'       :{'L':2,                        'XSIZE':6,'YSIZE' :7,                         'APER':11,'NOTE':12,'E':13},
        'INST'       :{'L':2,                                                                                'NOTE':12,'E':13},
        'WIRE'       :{'L':2,                                                                                'NOTE':12,'E':13},
        'IMON'       :{'L':2,                                                                                'NOTE':12,'E':13},
        'PROF'       :{'L':2,                                                                                'NOTE':12,'E':13},
        'BLMO'       :{'L':2,                                                                                'NOTE':12,'E':13},
        'LCAV'       :{'L':2,                                  'FREQ'  :7,'VOLT':8,'LAG':9,        'APER':11,'NOTE':12,'E':13},
        'MATR'       :{'L':2,                                                                      'APER':11,          'E':13}}

		self.colnames_common = ['TYPE','NAME','L','ANGLE','K0','K1','K2','K3','KS','T0','T1','T2','T3',     \
					'TILT','E1','E2','H1','H2','APER','NOTE','E','FREQ','VOLT','LAG','EFIELD',     \
					'HKIC','VKIC','XSIZE','YSIZE']

		if filetype == 'twiss':
			self._readTwissFile()
		if filetype == 'rmat':
			self._readRmatFile()
		if filetype == 'chrom':
			self._readChromFile()
		if filetype == 'envel':
			self._readEnvelopeFile()
		if filetype == 'survey':
			self._readSurveyFile()

	##########################################################################################
	def _readTwissFile(self) :
		'''Read and load a Mad8 twiss file in a DataFrame then save it as internal value 'data' '''
		
		colnames_twiss = ['ALPHX','BETX','MUX','DX','DPX','ALPHY','BETY','MUY','DY','DPY','X','PX','Y','PY','S']
		colnames = self.colnames_common + colnames_twiss
		self.data = _pd.DataFrame(columns=colnames)

		f = open(self.filename,'r')

		self.nrec = self._findNelemInFF(f)
		print('Mad8.readTwissFile > nrec='+str(self.nrec))

		ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
		ffe2 = _ff.FortranRecordReader('(5E16.9)')

		# loop over entries
		dList = []
		for i in range(0,self.nrec,1) :
			l1 = ffe1.read(f.readline())
			l2 = ffe2.read(f.readline())
			l3 = ffe2.read(f.readline())
			l4 = ffe2.read(f.readline())
			l5 = ffe2.read(f.readline())
			l_common = l1[0:6]+l2+l1[6:9]
			l_twiss = l3+l4+l5
			d = {'TYPE':l_common[0],'NAME':l_common[1].strip()}
			kt = self.keys_dict[d['TYPE']]
			for k in kt.keys() :
				d[k]=l_common[kt[k]]
			for i in range(len(colnames_twiss)) :
				d[colnames_twiss[i]]=l_twiss[i]
			dList.append(d)
		f.close()
		self.data = _pd.DataFrame(dList,columns=colnames)
		self.data.at[0,'E'] = self.data['E'][1]

	##########################################################################################
	def _readRmatFile(self) :
		'''Read and load a Mad8 rmat file in a DataFrame then save it as internal value 'data' '''

		colnames_rmat = ['R11','R12','R13','R14','R15','R16', 'R21','R22','R23','R24','R25','R26', \
				 'R31','R32','R33','R34','R35','R36', 'R41','R42','R43','R44','R45','R46', \
				 'R51','R52','R53','R54','R55','R56', 'R61','R62','R63','R64','R65','R66','S']
		colnames = self.colnames_common + colnames_rmat
		self.data = _pd.DataFrame(columns=colnames)

		f = open(self.filename,'r')

		self.nrec = self._findNelemInFF(f)
		print('Mad8.readRmatFile > nrec='+str(self.nrec))

		ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
		ffe2 = _ff.FortranRecordReader('(5E16.9)')
		ffe3 = _ff.FortranRecordReader('(6E16.9)')
		ffe4 = _ff.FortranRecordReader('(7E16.9)')
		
		# loop over entries
		for i in range(0,self.nrec,1) :
			l1 = ffe1.read(f.readline())
			l2 = ffe2.read(f.readline())
			l3 = ffe3.read(f.readline())
			l4 = ffe3.read(f.readline())
			l5 = ffe3.read(f.readline())
			l6 = ffe3.read(f.readline())
			l7 = ffe3.read(f.readline())
			l8 = ffe4.read(f.readline())
			l_common = l1[0:6]+l2+l1[6:9]
			l_rmat = l3+l4+l5+l6+l7+l8

			d = {'TYPE':l_common[0],'NAME':l_common[1].strip()}
			kt = self.keys_dict[d['TYPE']]
			for k in kt.keys() :
				d[k]=l_common[kt[k]]
			for i in range(len(colnames_rmat)) :
				d[colnames_rmat[i]]=l_rmat[i]
			self.data = _pd.concat([self.data, _pd.DataFrame(d, index=[0])], ignore_index=True)
		f.close()
		self.data.at[0,'E'] = self.data['E'][1]

	##########################################################################################
	def _readChromFile(self) :
		'''Read and load a Mad8 chrom file in a DataFrame then save it as internal value 'data' '''

		colnames_chrom = ['WX','PHIX','DMUX','DDX','DDPX','', \
				  'WY','PHIY','DMUY','DDY','DDPY','', \
				  'X','PX','Y','PY','S','']
		colnames = self.colnames_common + colnames_chrom
		self.data = _pd.DataFrame(columns=colnames)

		f = open(self.filename,'r')
		
		self.nrec = self._findNelemInFF(f)
		print('Mad8.readChromFile > nrec='+str(self.nrec))

		ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
		ffe2 = _ff.FortranRecordReader('(5E16.9)')
		ffe3 = _ff.FortranRecordReader('(6E16.9)')

		# loop over entries
		for i in range(0,self.nrec,1) :
			l1 = ffe1.read(f.readline())
			l2 = ffe2.read(f.readline())
			l3 = ffe3.read(f.readline())
			l4 = ffe3.read(f.readline())
			l5 = ffe3.read(f.readline())
			l_common = l1[0:6]+l2+l1[6:9]
			l_chrom = l3+l4+l5

			d = {'TYPE':l_common[0],'NAME':l_common[1].strip()}
			kt = self.keys_dict[d['TYPE']]
			for k in kt.keys() :
				d[k]=l_common[kt[k]]
			for i in range(len(colnames_chrom)) :
				d[colnames_chrom[i]]=l_chrom[i]
			self.data = _pd.concat([self.data, _pd.DataFrame(d, index=[0])], ignore_index=True)
		f.close()
		self.data.at[0,'E'] = self.data['E'][1]

	##########################################################################################
	def _readEnvelopeFile(self) :
		'''Read and load a Mad8 envelope file in a DataFrame then save it as internal value 'data' '''

		colnames_envelop = ['S11','S12','S13','S14','S15','S16', 'S21','S22','S23','S24','S25','S26',
				    'S31','S32','S33','S34','S35','S36', 'S41','S42','S43','S44','S45','S46',
				    'S51','S52','S53','S54','S55','S56', 'S61','S62','S63','S64','S65','S66','S']
		colnames = self.colnames_common + colnames_envelop
		self.data = _pd.DataFrame(columns=colnames)

		f = open(self.ilename,'r')

		self.nrec = self._findNelemInFF(f)
		print('Mad8.readEnvelopeFile > nrec='+str(self.nrec))

		ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
		ffe2 = _ff.FortranRecordReader('(5E16.9)')
		ffe3 = _ff.FortranRecordReader('(6E16.9)')
		ffe4 = _ff.FortranRecordReader('(7E16.9)')

		# loop over entries
		for i in range(0,self.nrec,1) :
			l1 = ffe1.read(f.readline())
			l2 = ffe2.read(f.readline())
			l3 = ffe3.read(f.readline())
			l4 = ffe3.read(f.readline())
			l5 = ffe3.read(f.readline())
			l6 = ffe3.read(f.readline())
			l7 = ffe3.read(f.readline())
			l8 = ffe4.read(f.readline())
			l_common = l1[0:6]+l2+l1[6:9]
			l_envelop = l3+l4+l5+l6+l7+l8

			d = {'TYPE':l_common[0],'NAME':l_common[1].strip()}
			kt = self.keys_dict[d['TYPE']]
			for k in kt.keys() :
				d[k]=l_common[kt[k]]
			for i in range(len(colnames_envelop)) :
				d[colnames_envelop[i]]=l_envelop[i]
			self.data = _pd.concat([self.data, _pd.DataFrame(d, index=[0])], ignore_index=True)
		f.close()
		self.data.at[0,'E'] = self.data['E'][1]

	##########################################################################################
	def _readSurveyFile(self) :
		'''Read and load a Mad8 survey file in a DataFrame then save it as internal value 'data' '''

		colnames_survey = ['X','Y','Z','S','THETA','PHI','PSI']
		colnames = self.colnames_common + colnames_survey
		self.data = _pd.DataFrame(columns=colnames)

		f = open(self.filename,'r')

		self.nrec = self._findNelemInFF(f)
		print('Mad8.readSurveyFile> nrec='+str(self.nrec))

		ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
		ffe2 = _ff.FortranRecordReader('(5E16.9)')
		ffe3 = _ff.FortranRecordReader('(4E16.9)')
		ffe4 = _ff.FortranRecordReader('(3E16.9)')

		# loop over entries
		for i in range(0,self.nrec,1) :
			l1 = ffe1.read(f.readline())
			l2 = ffe2.read(f.readline())
			l3 = ffe3.read(f.readline())
			l4 = ffe4.read(f.readline())
			l_common = l1[0:6]+l2+l1[6:9]
			l_survey = l3+l4

			d = {'TYPE':l_common[0],'NAME':l_common[1].strip()}
			kt = self.keys_dict[d['TYPE']]
			for k in kt.keys() :
				d[k]=l_common[kt[k]]
			for i in range(len(colnames_survey)) :
				d[colnames_survey[i]]=l_survey[i]
			self.data = _pd.concat([self.data, _pd.DataFrame(d, index=[0])], ignore_index=True)
		f.close()
		self.data.at[0,'E'] = self.data['E'][1]

	##########################################################################################
	def _findNelemInFF(self,openfile):
		'''Read the beginig of an opened fortran file and return the number of elements'''
		if self.filetype == 'chrom':
			# 3 random lines at the end of chrom tape
			openfile.readline()
			openfile.readline()
			openfile.readline()

		# Standard header definition 
		ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
		ffhr2 = _ff.FortranRecordReader('(A80)')
		# read header
		h1 = ffhr1.read(openfile.readline())
		h2 = ffhr2.read(openfile.readline())
		# number of records
		nrec = h1[7]

		return nrec

	def Clear(self):
		'''Empties all data structures in this instance'''
		self.__init__()

	####################################################################################
	def getIndexByNames(self,namelist):
		rows = self.getRowsByNames(namelist)
		if len(rows) == 1 :
			return rows.index[:].tolist()[0]
		return rows.index[:].tolist()

	def getIndexByTypes(self,typelist):
		rows = self.getRowsByTypes(typelist)
		if len(rows) == 1 :
			return rows.index[:].tolist()[0]
		return rows.index[:].tolist()

	def getIndexByValues(self,key,value=0,mode='supp'):
		rows = self.getRowsByValues(key,value,mode)
		if len(rows) == 1 :
			return rows.index[:].tolist()[0]
		return rows.index[:].tolist()

	def getIndexByNearestS(self,s):
		row = self.getRowByNearestS(s)
		return row.index[:].tolist()[0]

	####################################################################################
	def getNamesByIndex(self,indexlist):
		if type(indexlist) == list :
			return self.getRowsByIndex(indexlist)['NAME'].tolist()
		return self.getRowsByIndex(indexlist)['NAME'].tolist()[0]

	def getNamesByTypes(self,typelist):
		if type(typelist) == list :
			return self.getRowsByTypes(typelist)['NAME'].tolist()
		names = self.getRowsByTypes(typelist)['NAME'].tolist()
		if len(names) == 1:
			return names[0]
		return names

	def getNamesByValues(self,key,value=0,mode='supp'):
		rows =  self.getRowsByValues(key,value,mode)
		if len(rows) == 1 :
			return rows['NAME'].tolist()[0]
		return rows['NAME'].tolist()

	def getNameByNearestS(self,s):
		row = self.getRowByNearestS(s)
		return row['NAME'].tolist()[0]

	####################################################################################
	def getTypesByIndex(self,indexlist):
		if type(indexlist) == list :
			return self.getRowsByIndex(indexlist)['TYPE'].tolist()
		return self.getRowsByIndex(indexlist)['TYPE'].tolist()[0]

	def getTypesByNames(self,namelist):
		if type(namelist) == list :
			return self.getRowsByNames(namelist)['TYPE'].tolist()
		types = self.getRowsByNames(namelist)['TYPE'].tolist()
		if len(types) == 1:
			return types[0]
		return types

	def getTypesByValues(self,key,value=0,mode='supp'):
		rows = self.getRowsByValues(key,value,mode)
		if len(rows) == 1 :
			return rows['TYPE'].tolist()[0]
		return rows['TYPE'].tolist()

	def getTypeByNearestS(self,s):
		row = self.getRowByNearestS(s)
		return row['TYPE'].tolist()[0]

	####################################################################################
	def getRowsByIndex(self,indexlist):
		if type (indexlist) == list :
			return self.data.loc[indexlist]
		return self.data.loc[[indexlist]]

	def getRowsByNames(self,namelist):
		if type(namelist) == list :
			return self.data.loc[self.data['NAME'].isin(namelist)]
		return self.data.loc[self.data['NAME'] == namelist]

	def getRowsByTypes(self,typelist):
		if type(typelist) == list :
			return self.data.loc[self.data['TYPE'].isin(typelist)]
		return self.data.loc[self.data['TYPE'] == typelist]

	def getRowsByValues(self,key,value=0,mode='supp'):
		if mode == 'supp':
			return self.data.loc[self.data[key] > value]
		if mode == 'inf':
			return self.data.loc[self.data[key] < value]
		raise ValueError("Unrecognized mode, use 'supp' or 'inf'")

	def getRowByNearestS(self,s):
		S = self.data['S'].tolist()
		for index in range(self.nrec) :
			if s > S[index-1] and s < S[index] :
				if s - S[index-1] < S[index] - s :
					return self.data.loc[[index-1]]
				else :
					return self.data.loc[[index]]
		raise ValueError('Not found')

	def getRowsByFunction(self,f):
		return self.data.loc[f(self.data)]

	####################################################################################
	def getColumnsByKeys(self,keylist):
		return self.data[keylist]

	####################################################################################
	def getElementByIndex(self,indexlist,keylist):
		if type(indexlist) != list :
			indexlist = [indexlist]
		if type(keylist) != list :
			keylist = [keylist]
		elem = self.data.loc[indexlist,keylist]
		if elem.shape == (1,1) :
			return elem.values[0][0]
		return elem

	def getElementByNames(self,namelist,keylist):
		if type(namelist) != list :
			namelist = [namelist]
		if type(keylist) != list :
			keylist = [keylist]
		elem = self.data.loc[self.data['NAME'].isin(namelist),keylist]
		if elem.shape == (1,1) :
			return elem.values[0][0]
		return elem
	####################################################################################
	def getAperture(self,index,defaultAperSize=0.1):
		name = self.getNamesByIndex(index)
		aperture = self.data['APER'][index]
		if aperture == 0 or aperture == _np.nan :
			aperture = defaultAperSize
		return aperture

	def sMin(self):
		return self.data['S'][0]

	def sMax(self):
		return self.data['S'][self.nrec-1]

	def plotXY(self,Xkey,Ykey):
		X = self.getColumnsByKeys(Xkey)
		Y = self.getColumnsByKeys(Ykey)
		_plt.plot(X,Y)

	def calcBeamSize(self, EmitX, EmitY, Esprd, BunchLen=0):
		'''Calculate the beam sizes and beam divergences in both planes for each elements
		Then the four columns are added at the end of the DataFrame
		Works only if a twiss file was loaded previously'''
		if self.filetype != 'twiss':
			raise ValueError('The loaded file needs to be a twiss file')

		SigmaX = []
		SigmaY = []
		SigmaXP = []
		SigmaYP = []
		E0 = self.data['E'][1]
		for i in range(0,self.nrec,1) :
			BetaX = self.data['BETX'][i]
			BetaY = self.data['BETY'][i]
			GammaX = (1+self.data['ALPHX'][i]**2)/BetaX
			GammaY = (1+self.data['ALPHY'][i]**2)/BetaY
			DispX = self.data['DX'][i]
			DispY = self.data['DY'][i]
			DispXP = self.data['DPX'][i]
			DispYP = self.data['DPY'][i]

			# Beam size calculation
			SigmaX.append(_np.sqrt(BetaX*EmitX+(DispX*Esprd/E0)**2))
			SigmaY.append(_np.sqrt(BetaY*EmitY+(DispY*Esprd/E0)**2))
			# Beam divergence calculation
			SigmaXP.append(_np.sqrt(GammaX*EmitX+(DispXP*Esprd/E0)**2))
			SigmaYP.append(_np.sqrt(GammaY*EmitY+(DispYP*Esprd/E0)**2))

		self.data = self.data.assign(SIGX=SigmaX)
		self.data = self.data.assign(SIGY=SigmaY)
		self.data = self.data.assign(SIGXP=SigmaXP)
		self.data = self.data.assign(SIGYP=SigmaYP)
