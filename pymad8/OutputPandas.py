import pandas as _pd
import fortranformat as _ff

class OutputPandas : 

	def __init__(self, filename, filetype = 'twiss'):
		if filetype == 'twiss':
			self._readTwissFile(filename)

	def _readTwissFile(self, filename) :
		f = open(filename,'r')

		ffhr1 = _ff.FortranRecordReader('(5A8,I8,L8,I8)')
		ffhr2 = _ff.FortranRecordReader('(A80)')

		# read header
		h1 = ffhr1.read(f.readline())
		h2 = ffhr2.read(f.readline())
		nrec = h1[7]

		print('Mad8.readTwissFile > nrec='+str(nrec))

		ffe1 = _ff.FortranRecordReader('(A4,A16,F12.6,4E16.9,A19,E16.9)')
		ffe2 = _ff.FortranRecordReader('(5E16.9)')

		# loop over entries
		l_all = []
		for i in range(0,nrec,1) :
			l = []
			l1 = ffe1.read(f.readline())
			l2 = ffe2.read(f.readline())
			l3 = ffe2.read(f.readline())
			l4 = ffe2.read(f.readline())
			l5 = ffe2.read(f.readline())
			# print(l1)
			l.extend(l1)
			l.extend(l2)
			l.extend(l3)
			l.extend(l4)
			l.extend(l5)

			l_all.append(l)

		self.data = _pd.DataFrame(l_all)
