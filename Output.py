import Input as _Input

class Saveline :
    def __init__(self, fileName) : 
        self.fileName = fileName
        self.readFile(self.fileName)
        self.dictFile = []
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
            
    def expandLine(self, lineName = "EBDS") : 
        line = self.findNamedDict(lineName)['LINE']
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
