import Output as _Output

def MakeTrackFiles(savelineFileName, line, outputFileNameStub) : 
    sl  = _Output.Saveline(savelineFileName, line)
    sl.removeDuplicates()
    sl.writeRenamed(outputFileNameStub+"_renamed.mad8")
    MakeObserveFile(sl.expandedLineRenamed[0:-1:3],outputFileNameStub+"_observe.mad8")
    MakeTableArchiveFile(sl.expandedLineRenamed[0:-1:3],outputFileNameStub+"_archive.mad8")
    MakeTrackCallingFile(outputFileNameStub)

def MakeTrackCallingFile(fileNameStub) : 
    pass

def MakeObserveFile(elementlist,filename) : 
    f = open(filename,'w') 
    for e in elementlist : 
        ws = 'OBSERVE, PLACE="'+e+'", TABLE="'+e+'"\n'
        f.write(ws)
    f.close()

def MakeTableArchiveFile(elementlist, filename) :
    f = open(filename,'w')
    for e in elementlist :
        ws = 'ARCHIVE, TABLE="'+e+'", FILENAME="./track/'+e+'"\n'
        f.write(ws)
    f.close()
