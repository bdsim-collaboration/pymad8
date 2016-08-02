def MakeObserveFile(elementlist,filename) : 
    f = open(filename,'w') 
    for e in elementlist : 
        ws = 'OBSERVE, PLACE="'+e+'", TABLE="'+e+'"\n'
        print ws
        f.write(ws)
    f.close()

def MakeTableArchiveFile(elementlist, filename) :
    f = open(filename,'w')
    for e in elementlist :
        ws = 'ARCHIVE, TABLE="'+e+'", FILENAME="./track/'+e+'"\n'
        print ws
        f.write(ws)
    f.close()
