def tidy(input) :     
    '''tidy input, remove EOL, remove empty lines
       input : list of file lines 
    ''' 
    output = []

    for l in input : 
        l  = l.strip(' \n')   # tidy end of lines 
        if len(l) == 0 :      # strip empty lines 
            continue 
        if l.find("RETURN") != -1 : 
            continue
        output.append(l)
    return output

def removeContinuationSymbols(input) : 
    '''remove continuation symbols from input
       input : list of file lines'''

    output = []
    ml     = '' # merged line 
    # fine line continuations
    for l in input : 
        ai = l.find('&') 
        
        if ai != -1 : 
            l = l.replace('&','')
            ml = ml+l
        else : 
            if len(ml) == 0 :
                ml = l
            else :
                ml = ml+l
            output.append(ml)
            ml = ''

    return output


def removeComments(input) : 
    '''remove comment lines'''
    output = [] 
    for l in input : 
        if l[0] == '!' :
            continue 
        else : 
            output.append(l)

    return output

def decodeFileLine(input) : 
    '''decode line
       input is a string of a mad8 line'''


    splitInput = input.split();    
    
    for i in range(0,len(splitInput)) : 
        splitInput[i] = splitInput[i].strip(',')

    d = dict() 

    if len(splitInput) == 1 : 
        pass
    elif len(splitInput) > 1 : 
        type = splitInput[1].strip(',');
        if type == 'LINE' :
            if len(splitInput) == 4 :
                input = input.replace(',',' ')
                splitInput = input.split()

            
            d = decodeLine(splitInput)
        elif type == 'INSTRUMENT' : 
            d = decodeNamed(splitInput)
        elif type == 'MONITOR' :
            d = decodeNamed(splitInput) 
        elif type == 'WIRE' : 
            d = decodeNamed(splitInput) 
        elif type == 'MARKER' : 
            d = decodeNamed(splitInput) 
        elif type == 'PROFILE' : 
            d = decodeNamed(splitInput) 
        elif type == 'LCAVITY' :
            d = decodeLcavity(splitInput)
        elif type == 'DRIFT' : 
            d = decodeDrift(splitInput) 
        elif type == 'SBEND' : 
            d = decodeSbend(splitInput)             
        elif type == 'QUADRUPOLE' : 
            d = decodeQuadrupole(splitInput)             
        elif type == 'SEXTUPOLE' : 
            d = decodeSextupole(splitInput)
        elif type == 'OCTUPOLE' : 
            d = decodeOctupole(splitInput)
        elif type == 'DECAPOLE' : 
            d = decodeDecapole(splitInput)
        elif type == 'MULTIPOLE' : 
            d = decodeMultipole(splitInput)
        elif type == 'VKICKER' : 
            d = decodeKicker(splitInput)
        elif type == 'HKICKER' : 
            d = decodeKicker(splitInput)
        elif type == 'RCOLLIMATOR' : 
            d = decodeCollimator(splitInput)
        elif type == 'ECOLLIMATOR' : 
            d = decodeCollimator(splitInput)
        else :
            if len(splitInput) == 2 : 
                d = decodeNamed(splitInput)
            else :
                d = decodeLcavity(splitInput)

    return d

def decodeLine(input) :
    d = decodeNameAndType(input) 

    input[3] = input[3].replace('(','')
    input[-1] = input[-1].replace(')','')    
    d['LINE'] = input[3:]    

        
    return d

def decodeNamed(input) :
    d = decodeNameAndType(input) 
    return d

def decodeLcavity(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeDrift(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeSbend(input) : 
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d    

def decodeQuadrupole(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeSextupole(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeOctupole(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeDecapole(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeMultipole(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeKicker(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeCollimator(input) :
    d = decodeNameAndType(input) 
    for t in input[2:] : 
        [key,val] = splitKeyValue(t)
        d[key] = val
    return d

def decodeNameAndType(input) : 
    name = input[0].strip(':')
    type = input[1]
    d = dict()
    d['name'] = name
    d['type'] = type
    return d 

def splitKeyValue(t) : 
    st = t.split('=')
    key   = st[0]
    value = float(st[1])

    return [key,value]
