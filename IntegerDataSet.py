import numpy as np

def CreateIntegerDataWithDispersedFRB(startTime=1,DM=0,inputArray=None,
                                      lenSeries=65536,numChannels=4096,
                                      fMin=400,fMax=800,sampTime=1E-3):
    """ 
    Returns an array of zeros with dtype int16. Adds an dispersed FRB signal to inputArray (optional). This is intended as a numerical precision test of the FDMT algorithm vs brute force.
    The background will be zeros and the FRB will have an amplitude of 1/channel and a width of 1 time bin.

    Input:
    startTime=1,      : FRB start time (seconds)
    DM=0,             : DM in pc cm^-3
    inputArray=None,  : inputArray (of integers, not checked). If you want to add multiple FRBs
    lenSeries=65536,  : Number of time bins. Taken from inputArray size if specified.
    numChannels=4096, : Num freq channels. Ditto as above.
    fMin=400,         : min freq (MHz)
    fMax=800,         : max freq (MHz)
    sampTime=1E-3     : time bin size (seconds)
    """

    if inputArray is None:
        inputArray = np.zeros((numChannels,lenSeries),dtype=np.int16)
    else:
        inputArray = np.int16(inputArray)

        numChannels,lenSeries = inputArray.shape

    ## Calculate time axis indices for every freq channel (i.e. calculate the DM curve).
    df = (fMax - fMin)/np.float(numChannels)  #bandwidth per channel
    f = np.arange(fMin,fMax,df)+df/2          #(adding half df to get the center of the band) 

    dt = 4.148808*DM*1E3*(1/f**2-1/fMax**2)/sampTime # This gives the timeBin shifts. 
                                                     # The -1/fMax**2 sets high freq origin = 0

    ## now shift the dispersion curve by startTime
    dt = np.int32(dt+startTime/sampTime)

    for i in range(numChannels):
        ## Make sure that the max is within lenSeries
        if dt[i]<lenSeries and dt[i]>=0:
            inputArray[i,dt[i]] += 1
        

    return inputArray

    
    
    
    
    
    
