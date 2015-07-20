Licence = """
<OWNER> = Shriharsh Tendulkar (McGill)
<YEAR> = 2015

Module to generate simulated data from CHIME. Inserts fake signals as requested.
"""

import numpy as np
from numpy import random
from matplotlib import pyplot as plt

VERBOSE = True

class AmplitudeTimeSeries:
    """ 
    Defines a class of amplitude time series. These are complex voltages coming out of the telescope. 
    The number of channels can be single or multiple (for example filterbank data)
    Frequencies are always handled in MHz. Time is always handled in seconds
    
    Attributes:
    timeSeries -- complex array of amplitude timeseries
    numChannels -- scalar integer, minimum 1
    fMin, fMax -- minimum and maximum frequency. The channels are assumed to be equispaced for now.
    """
    def __init__(self,timeSeries=None,
                 lenSeries=2**18,
                 numChannels=1,
                 fMin=400,fMax=800,
                 sampTime=None,
                 noiseRMS=0.1):
        """ Initializes the AmplitudeTimeSeries instance. 
        If a array is not passed, then a random whitenoise dataset is generated.
        Inputs: 
        Len -- Number of time data points (usually a power of 2) 2^38 gives about 65 seconds 
        of 400 MHz sampled data
        The time binning is decided by the bandwidth
        fMin -- lowest frequency (MHz)
        fMax -- highest frequency (MHz)
        noiseRMS -- RMS value of noise (TBD)
        noiseAlpha -- spectral slope (default is white noise) (TBD)
        ONLY GENERATES WHITE NOISE RIGHT NOW!
        """
        self.shape = (np.uint(lenSeries),np.uint(numChannels))
        self.fMax = fMax
        self.fMin = fMin        
        
        if sampTime is None:
            self.sampTime = np.uint(numChannels)*1E-6/(fMax-fMin)
        else:
            self.sampTime = sampTime

        if timeSeries is None:
            # then use the rest of the data to generate a random timeseries
            if VERBOSE:
                print "AmplitudeTimeSeries __init__ did not get new data, generating white noise data"

            self.timeSeries = np.complex64(noiseRMS*(np.float16(random.standard_normal(self.shape))
                                                     +np.float16(random.standard_normal(self.shape))*1j)/np.sqrt(2))
            
        else:
            if VERBOSE:
                print "AmplitudeTimeSeries __init__ got new data, making sure it is reasonable."

            if len(timeSeries.shape) == 1:
                self.shape = (timeSeries.shape[0],1)
                
            else:
                self.shape = timeSeries.shape

            self.timeSeries = np.reshape(np.complex64(timeSeries),self.shape)
            
            self.fMin = fMin
            self.fMax = fMax

            if sampTime is None:
                self.sampTime = numChannels*1E-6/(fMax-fMin)
            else:
                self.sampTime = sampTime

        return None

    def InjectFRB(self,epochFRB=0,ampFRB=1,decayFRB=5,riseFRB=0):
        """ 
        Takes a time series, injects an FRB with a given shape and amplitude at a certain epoch.
        Inputs:
        
        timeSeries -- numpy array of some length, with complex variables.
        sampTime -- time sampling of the timeseries to define the scale for 
        epochFRB -- location of the FRB peak from the start of the time series (seconds).
        ampFRB -- amplitude of the FRB peak. NOTE: this is not the total power. This might get re-written later.
        decayFRB -- FRB decay time for the scattering tail (seconds)
        riseFRB -- FRB rise time (seconds). Default is zero.
        """
        ## the complicated expression is just to the get right shape.
        timePoints = (np.meshgrid(np.arange(0,self.shape[1]),
                                  np.arange(0,self.shape[0]))[1])*self.sampTime - epochFRB
        
        ampFRB = ampFRB/self.shape[1] ## reduce the amplitude based on the number of channels
        
        self.timeSeries += FastRiseExpDecay(timePoints,ampFRB,decayFRB,riseFRB)

    def CoherentDisperse(self,DM):
        """ 
        Takes in a raw complex time series, applies dispersion and returns.
        fMin and fMax are in MHz
        sampTime is in seconds.
        DM is in pc cm^{-3}
        Currently implemented only for numChannels=1
        """ 

        assert self.shape[1]==1, "This is not currently implemented for multiple channels!!"

        nTotal = len(self.timeSeries)
        f = np.arange(0,self.fMax-self.fMin, float(self.fMax-self.fMin)/nTotal) # this is in MHz
        
        DM = DM*4.148808*1E9 #DispersionConstant = 4.148808*10**9
        
        # The added linear term makes the arrival times of the highest frequencies be 0
        # adapted from Barak Zakay's code.
        H = np.exp(-(2*np.pi*1j * DM /(self.fMin + f) + 2*np.pi*1j*DM*f /(self.fMax**2)))
        
        self.timeSeries = np.fft.ifft(np.fft.fft(self.timeSeries) * H)

    def ApplyFilterBank(self,newNumChannels):
        """ 
        Applies a filterbank to the channels. Takes in the fMin to fMax data and rebins it into more channels.
        The sampling time is correspondingly increased.

        Channel 0 is the lowest frequency, Channel numChan - 1 is the highest.
        TBD!
        """
        return 0

    def TimeBin(self,factor):
        """
        Rebins the time series by a factor.
        """
        
        # Make sure you can divide the timeseries properly
        
        if self.shape[0] % factor > 0:
            if VERBOSE:
                print "Cannot cleanly divide the TimeSeries. Did nothing."
        else:
            newArray = np.empty((int(self.shape[0]/factor),self.shape[1]),dtype=self.timeSeries.dtype)
            for i in range(newArray.shape[0]):
                newArray[i,:] = np.mean(self.timeSeries[i*factor:(i+1)*factor,:],axis=0)
            self.timeSeries = newArray
            self.sampTime = self.sampTime*factor
            self.shape = self.timeSeries.shape
    
    def PlotTimeSeries(self,offSet=0):
        """
        Plots the absolute values of the time series
        offSet is a scalar offSet of each channel plot. 
        Increasing channels get shifted higher and higher 
        """
        plt.figure()
        timePoints = np.arange(0,self.timeSeries.shape[0])*self.sampTime
        #timePoints = (np.meshgrid(np.arange(0,self.shape[1]),
        #                          np.arange(0,self.shape[0]))[0])*self.sampTime

        offSets = np.outer(offSet*np.ones((self.shape[0],1)),np.arange(self.shape[1]))

        plt.plot(timePoints,np.abs(self.timeSeries)+offSets)
        plt.xlabel('Time (sec)')
        plt.ylabel('Power')
        plt.show()


def FastRiseExpDecay(timePoints,ampFRB,decayFRB,riseFRB):
    """ 
    Given an array of time points, this will generate an array of complex amplitudes corresponding to the shape of the FRB. The peak is at position time = 0. Negative times are okay. 
    The amplitude function is defined as 
    
    ampFRB*np.exp(timePoints/riseFRB), for time < 0
    ampFRB*np.exp(-timePoints/decayFRB), for time >=0
    
    The amplitude is used to generate gaussian distributed complex noise.
    """
    x = np.zeros(timePoints.shape)
    negBlock = np.where(timePoints<0)
    posBlock = np.where(timePoints>=0)

    x[negBlock] = np.exp(timePoints[negBlock]/riseFRB)
    x[posBlock] = np.exp(-timePoints[posBlock]/decayFRB)

    #divide by sqrt(2) to keep the amplitude 
    return ampFRB*x/np.sqrt(2)*(np.random.standard_normal(x.shape)+1j*np.random.standard_normal(x.shape))

 
class IntensityTimeSeries:
    """ 
    Defines a class of intensity time series.  
    The number of channels can be single or multiple (for example filterbank data)
    Frequencies are always handled in MHz. Time is always handled in seconds
    
    Attributes:
    timeSeries -- complex array of amplitude timeseries
    numChannels -- scalar integer, minimum 1
    fMin, fMax -- minimum and maximum frequency. The channels are assumed to be equispaced for now.
    """
    def __init__(self,timeSeries=None,
                 lenSeries=2**18,
                 numChannels=1,
                 fMin=400,fMax=800,
                 sampTime=None,
                 noiseRMS=0.1):

        return None
