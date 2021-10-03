from tkinter.constants import Y
import numpy as np
import math
import scipy.signal as signal
import matplotlib.pyplot as plt
import struct
import os

class Radon:
    def __init__(self):
        pass
    
    # Gain Adjust (GA)
    def PA1(self, xn, gain):
        gain_amp = 10**(gain/20)
        gain_adj = np.round(gain_amp*8192)
        print('gain factor register dlPa1GainAdj = {}'.format(int(gain_adj)))
        yn = xn * gain_adj / 2 **13
        yn = np.round(yn / 2)
        return yn

    #	Power Measurements (PWRM)
    def PWRM(self, xn, num_bits=16, samples=2457600):
        data = np.array(xn)
        data = data[0:samples]
        dbfs = 20*np.log10(self.rms(data)/2**(num_bits-1)-1)
        return dbfs

    def rms(self, data, axis=None):
        data = np.array(data)
        rms_value = np.sqrt(0 + np.mean(np.abs(data)**2, axis=axis))
        return rms_value

    def Filter_HB(self, fs=None, Ntaps=None, AttdB=None, Transition=None):
        try:
            numtaps, beta = signal.kaiserord(AttdB, Transition/(0.5*fs))
        except:
            print("Check AttdB and Transition!")
            pass

        if Ntaps==None:
            Ntaps = numtaps

        if fs==None:
            fs=1
        taps = signal.firwin(Ntaps, fs/2/2, window=('kaiser', beta), scale=False, nyq=fs/2)
        return taps

    def PLOT(self, signal, fs, Nfft=None, NbitInt=None, fnum=None, pltDisp=None):
        if NbitInt!=None:
            dbfs0 = 20*np.log10(2**NbitInt-1)
            dispYlab = 'dbfs'
        else:
            dbfs0 = 0
            dispYlab = 'dbbits'

        if fnum is None:
            pass
        if Nfft==None:
            Nfft = np.size(signal)
        if pltDisp==None:
            dispLegn = ''
            dispTile = ''
        elif type(pltDisp) is str:
            dispLegn = pltDisp+', '
            dispTile = ''
        elif type(pltDisp) is list and len(pltDisp)==1:
            dispLegn = pltDisp[0]
        elif type(pltDisp) is list and len(pltDisp)==2:
            dispLegn = pltDisp[0]+', '
            dispTile = pltDisp[1]

        signal_FFT = np.fft.fft(signal, Nfft)
        psd = np.fft.fftshift(np.abs(signal_FFT)/Nfft)**2
        psd_dB = 10*np.log10(np.abs(psd)) - dbfs0
        wPwr_dB = np.round(10*np.log10(np.sum(psd)) - dbfs0, 2)
        dispLegn = dispLegn+str(wPwr_dB)+dispYlab
        freqs = np.arange(-Nfft/2,Nfft/2)*int(fs/Nfft)
        plt.figure(fnum)
        plt.plot(freqs,psd_dB,label=dispLegn)
        plt.title(dispTile)
        plt.xlabel('freqs')
        plt.ylabel(dispYlab)
        plt.legend()
        plt.show()

    def get_capdata_IQ(self, filebin, Num, NbitsFloating=3, fnum=None):
        Data = np.fromfile(filebin,dtype='i4')/2**NbitsFloating
        I = Data[0::2]
        Q = Data[1::2]
        IQ = I+1j*Q
        return IQ

    def Ipo2(self, x, ratio=2, b=1):
        yUp = np.zeros(ratio*x.size, dtype=complex)
        yUp[::ratio] = x
        y = self.Filter(yUp, b)
        return y

    def Dec2(self, x, ratio=2, b=1):
        yFilter = self.Filter(x, b)
        y = yFilter[::ratio]
        return y
    
    def SRC(self, x, b=1, NUps=4, NDns=3):
        # Sample Rate Conversion
        NBufFilt = int(len(b)/NUps)
        bBuf = np.reshape(b, (NUps, NBufFilt))
        for i in range(NUps):
            tmp = self.Filter(x, bBuf[i])
            if i==0:
                y = np.zeros((NUps,len(tmp)), dtype=complex)
            y[i] = tmp
        y1 = y.flatten('F')
        y2 = self.Dec2(y1, NDns, b=1)
        return y2

    def Filter(self, x, b):
        y = np.convolve(x, b, mode='full')
        return y

    def Ipo2_SRC(self, x, fsIn, fsOut, coefFiltDB, flag_comp_gain=0, fnum=None):
        Nstage = 8
        NbitsFloat = [19,18,19,18,18,18,18,18]
        NbitsFrac = 3
        Ratio = np.ones(Nstage)
        
        for i in range(Nstage):
            if i!=1 and fsIn>fsOut/2:
                NUps = 1;
                NDns = 1
            elif i!=1 and fsIn<fsOut:
                NUps = 2;
                NDns = 1;
                b = self.DB_filtCoefHB(coefFiltDB[i], NbitsFloat[i])
                # self.PLOT(b*2048, fsIn*NUps, 2048, None, fnum)
                y = self.Ipo2(x, 2, b)*flag_comp_gain*NUps
                # self.PLOT(y, fsIn*NUps, len(y), None, fnum)
            elif i==1 and fsIn<fsOut: #SRC
                if np.mod((fsOut/(fsIn*2)),1)==0:
                    NUps = 2
                    NDns = 1
                elif np.mod((fsOut/(fsIn*4/3)),1)==0:
                    NUps = 4
                    NDns = 3
                elif np.mod((fsOut/(fsIn*8/5)),1)==0:
                    NUps = 8
                    NDns = 5
                elif np.mod((fsOut/(fsIn*16/15)),1)==0:
                    NUps = 16
                    NDns = 15
                b = np.array(coefFiltDB[i])/2**NbitsFloat[i]
                y = self.SRC(x, b, NUps, NDns)*flag_comp_gain*NUps
                # self.PLOT(y, fsIn*NUps/NDns, len(y), None, fnum)

            y = np.round(y, NbitsFrac)
            x = y
            Ratio[i] = NUps/NDns
            fsIn = fsIn*Ratio[i]
        return y

    def DB_filtCoefHB(self,coefFiltDB, NbitsFloat):
        b = coefFiltDB
        Nb = len(b)
        bHB = np.zeros(2*2*Nb-1)
        bHB[0:2*Nb:2] = b
        bHB[2*Nb-1] = 2**(NbitsFloat-1)
        bHB[2*Nb::2] = b[-1::-1]
        bHB = bHB/2**NbitsFloat
        return bHB


rd = Radon()
filebin=r'NR20MHz_23p04MHz_32768_MTAP5.bin'
filebin2=r'NR20MHz_245p76MHz_32768_MTAP6.bin'
fsMtap5 = 23.04e6
fsMtap6 = 245.76e6
NbitsFloating = 3

## Input: signal
sigMtap5 = rd.get_capdata_IQ(filebin, 32768, NbitsFloating)
rd.PLOT(sigMtap5, fsMtap5, 2048, 14, 901, 'SigMtap5')
sigMtap6 = rd.get_capdata_IQ(filebin2, 32768, NbitsFloating)
rd.PLOT(sigMtap6, fsMtap6, 2048, 14, 902, 'SigMtap6')
sigIn = sigMtap5

## Input: filter coefficient DB and fsOut
b0 = [-22,-1,21,-41,55,-59,47,-18,-23,71,-113,138,-133,94,-22,-71,167,-242,274,-246,153,-7,-167,333,-449,479,-402,217,47,-341,601,-761,769,-600,269,170,-630,1009,-1207,1157,-835,283,405,-1087,1607,-1827,1660,-1097,218,815,-1786,2470,-2677,2304,-1370,17,1496,-2850,3721,-3856,3140,-1633,-423,2631,-4517,5619,-5592,4300,-1866,-1316,4639,-7381,8848,-8520,6191,-2050,-3299,8893,-13545,16035,-15319,10738,-2166,-9916,24446,-39902,54509,-66499,74370,447210];
b1 = [4,-10,22,-44,80,-134,214,-330,492,-714,1006,-1390,1886,-2516,3312,-4316,5580,-7188,9262,-12022,15878,-21710,31824,-54684,166568];
b2 = [-9,92,-452,1485,-3834,9974,61806,-4363,998,-173,11,2,-19,183,-912,3136,-8918,30194,49388,-9958,3149,-854,162,-16,-16,162,-854,3149,-9958,49388,30194,-8918,3136,-912,183,-19,2,11,-173,998,-4363,61806,9974,-3834,1485,-452,92,-9];
b3 = [-1018,7912,-34060,158236];
b4 = [-358,3324,-16012,78582];
bDB = [b1, b2, b3, b4]

## Output: signal interpolation and SRC
sigOut = rd.Ipo2_SRC(sigIn, fsMtap5, fsMtap6, bDB, 1)

## Xcor
Nsamps6 = len(sigMtap6)
NsampsOut = len(sigOut)
sigMtap6b = np.zeros(NsampsOut)
sigMtap6b[:Nsamps6] = sigMtap6
Xcor = np.abs(np.fft.ifft(np.conj(np.fft.fft(sigOut))*(np.fft.fft(sigMtap6b))))
delay = np.argmax(Xcor)+1/3
# plt.figure(101)
# plt.plot(Xcor,label='xcor')
# plt.legend()

## Sync.
indexAng1 = np.linspace(0,0.5,int(NsampsOut/2))
indexAng2 = np.linspace(-0.5,0,int(NsampsOut/2))
indexAng = np.concatenate((indexAng1,indexAng2), axis=0)
# plt.figure(102)
# plt.plot(indexAng,label='indexAng')
sigOutCor = np.fft.ifft(np.fft.fft(sigOut)*np.exp(1j*2*np.pi*indexAng*-delay))
plt.figure(103)
plt.plot(sigMtap6[::],label='sigMtap6(captured)')
plt.plot(sigOutCor[0::],label='sigOut(sync.)')
plt.legend()

delayFlit = 500
rd.PLOT(sigMtap6[delayFlit:], fsMtap6, Nsamps6-delayFlit, 14, 903, 'sigMtap6')
rd.PLOT(sigOutCor[delayFlit:Nsamps6], fsMtap6, Nsamps6-delayFlit, 14, 903, 'sigOut(sync.)')

stop=1
