# Christopher A. Hales
# 22 June 2017
# Version 1.0
#
#
# This code is released under a BSD 3-Clause License
# See LICENSE for details
#
# This code was used to obtain the results shown in arXiv:1706.06612
# and EVLA Memo 201 / ALMA Memo 603.
#
#
# This code will plot spurious full-array on-axis linear polarization
# (95th percentile) when viewing an unpolarized source following calibration
# with an array with circular feeds. Solutions will be plotted for a
# polarization calibator observed over a range of input S/N and parallactic
# angle coverage values.
#
# Note that the full-array spurious polarization is calculated by dividing
# the error in the modulus of instrumental polarization leakage by
# sqrt(2.N_ant/pi) where N_ant is the number of antennas in the array.
# ie if you want to recover d-term modulus error, multiply by this factor.
#
# S/N = source total intensity divided by noise, where noise is
#       given by the sensitivity within an individual channel
#       dual-pol image in which all baselines were combined (e.g. a
#       Stokes I image) for a time period spanning 1 slice (1+ scans
#       combined at approximately constant parallactic angle).
#       The code will assume that S/N is the same for all slices.
#       e.g. see https://science.nrao.edu/facilities/vla/docs/
#                        manuals/oss/performance/sensitivity
#            noise = SEFD/corr_eff/sqrt[2.Nant.(Nant-1).timespan.bandwidth]
#
# The plots produced by this code will enable the following questions
# to be answered:
#     "For a given S/N, what is the parallactic angle coverage I require
#      to obtain spurious on-axis polarization less than X percent?"
#      
# or alternatively
#     "For a given parallactic angle coverage, what is the S/N I require
#      to obtain spurious on-axis polarization less than X percent?"
#
# The code focuses on the leakage terms accessible through the
# crosshands, ie relative leakages.  Calculation of errors for
# absolute leakages (eg relevant for total intensity dynamic
# range calculations) require additional modifications to this
# code and are left for future investigation.
#
#
#
# The code investigates characteristic error in the leakage modulus
# "D" for a single polarization (R or L) on a single antenna by
# considering leakage in V_RL (or could use V_LR) along baseline i-j.
# The code accounts for Nant-1 baselines to antenna i when solving
# for the characteristic error.
#
# The code assumes that leakages are constant with time (mechanical
# details probably don't change rapidly), and that the leakage
# contribution to the cross-hands is fixed.  The latter is a good
# approximation because higher order combinations of leakage and
# polarization are negligible for most telescopes due to leakages
# not being close to unity.
# 
#
#
# The code will calculate results for 2 representative scenarios where
# the calibrator has 3% or 10% fractional linear polarization.
#
# The code will produce plots showing the typical error in the modulus
# of D, divided by sqrt(2.N_ant/pi) to obtain spurious full-array on-axis
# polarization, as a functions of S/N parallactic angle separation between
# slices. Two cases will be considered for both fractional polarization
# scenarios: Q & U known a priori, and unknown.
#
# When Q & U are known, 2 independent slices are required to recover D.
#    ie 2 points needed to identify origin of circle when radius is
#    known. The origin degeneracy is broken by the known direction of
#    rotation with parallactic angle.
#
# When Q & U are unknown, 3 independent slices are required to recover D
#    as well as Q & U (ie 3 points needed to recover radius and center).
#    This code will assume that the 3 points are spaced equally within
#    the specified total parallactic angle span.  This code will also
#    investiage a 10 slice approach.
# 
# This code will not consider the errors in recovered Q & U, as these
# are not of primary interest for characterizing the instrument.
#
#
#

import numpy as np
from scipy import optimize
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#from matplotlib.backends.backend_pdf import PdfPages

################################################################################

# number of antennas in array
Nant = 27

# fractional linear polarization of calibrator = L/I = sqrt(Q^2+U^2)/I
# no need to specify position angle for this exercise, let's assume it is
# 45 degrees (i.e. all crosshand signal in U at parallactic angle 0 degrees)
# specify two representative cases
fraclinpol = np.r_[0.03,0.1]

################################################################################

pi  = np.pi
d2r = pi/180.

Nb   = Nant*(Nant-1)/2

parsep = np.linspace(1,180,90)
snr    = np.logspace(1,5,90)
x,y    = np.meshgrid(snr,parsep)

# number of samples from which to measure error in d
# note that uncertainty in estimate of rms from N samples is
# np.sqrt((1+0.75/(N-1))**2 * (1-1./N) - 1.)
# p. 63, Johnson N. L., Kotz S., 1970, Distributions in Statistics:
# Continuous Univariate Distributions 1. Houghton Mifflin, NY
# ie we require 1e3 samples to get error in rms to 2%
# Non-linearity of problem makes this an approximation,
# but should be reasonable enough
samples = 1e3

# fast track
#parsep = np.linspace(1,180,10)
#snr    = np.logspace(1,5,10)
#x,y    = np.meshgrid(snr,parsep)
#samples = 1e2

mypercentile = 95

samples = np.int(samples)

# number of scans equally separated over total parallactic angle coverage,
# when calibrator Stokes vector unknown a priori
nscans = np.r_[3,10]

def calc_R(xc, yc):
    return np.sqrt((V_RL.real-xc)**2 + (V_RL.imag-yc)**2)

def func_3pt(c):
    Ri = calc_R(*c)
    return Ri - Ri.mean()

onxpol  = np.zeros([3,fraclinpol.size,parsep.size,snr.size])
myerr   = np.zeros([2,fraclinpol.size,parsep.size,snr.size,2])

# == PLOT 1 ==
# polarization known and 2 parallactic angle slices
p=0
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    for i in range(parsep.size):
        print 'case: '+str(p+1)+'/3'+\
              '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)+\
              '   parallactic angle: '+str(i+1)+'/'+str(parsep.size)
        for k in range(snr.size):
            temparr = np.zeros(samples)
            for s in range(samples):
                # If sigma defined as dual pol (eg Stokes I) noise, then noise in
                # real or imag part of single pol product (e.g. V_RL) is sqrt(2)*sigma,
                # with additional factor sqrt(Nb) to convert to single-baseline noise
                # Note that we also need to divide by sqrt(Nant-1) because Nant-1
                # baselines can be used to solve for D_Ri on antenna i.
                # Assume source PA=45 (Q=0, U=L), without loss of generality
                # Assume D_true=0+0i, without loss of generality
                #   (we are only interested in error on recovery of D, don't care
                #    about systematic offset of center due to actual D terms)
                # Assume I = 1 Jy, without loss of generality
                
                V_RL = np.zeros(2,dtype=complex)
                # technically not V_RL, actually average over Nant-1 baselines, meh
                V_RL[0] = L*1j + \
                         (np.random.randn()+np.random.randn()*1j) / \
                          snr[k]*np.sqrt(2)/np.sqrt(Nant-1)*np.sqrt(Nb)
                V_RL[1] = L*1j*np.exp(-2j*parsep[i]*d2r) + \
                         (np.random.randn()+np.random.randn()*1j) / \
                          snr[k]*np.sqrt(2)/np.sqrt(Nant-1)*np.sqrt(Nb)
                
                # Real solvers properly account for rotation direction.
                # Here, to avoid degeneracy in 2 point fit, calculate result
                # analytically and select solution closest to origin.
                # No significant impact from this simplification.
                sep = np.sqrt((V_RL[1].real-V_RL[0].real)**2+\
                      (V_RL[1].imag-V_RL[0].imag)**2)
                if sep <= 2.*L:
                    bisec = np.sqrt(L**2-(sep/2.)**2)/sep
                    xc_1  = np.mean(V_RL.real) + (V_RL[0].imag-V_RL[1].imag)*bisec
                    xc_2  = np.mean(V_RL.real) - (V_RL[0].imag-V_RL[1].imag)*bisec
                    yc_1  = np.mean(V_RL.imag) + (V_RL[1].real-V_RL[0].real)*bisec
                    yc_2  = np.mean(V_RL.imag) - (V_RL[1].real-V_RL[0].real)*bisec
                    if xc_1**2+yc_1**2 < xc_2**2+yc_2**2:
                        xc = xc_1
                        yc = yc_1
                    else:
                        xc = xc_2
                        yc = yc_2
                else:
                    # at low S/N, points separated by greater than known diameter
                    # assign center at midpoint
                    # this is consistent with treatment above
                    xc = np.mean(V_RL.real)
                    yc = np.mean(V_RL.imag)
                
                # (xc,yc) represents [D_Ri + sum^(Na-1) (D_Lj)* / (Na-1)]
                # characteristic offset deltaD represents both D_Ri and D_Lj
                # ie model using random direction imposed on characteristic deltaD
                # this factor becomes increasingly negligible for larger arrays.
                # Assume it is negligible here.
                # Note that the variance in the sum needs to be divided by 2
                # because it imposes a random direction on characteristic deltaD
                # typical error in mod(D), where units of D are percent
                # factor 1/sqrt(2) projects the 2D error onto 1D D-term
                # then divided by sqrt(2.*Nant/pi) to get spurious on-axis pol
                temparr[s] = np.sqrt(xc**2+yc**2) / np.sqrt(2) / \
                             np.sqrt(2.*Nant/pi) * 100.
            
            # temparr is Rayleigh distributed
            onxpol[p,m,i,k] = np.percentile(temparr,mypercentile)

# == PLOT 2 ==
# polarization unknown and 3 parang slices separated equally
p=1
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    for i in range(parsep.size):
        print 'case: '+str(p+1)+'/3'+\
              '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)+\
              '   parallactic angle: '+str(i+1)+'/'+str(parsep.size)
	for k in range(snr.size):
            # Error 1: Instead of points being fit on a circle centered near zero,
            #          the low S/N and small parang coverage will cause the
            #          optimization to fit a circle with really huge radius and
            #          center located at some crazy distance away.  In real
            #          production code this ambiguity won't occur because
            #          the sense of rotation will be taken into account.
            #          As our units are in fractional polarization, we can
            #          perform a simplistic check for this error by getting
            #          rid of solutions that are offset from (0,0) by more than
            #          unity.  The error will become more prevalent at low S/N
            #          and small parang range.  When these are low enough,
            #          eventually all solutions will fail, but this should
            #          happen in a region of parameter space away from where
            #          we are really interested in seeing.  We can plot the
            #          the fraction of err1/samples solutions to see where
            #          things start to go funny (note the text above about
            #          setting the value of "samples" and resulting error).
            err1=0
            # Error 2: optimize.leastsq couldn't find a valid solution
            #          These should be minimal and thus negligible
            err2=0
            temparr = np.zeros(samples)
            for s in range(samples):
                V_RL = np.zeros(nscans[p-1],dtype=complex)
                for n in range(nscans[p-1]):
                    V_RL[n] = L*1j*np.exp(-2j*parsep[i]*d2r * \
                              np.float(n)/np.float(nscans[p-1]-1)) + \
                             (np.random.randn()+np.random.randn()*1j) / \
                              snr[k]*np.sqrt(2)/np.sqrt(Nant-1)*np.sqrt(Nb)
                
                # set starting estimate at (0,0)
                # could also set at np.mean(V_RL.real), np.mean(V_RL.imag)
                # but not necessary here (we aren't imposing a systematic D)
                # http://www.scipy.org/Cookbook/Least_Squares_Circle
                (xc,yc),ier = optimize.leastsq(func_3pt, (0,0))
                if (ier>=1) and (ier<=4):
                    # valid solution found
                    # in my testing, the failure rate of this function is minimal
                    # with typically no failures, every now and then one,
                    # and just once I saw 10.  These are all <<samples
                    # so the statistics won't be affected.  All good.
                    # This doesn't need to be production code :)
                    rad2 = xc**2+yc**2
                    if rad2 < 1.:
                        temparr[s] = np.sqrt(rad2) / np.sqrt(2) /\
                                     np.sqrt(2.*Nant/pi) * 100.
                    else:
                        err1 += 1
                else:
                    err2 += 1
            
            onxpol[p,m,i,k] = np.percentile(temparr,mypercentile)
            myerr[p-1,m,i,k,0]  = np.float(err1)/np.float(samples)
            myerr[p-1,m,i,k,1]  = np.float(err2)/np.float(samples)
            if err1 == samples:
                # insert dummy value so it doesn't show up as zero
                onxpol[p,m,i,k] = 100.

# == PLOT 3 ==
# polarization unknown and 10 parang slices separated equally
p=2
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    for i in range(parsep.size):
        print 'case: '+str(p+1)+'/3'+\
              '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)+\
              '   parallactic angle: '+str(i+1)+'/'+str(parsep.size)
	for k in range(snr.size):
            err1=0
            err2=0
            temparr = np.zeros(samples)
            for s in range(samples):
                V_RL = np.zeros(nscans[p-1],dtype=complex)
                for n in range(nscans[p-1]):
                    V_RL[n] = L*1j*np.exp(-2j*parsep[i]*d2r * \
                              np.float(n)/np.float(nscans[p-1]-1)) + \
                             (np.random.randn()+np.random.randn()*1j) / \
                              snr[k]*np.sqrt(2)/np.sqrt(Nant-1)*np.sqrt(Nb)
                
                (xc,yc),ier = optimize.leastsq(func_3pt, (0,0))
                if (ier>=1) and (ier<=4):
                    rad2 = xc**2+yc**2
                    if rad2 < 1.:
                        temparr[s] = np.sqrt(rad2) / np.sqrt(2) /\
                                     np.sqrt(2.*Nant/pi) * 100.
                    else:
                        err1 += 1
                else:
                    err2 += 1
            
            onxpol[p,m,i,k] = np.percentile(temparr,mypercentile)
            myerr[p-1,m,i,k,0]  = np.float(err1)/np.float(samples)
            myerr[p-1,m,i,k,1]  = np.float(err2)/np.float(samples)
            if err1 == samples:
                onxpol[p,m,i,k] = 100.

np.save('results_circ_dat',onxpol)
np.save('results_circ_err',myerr)

for p in range(3):
    for m in range(fraclinpol.size):
        L=fraclinpol[m]
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        if p == 2:
            mylevels = np.r_[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]
        else:
            mylevels = np.r_[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]
        
        cs = ax.contour(x,y,onxpol[p][m],levels=mylevels,colors='k')
        #cs = ax.contour(x,y,onxpol[p][m],levels=mylevels,cmap=cm.jet,norm=LogNorm())
        plt.clabel(cs,inline=1,fontsize=10)
        ax.set_xscale('log')
        if p == 2:
            ax.set_title(str(mypercentile)+'$^{th}$ percentile '+\
                          'spurious on-axis fractional linear pol. [%] \nfor '+\
                          '10 slice strategy with calibrator L/I~'+\
                          '{:.0f}'.format(L*100.)+'%',fontsize=18,x=0.45,y=1.04)
        else:
            ax.set_title(str(mypercentile)+'$^{th}$ percentile '+\
                          'spurious on-axis fractional linear pol. [%] \nfor '+\
                           str(p+2)+' slice strategy with calibrator L/I~'+\
                          '{:.0f}'.format(L*100.)+'%',fontsize=18,x=0.45,y=1.04)
        
        ax.set_xlabel('S/N [Stokes I, Nant='+str(Nant)+', 1 channel, 1 slice]',\
                       labelpad=9,fontsize=17)
        ax.set_ylabel('Total parallactic angle coverage [deg]',\
                      labelpad=11,fontsize=17)
        plt.tight_layout()
        #plt.show()
        if p == 2:
            plt.savefig('results_circ_S10_L'+'{:.0f}'.format(L*100.)+'.png')
        else:
            plt.savefig('results_circ_S'+str(p+2)+'_L'+'{:.0f}'.format(L*100.)+'.png')
        
        #pp = PdfPages('results_circ_S'+str(p+2)+'_L'+'{:.0f}'.format(L*100.)+'.pdf')
        ## if you see a unicode warning from savefig, ignore it
        ## unsure why it appears, perhaps due to use of colors='k' rather than cmap
        #pp.savefig()
        #pp.close()
