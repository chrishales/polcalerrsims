# Christopher A. Hales
# 30 May 2017
# Version 1.0
#
#
# This code is released under a BSD 3-Clause License
# See LICENSE for details
#
# This code was used to obtain the results shown in <FIXME add arxiv link>
# and EVLA Memo 201 / ALMA Memo 603.
#
#
# This code will plot spurious full-array on-axis linear polarization
# (95th percentile) when viewing an unpolarized source following
# calibration with an array with linear feeds. Solutions will be plotted
# for a polarization calibator observed over a range of input S/N and
# parallactic angle coverage values.
#
# Note that the full-array spurious linear polarization is calculated by
# dividing the error in the modulus of instrumental polarization leakage
# by sqrt(N_ant) where N_ant is the number of antennas in the array.
# ie if you want to recover d-term modulus error, multiply by this factor.
#
# Note also that the statistics for spurious linear and spurious circular
# polarization are the same, though in the latter there may be an
# additional zero-point issue to contend with (see also comments below).
# For spurious elliptical polarization, multiply the values in the plot
# by sqrt(pi/2).
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
#
#
# The code focuses on the characteristic error in the leakage modulus
# "D" for a single polarization (X or Y) on a single antenna by
# considering leakage in V_XY (or could use V_YX) along baseline i-j.
# The code utilizes Nant-1 baselines to antenna i when solving for
# the characteristic error.
#
# The code assumes that leakages are constant with time (mechanical
# details probably don't change rapidly).
#
#
#
# It is difficult to generalize the solutions to all calibrators.
# This code will assume that the first observed slice is at zero
# parallactic angle (probably likely for the 1 and 2 scan approaches)
# and that the calibrator has position angle 45deg.  Worst-case
# errors will probably result by placing points symmetrically
# about U_psi=0, but users would run into trouble trying to
# solve for crosshand phase, so let's also avoid this setup here.
# Note that the true worst-case solutions of relevance for this
# simulation will take place when 2 scans have similar U_psi
# (i.e. negligible parallactic angle separation), whether in
# the 2 or 3 scan strategies.
#
# Of course, observers should do whatever they can to maximize
# U_psi coverage, rather that simply parallactic angle coverage!!
# This is even more important in the 2 scan strategy than in the
# 3 scan strategy, which is of course handy because we know the
# source polarization a priori in the 2 scan approach and can
# therefore target specific U_psi values.
#
#
#
# The code will calculate results for 2 representative scenarios where
# the calibrator has 3% or 10% fractional linear polarization.
#
# The code will produce plots showing the typical error in the modulus
# of D, divided by sqrt(N_ant) to obtain spurious full-array on-axis
# linear polarization, as a functions of S/N parallactic angle
# separation between slices. Position angle errors will also be
# estimated, including both statistical and systematic errors,
# the latter when relevant (arising from relative leakage solutions).
#
# This simulation will assume that explicit position angle calibration
# has NOT been performed.  A systematic error will thus be included.
#
# Two cases will be considered for both fractional polarization
# scenarios: Q & U known a priori, and unknown.
#
# When Q & U are known, 1 or 2 independent slices are required to
#    recover D, relative or absolute, respectively.  For either
#    strategy, 1 slice is needed to determine the crosshand phase.
#
# When Q & U are unknown, 3 independent slices are required to recover
#    the crosshand phase as well as Q & U. 2+ of these slices are then
#    needed to recover absolute leakages.  This code will assume that
#    the 3 points are spaced equally within the specified total
#    parallactic angle span. This code will also investiage a 10 slice
#    approach.
# 
# This code will only take into account errors in recovered Q & U as
# appropriate for the 3 and 10 slice strategies.
#
# This code will assume that Stokes V is zero for all calibrators.
#
#
#

import numpy as np
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

################################################################################

# number of antennas in array
Nant = 40

# fractional linear polarization of calibrator = L/I = sqrt(Q^2+U^2)/I
# assume position angle is 45 degrees, i.e. all crosshand signal in U
# at parallactic angle 0 degrees
# specify two representative cases
fraclinpol = np.r_[0.03,0.1]

# typical (mean) d-term modulus (~0.015 for ALMA in bands 3,6,7)
# this is = sqrt(pi/2) times 1 sigma real or imaginary part
d_typical = 0.015
# above value is also used to estimate worst-case systematic position angle error
# for 1 slice strategy due to use of relative leakages, assuming
# Re(d_typical) ~= d_typical
# (statistically, this will probably be more like d_typical/sqrt(2/pi), but
#  take worst-case estimate here)

# mechanical feed alignment uncertainty for individual antenna, in degrees
systpaerr_foffset = 2.0
# This will be added in quadrature to the statistical errors calculated,
# as well as the relative-leakage systematic error above for the 1 slice strategy

# recover numerical or theoretical solutions for 1 slice strategy?
# results are practically indistinguisable
# 0=theory 1=numerical
strategy1slice = 0

################################################################################

pi  = np.pi
d2r = pi/180.

Nb = Nant*(Nant-1)/2

systpaerr_foffset_rad = systpaerr_foffset * d2r

# number of samples from which to measure error in d
# note that uncertainty in estimate of rms from N samples is
# np.sqrt((1+0.75/(N-1))**2 * (1-1./N) - 1.)
# p. 63, Johnson N. L., Kotz S., 1970, Distributions in Statistics:
# Continuous Univariate Distributions 1. Houghton Mifflin, NY
# ie we require 1e3 samples to get error in rms to 2%
# Non-linearity of problem makes this an approximation,
# but should be reasonable enough
samples = 1e4

parsep = np.linspace(0.1,179.8,90)
snr    = np.logspace(1,6,90)
x,y    = np.meshgrid(snr,parsep)

# fast track
#parsep = np.linspace(0.1,179.8,10)
#snr    = np.logspace(1,6,10)
#x,y    = np.meshgrid(snr,parsep)
#samples = 1e2

mypercentile = 95

samples = np.int(samples)

# number of scans equally separated over total parallactic angle coverage,
# when calibrator Stokes vector unknown a priori
nscans = np.r_[3,10]

def calc_rho_12(dat):
    # recover crosshand phase for 1 or 2 slice strategy
    # Q&U will be known
    # assume true crosshand phase is zero, no loss in generality
    return np.arctan2(dat.imag,dat.real)

def calc_d(u_psi,q_psi,V_XY,dxi):
    # least squares solution, real and imag components separately
    # close enough
    # X --> dxi
    # Y --> sum over dyj
    # A-G --> real constants when fitting for X and Y
    A = np.sum( (1.-q_psi)*u_psi )
    B = np.sum( (1.-q_psi)**2 )
    C = np.sum( 1.-q_psi**2 )
    Dx = np.sum( np.real(V_XY)*(1.-q_psi) )
    Dy = np.sum( np.imag(V_XY)*(1.-q_psi) )
    E  = np.sum( (1.+q_psi)*u_psi )
    F  = np.sum( (1.+q_psi)**2 )
    Gx = np.sum( np.real(V_XY)*(1.+q_psi) )
    Gy = np.sum( np.imag(V_XY)*(1.+q_psi) )
    Xx = (C*Gx - Dx*F + A*F - C*E)/(C**2 - B*F)
    Xy = (C*Gy - Dy*F            )/(C**2 - B*F)
    X = Xx + Xy*1j
    # Y not needed
    #Yx = (Gx - C*Xx - E)/F
    #Yy = (Gy - C*Xy    )/F
    #Y = Yx + Yy*1j
    return np.abs(X-dxi)

def calc_rho(u_psi):
    xm  = np.average(u_psi.real)
    ym  = np.average(u_psi.imag)
    num = np.sum((u_psi.real-xm)*(u_psi.imag-ym))
    den = np.sum((u_psi.real-xm)**2)
    return np.arctan2(num,den)

def func_qu(psi,q,u,err):
    return u*np.cos(2.*psi) - q*np.sin(2.*psi) + err

def calc_qu(rho,psi,u_psi):
    # solve along real axis
    # ideally we would perform the fit with points weighted
    # by their distance from the line fit by rho (e.g.
    # away from the mean imaginary offset), but it
    # shouldn't make a huge difference in practice.
    # The variances are all equal anyway (by design), so
    # perhaps it isn't the right thing to do.  Also, meh
    u_psi_rot_real = np.real(np.exp(-1j*rho) * u_psi)
    popt, pcov = curve_fit(func_qu,psi,u_psi_rot_real)
    return popt[0:2]

# strategy (1/2/3 slice) -- fracpol -- leakage/posang error -- parsep -- snr
#          p                   m               b                 i        k
onxpol = np.zeros([4,fraclinpol.size,2,parsep.size,snr.size])

# == PLOT 1 ==
# polarization known and 1 parallactic angle slice
# crudely assume that errors arise from crosshand phase error combined
# with statistical errors for relative d-term recovery
# it's probably a bit more complicated in reality, but
# this should recover main trends with pretty good accuracy
# which is all we are after
p=0
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    print 'case: '+str(p+1)+'/4'+\
          '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)
    for k in range(snr.size):
        temparr = np.zeros(samples)
        if strategy1slice:
            for s in range(samples):
                # If sigma defined as dual pol (eg Stokes I) noise, then noise in
                # real or imag part of single pol product (e.g. V_XY) is sqrt(2)*sigma,
                # with additional factor sqrt(Nb) to convert to single-baseline noise.
                # Note that we also need to divide by sqrt(Nant-1) because Nant-1
                # baselines can be used to solve for D_Xi on antenna i.
                # Assume source PA=45 (Q=0, U=L), without loss of generality
                # Q_psi = Qcos2p+Usin2p = Lsin2p
                # U_psi = Ucos2p-Qsin2p = Lcos2p
                # Assume I = 1 Jy, without loss of generality
                u_psi = np.r_[L]
                # therefore q_psi = np.r_[0.] , but not needed here
                
                # get crosshand phase (1 per array defined on refant)
                # assume solution has been iterated and that the
                # final d-term errors are small enough to ne negligible
                # in the baseline-averaged crosshand phase recovery
                # Thus only statistical errors matter here
                # For each slice, all baselines, dual-pol, real or imag component:
                # noise is 1/snr
                noise = np.r_[(np.random.randn()+np.random.randn()*1j)/snr[k]]
                # rho_err is the error in frame rotation
                rho_err = calc_rho_12(u_psi + noise)
                
                # d-term errors will result because the d-terms will
                # soak up the rho_err induced offset from the true
                # calibrator polarization and leakage offsets.
                # This can be crudely modelled by taking into
                # account the typical offset induced by rho_err in V_XY.
                # Assume small angle error; errors will get crazy
                # at large angles regardless, so no significant effect
                offset = rho_err * (u_psi + \
                         np.random.normal(0,d_typical*np.sqrt(2./pi)) + \
                         np.random.normal(0,d_typical*np.sqrt(2./pi)) )
                
                # now proceed as if we are calculating d-term modulus errors
                # for an unpolarized source, ie model the true
                # source polarization by the term above
                temparr[s] = np.sqrt((offset**2+np.float(Nant)/snr[k]**2)/2.)
                # offset is practically negligible in the equation above
                # for most observing conditions because rho_err gets
                # progressively smaller with increasing S/N, always
                # remaining smaller than the statistical error term
        
        # typical error in mod(D), where units of D are percent
        # then divided by sqrt(Nant) to get spurious on-axis LINEAR pol
        # this is same result as for spurious CIRCULAR pol
        # if we were interested in spurious ELLIPTICAL pol then
        # we would instead need to divide by sqrt(2.*Nant/pi)
        if strategy1slice:
            # result from this simulation
            onxpol[p,m,0,0,k] = np.percentile(temparr,mypercentile)/np.sqrt(Nant)*100.
        else:
            # practically indistinguishable result from theory
            onxpol[p,m,0,0,k] = 1./np.sqrt(2.*snr[k]**2)*100.
        
        # absolute position angle error from quadrature sum of:
        #   * statistical error from real part of d-term error (which will
        #     be the same as the value in temparr as this has already been
        #     deprojected with factor 1/sqrt2, yeah lazy)
        #   * systematic error from relative leakage solutions (magnitude
        #     of typical d-term real part)
        #   * systematic error from mechanical feed offset uncertainty
        if strategy1slice:
            # result from this simulation
            onxpol[p,m,1,0,k] = np.sqrt(np.percentile(temparr,mypercentile)**2 + d_typical**2 + \
                                        systpaerr_foffset_rad**2/np.float(Nant)) / d2r
        else:
            # practically indistinguishable result from theory
            onxpol[p,m,1,0,k] = np.sqrt(np.float(Nant)/snr[k]**2/2. + d_typical**2 + \
                                        systpaerr_foffset_rad**2/np.float(Nant)) / d2r

# == PLOT 2 ==
# polarization known and 2 parallactic angle slices
p=1
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    for i in range(parsep.size):
        print 'case: '+str(p+1)+'/4'+\
              '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)+\
              '   parallactic angle: '+str(i+1)+'/'+str(parsep.size)
        for k in range(snr.size):
            temparr = np.zeros(samples)
            for s in range(samples):
                # same initial conditions as for plot 1, no loss in generality
                u_psi = np.r_[L ,L*np.cos(2.*parsep[i]*d2r)]
                q_psi = np.r_[0.,L*np.sin(2.*parsep[i]*d2r)]
                
                # get crosshand phase, again same as for plot 1
                # use slice at max u_psi, this would be used in practice
                noise = np.r_[(np.random.randn()+np.random.randn()*1j)/snr[k]]
                rho_err = calc_rho_12(u_psi[0] + noise)
                
                # now model typical D-term error in the crosshand phase frame,
                # but where the raw data is rotated additionally by rho_err
                # V_XY_avg = u_psi + (1.-q_psi)*dxi + (1.+q_psi)*sum_[j]^[Nant-1](dyj*)
                # where noise is given by the average over Nant-1 baselines
                # We need to inject true dxi and dyj then try to recover dxi
                dxi = np.random.normal(0,d_typical*np.sqrt(2./pi)) + \
                      np.random.normal(0,d_typical*np.sqrt(2./pi))*1j
                dyj = np.average( np.random.normal(0,d_typical*np.sqrt(2./pi),Nant-1) + \
                              np.random.normal(0,d_typical*np.sqrt(2./pi),Nant-1)*1j )
                V_XY = np.zeros(2,dtype=complex)
                # Technically not V_XY, actually V_XY_corrupted averaged over
                # Nant-1 baselines ... meh ... though if you are reading
                # this and really do care about something as finicky as
                # a variable name, along with the details of this code,
                # then let's sit down for a coffee! We can celebrate
                # that 2 people in the world have looked at this code!
                V_XY = ( u_psi + (1.-q_psi)*dxi + (1.+q_psi)*dyj ) * \
                         np.exp(rho_err*1j) + \
                        (np.random.randn(2)+np.random.randn(2)*1j) / \
                         snr[k]*np.sqrt(2)*np.sqrt(Nb)/np.sqrt(Nant-1)
                
                # get d-term modulus error
                # divide by sqrt(2) because this error contributes a random
                # direction on the true d-term modulus
                temparr[s] = calc_d(u_psi,q_psi,V_XY,dxi)/np.sqrt(2)
            
            # spurious lin pol
            onxpol[p,m,0,i,k] = np.percentile(temparr,mypercentile)/np.sqrt(Nant)*100.
            
            # absolute position angle error
            # strategy recovers absolute leakages
            onxpol[p,m,1,i,k] = np.sqrt(np.percentile(temparr,mypercentile)**2 + \
                                        systpaerr_foffset_rad**2/np.float(Nant)) / d2r

# == PLOT 3 ==
# polarization unknown and 3 parallactic angle slices
p=2
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    for i in range(parsep.size):
        print 'case: '+str(p+1)+'/4'+\
              '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)+\
              '   parallactic angle: '+str(i+1)+'/'+str(parsep.size)
        for k in range(snr.size):
            temparr = np.zeros(samples)
            for s in range(samples):
                # same initial conditions as for plot 1, no loss in generality
                psi       = np.zeros(nscans[p-2])
                u_psi_obs = np.zeros(nscans[p-2],dtype=complex)
                for n in range(nscans[p-2]):
                    psi[n]       = parsep[i]*d2r * np.float(n)/np.float(nscans[p-2]-1)
                    u_psi_obs[n] = L*np.cos(2.*psi[n]) + \
                                   (np.random.randn()+np.random.randn()*1j)/snr[k]
                
                # get crosshand phase and Stokes Q & U
                # assume solution has been iterated and that the
                # final d-term errors are small-ish in the
                # baseline-averaged crosshand phase recovery.
                # Thus only statistical errors matter here
                # Don't assume that the curve goes through the origin;
                # this will then be the same behaviour as the real solver.
                # While I*D and Q*D may be largely removed through
                # iteration, some small residual may exist, which the
                # solver fits out. Note that if iteration is not used,
                # the real CASA solver will handle the I*D offset,
                # but it will not properly account for the Q*D terms,
                # in turn slightly corrupting the fits.
                rho_err = calc_rho(u_psi_obs)
                q , u   = calc_qu(rho_err,psi,u_psi_obs)
		u_psi   = u*np.cos(2.*psi) - q*np.sin(2.*psi)
		q_psi   = q*np.cos(2.*psi) + u*np.sin(2.*psi)
		
		# now continue as with plot 2
                dxi = np.random.normal(0,d_typical*np.sqrt(2./pi)) + \
                      np.random.normal(0,d_typical*np.sqrt(2./pi))*1j
                dyj = np.average( np.random.normal(0,d_typical*np.sqrt(2./pi),Nant-1) + \
                              np.random.normal(0,d_typical*np.sqrt(2./pi),Nant-1)*1j )
                V_XY = np.zeros(nscans[p-2],dtype=complex)
                V_XY = ( u_psi + (1.-q_psi)*dxi + (1.+q_psi)*dyj ) * \
                         np.exp(rho_err*1j) + \
                        (np.random.randn(nscans[p-2])+np.random.randn(nscans[p-2])*1j) / \
                         snr[k]*np.sqrt(2)*np.sqrt(Nb)/np.sqrt(Nant-1)
                
                temparr[s] = calc_d(u_psi,q_psi,V_XY,dxi)/np.sqrt(2)
            
            onxpol[p,m,0,i,k] = np.percentile(temparr,mypercentile)/np.sqrt(Nant)*100.
            onxpol[p,m,1,i,k] = np.sqrt(np.percentile(temparr,mypercentile)**2 + \
                                        systpaerr_foffset_rad**2/np.float(Nant)) / d2r

# == PLOT 4 ==
# polarization unknown and 10 parallactic angle slices
p=3
for m in range(fraclinpol.size):
    L=fraclinpol[m]
    for i in range(parsep.size):
        print 'case: '+str(p+1)+'/4'+\
              '   fracpol: '+str(m+1)+'/'+str(fraclinpol.size)+\
              '   parallactic angle: '+str(i+1)+'/'+str(parsep.size)
        for k in range(snr.size):
            temparr = np.zeros(samples)
            for s in range(samples):
                psi       = np.zeros(nscans[p-2])
                u_psi_obs = np.zeros(nscans[p-2],dtype=complex)
                for n in range(nscans[p-2]):
                    psi[n]       = parsep[i]*d2r * np.float(n)/np.float(nscans[p-2]-1)
                    u_psi_obs[n] = L*np.cos(2.*psi[n]) + \
                                   (np.random.randn()+np.random.randn()*1j)/snr[k]
                
                rho_err = calc_rho(u_psi_obs)
                q , u   = calc_qu(rho_err,psi,u_psi_obs)
		u_psi   = u*np.cos(2.*psi) - q*np.sin(2.*psi)
		q_psi   = q*np.cos(2.*psi) + u*np.sin(2.*psi)
		
		# now continue as with plot 2
                dxi = np.random.normal(0,d_typical*np.sqrt(2./pi)) + \
                      np.random.normal(0,d_typical*np.sqrt(2./pi))*1j
                dyj = np.average( np.random.normal(0,d_typical*np.sqrt(2./pi),Nant-1) + \
                              np.random.normal(0,d_typical*np.sqrt(2./pi),Nant-1)*1j )
                V_XY = np.zeros(nscans[p-2],dtype=complex)
                V_XY = ( u_psi + (1.-q_psi)*dxi + (1.+q_psi)*dyj ) * \
                         np.exp(rho_err*1j) + \
                        (np.random.randn(nscans[p-2])+np.random.randn(nscans[p-2])*1j) / \
                         snr[k]*np.sqrt(2)*np.sqrt(Nb)/np.sqrt(Nant-1)
                
                temparr[s] = calc_d(u_psi,q_psi,V_XY,dxi)/np.sqrt(2)
            
            onxpol[p,m,0,i,k] = np.percentile(temparr,mypercentile)/np.sqrt(Nant)*100.
            onxpol[p,m,1,i,k] = np.sqrt(np.percentile(temparr,mypercentile)**2 + \
                                        systpaerr_foffset_rad**2/np.float(Nant)) / d2r

np.save('results_lin_dat',onxpol)

for p in range(4):
    if p == 0:
        fig, ax1 = plt.subplots()
        if strategy1slice:
            ax1.plot(snr,onxpol[p][0][0][0],'b-',label='calibrator L/I={:.0f}'.format(fraclinpol[0]*100.)+'%')
            ax1.plot(snr,onxpol[p][1][0][0],'r-',label='calibrator L/I={:.0f}'.format(fraclinpol[1]*100.)+'%')
        else:
            ax1.plot(snr,onxpol[p][0][0][0],'k-')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        plt.grid()
        ax2 = ax1.twinx()
        if strategy1slice:
            ax2.plot(snr,onxpol[p][0][1][0],'b--')
            ax2.plot(snr,onxpol[p][1][1][0],'r--')
            legend = ax1.legend(loc='upper right')
            ax1.set_title('1 slice strategy using calibrator with known Stokes vector\n'+\
                          'Numerical results, dashed curves = right axis',fontsize=16,y=1.04)
            ax1.set_ylabel(str(mypercentile)+'$^{th}$ percentile spurious on-axis\n'+\
                          'fractional linear polarization [%]',\
                           labelpad=10,fontsize=16)
        else:
            ax2.plot(snr,onxpol[p][0][1][0],'k--')
            ax1.set_title('1 slice strategy using calibrator with known Stokes vector\n'+\
                          'Theoretical results, dashed curve = right axis',fontsize=16,y=1.04)
            ax1.set_ylabel('Spurious on-axis fractional linear polarization [%]\n',\
                           fontsize=16,labelpad=-9)
        
        ax1.set_xlabel('S/N [Stokes I, Nant='+str(Nant)+', 1 channel, 1 slice]',\
                       labelpad=9,fontsize=16)
        ax2.set_ylabel('Absolute position angle error [deg]\n',\
                       labelpad=10,fontsize=16)
        ax1.set_xlim([1e1,1e5])
        ax2.set_xlim([1e1,1e5])
        ax1.set_ylim([1e-4,1e1])
        ax2.set_ylim([0,10])
        plt.tight_layout()
        #plt.show()
        plt.savefig('results_lin_S'+str(p+1)+\
                    '_D'+'{:.0f}'.format(d_typical*100.)+\
                    '_F'+'{:.0f}'.format(systpaerr_foffset)+'.png')
    elif p==3:
        for m in range(fraclinpol.size):
            L=fraclinpol[m]
            fig=plt.figure()
            ax = fig.add_subplot(1,1,1)
            mylevels = np.r_[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]
            cs = ax.contour(x,y,onxpol[p][m][0],levels=mylevels,colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][0],colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][0],levels=mylevels,cmap=cm.jet,norm=LogNorm())
            plt.clabel(cs,inline=1,fontsize=10)
            ax.set_xscale('log')
            ax.set_title(str(mypercentile)+'$^{th}$ percentile '+\
                          'spurious on-axis fractional linear pol. [%] \nfor '+\
                          '10 slice strategy with calibrator L/I~'+\
                          '{:.0f}'.format(L*100.)+'%',fontsize=18,x=0.45,y=1.04)
            ax.set_xlabel('S/N [Stokes I, Nant='+str(Nant)+', 1 channel, 1 slice]',\
                           labelpad=9,fontsize=17)
            ax.set_ylabel('Total parallactic angle coverage [deg]',\
                          labelpad=11,fontsize=17)
            ax.set_xlim([1e2,1e6])
            ax.set_ylim([0,180])
            plt.tight_layout()
            #plt.show()
            plt.savefig('results_lin_S10'+\
                        '_L'+'{:.0f}'.format(L*100.)+\
                        '_D'+'{:.0f}'.format(d_typical*100.)+'.png')
            
	    fig=plt.figure()
            ax = fig.add_subplot(1,1,1)
            mylevels = np.r_[0.35,0.5,1,2,5,10,20,50]
            cs = ax.contour(x,y,onxpol[p][m][1],levels=mylevels,colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][1],colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][1],levels=mylevels,cmap=cm.jet,norm=LogNorm())
            plt.clabel(cs,inline=1,fontsize=10)
            ax.set_xscale('log')
            ax.set_title(str(mypercentile)+'$^{th}$ percentile '+\
                          'absolute position angle error [deg] \nfor '+\
                          '10 slice strategy with calibrator L/I~'+\
                          '{:.0f}'.format(L*100.)+'%',fontsize=18,x=0.45,y=1.04)
            ax.set_xlabel('S/N [Stokes I, Nant='+str(Nant)+', 1 channel, 1 slice]',\
                           labelpad=9,fontsize=17)
            ax.set_ylabel('Total parallactic angle coverage [deg]',\
                          labelpad=11,fontsize=17)
            ax.set_xlim([1e2,1e6])
            ax.set_ylim([0,180])
            plt.tight_layout()
            #plt.show()
            plt.savefig('results_lin_S10'+\
                        '_L'+'{:.0f}'.format(L*100.)+\
                        '_D'+'{:.0f}'.format(d_typical*100.)+\
                        '_F'+'{:.0f}'.format(systpaerr_foffset)+'_pa.png')
    else:
        for m in range(fraclinpol.size):
            L=fraclinpol[m]
            fig=plt.figure()
            ax = fig.add_subplot(1,1,1)
            mylevels = np.r_[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]
            cs = ax.contour(x,y,onxpol[p][m][0],levels=mylevels,colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][0],colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][0],levels=mylevels,cmap=cm.jet,norm=LogNorm())
            plt.clabel(cs,inline=1,fontsize=10)
            ax.set_xscale('log')
            ax.set_title(str(mypercentile)+'$^{th}$ percentile '+\
                          'spurious on-axis fractional linear pol. [%] \nfor '+\
                           str(p+1)+' slice strategy with calibrator L/I~'+\
                          '{:.0f}'.format(L*100.)+'%',fontsize=18,x=0.45,y=1.04)
            ax.set_xlabel('S/N [Stokes I, Nant='+str(Nant)+', 1 channel, 1 slice]',\
                           labelpad=9,fontsize=17)
            ax.set_ylabel('Total parallactic angle coverage [deg]',\
                          labelpad=11,fontsize=17)
            ax.set_xlim([1e2,1e6])
            ax.set_ylim([0,180])
            plt.tight_layout()
            #plt.show()
            plt.savefig('results_lin_S'+str(p+1)+\
                        '_L'+'{:.0f}'.format(L*100.)+\
                        '_D'+'{:.0f}'.format(d_typical*100.)+'.png')
            
	    fig=plt.figure()
            ax = fig.add_subplot(1,1,1)
            mylevels = np.r_[0.35,0.5,1,2,5,10,20,50]
            cs = ax.contour(x,y,onxpol[p][m][1],levels=mylevels,colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][1],colors='k')
            #cs = ax.contour(x,y,onxpol[p][m][1],levels=mylevels,cmap=cm.jet,norm=LogNorm())
            plt.clabel(cs,inline=1,fontsize=10)
            ax.set_xscale('log')
            ax.set_title(str(mypercentile)+'$^{th}$ percentile '+\
                          'absolute position angle error [deg] \nfor '+\
                           str(p+1)+' slice strategy with calibrator L/I~'+\
                          '{:.0f}'.format(L*100.)+'%',fontsize=18,x=0.45,y=1.04)
            ax.set_xlabel('S/N [Stokes I, Nant='+str(Nant)+', 1 channel, 1 slice]',\
                           labelpad=9,fontsize=17)
            ax.set_ylabel('Total parallactic angle coverage [deg]',\
                          labelpad=11,fontsize=17)
            ax.set_xlim([1e2,1e6])
            ax.set_ylim([0,180])
            plt.tight_layout()
            #plt.show()
            plt.savefig('results_lin_S'+str(p+1)+\
                        '_L'+'{:.0f}'.format(L*100.)+\
                        '_D'+'{:.0f}'.format(d_typical*100.)+\
                        '_F'+'{:.0f}'.format(systpaerr_foffset)+'_pa.png')
