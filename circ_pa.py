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
# This code will plot crosshand phase uncertainty (1 sigma) vs
# linear polarization signal to noise, the latter defined by the sensitivity
# within a single-channel full-array image in Stokes Q or U.
#
#
#

import numpy as np
import matplotlib.pyplot as plt

pi  = np.pi
d2r = pi/180.

snr   = np.logspace(0,3,100)
PAerr = np.zeros(100)

# number of samples from which to measure error in d
# note that uncertainty in estimate of rms from N samples is
# np.sqrt((1+0.75/(N-1))**2 * (1-1./N) - 1.)
# p. 63, Johnson N. L., Kotz S., 1970, Distributions in Statistics:
# Continuous Univariate Distributions 1. Houghton Mifflin, NY
# ie we require 1e4 samples to get error in rms to 1%
samples = 1e4

samples = np.int(samples)

# assume all linearly polarized signal in Q for simplicity,
# won't affect generality of results
for k in range(snr.size):
    # noise in the real or imag part is given by noise
    # in full-array per-channel Stokes Q or U image
    qu       = 1. + (np.random.randn(samples)+np.random.randn(samples)*1j) / snr[k]
    PAerr[k] = np.std(np.arctan2(qu.imag,qu.real)/2./d2r)

plt.loglog(snr,PAerr,'k')
plt.xlabel('S/N=$(Q^2+U^2)^{0.5}/\sigma$ where $\sigma$ is noise in\nper-channel full-array Stokes Q or U image',
           labelpad=5,fontsize=16)
plt.ylabel('$1\sigma$ position angle error [deg]',fontsize=16)
#plt.title('Error in calibrated position angle',fontsize=18)
plt.grid()
plt.xlim([1e0,1e3])
plt.tight_layout()
plt.show()
#plt.savefig('circ_pa.png')
