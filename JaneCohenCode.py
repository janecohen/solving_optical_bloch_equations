#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:37:43 2024

@author: janecohen
"""

#%% imports 

import matplotlib.pyplot as plt  
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter 
import math as m  
import numpy as np 
import timeit 

#%% Optical Bloch Equations (OBEs)

"OBEs - CW excitation, RWA"
# on-resonance excitation: delta0L = 0
def OBE_CW(y,t):
    dy=np.zeros((len(y))) 
    dy[0] = 0.
    dy[1] = Omega0/2*(2.*y[2]-1.)
    dy[2] = -Omega0*y[1]
    return dy

"OBEs - RWA with time-dependent pulse"
def OBE_RWA(y,t): 
    dy=np.zeros((len(y))) 
    dy[0] = 0. #real part of du
    dy[1] = pulse_gauss(t)/2*(2.*y[2]-1.) #imaginary part of du
    dy[2] = -pulse_gauss(t)*y[1] # n_e
    return dy

"OBEs - Full wave"
def OBE_full(y,t): 
    dy=np.zeros((len(y))) 
    dy[0] = -gammad*y[0] + w0*y[1] #real part of du
    dy[1] = -gammad*y[1] - w0*y[0] + pulse_full(t)*(2.*y[2]-1.) #imaginary part
    dy[2] = -2.*pulse_full(t)*y[1] # n_e
    return dy

#%% Gaussian funtions

"Gaussian pulse"
def pulse_gauss(t):
    Omega = Omega0*np.exp(-((t-toff)**2))
    return Omega

"Full Gaussian pulse"
def pulse_full(t):
    Omega = Omega0*np.exp(-((t-toff)**2))*np.sin(wL*(t-toff)+phi)
    return Omega


#%% Solvers

"Simple Euler ODE Solver"
def EulerForward(f,y,t,h):
    k1 = h*f(y,t)                    
    y=y+k1
    return y 

"Runge-Kutta ODE solver"
def RungeKutta(f,y,t,h):
    k1 = h*f(y,t)
    k2 = h*f(y+k1/2, t+h/2)
    k3 = h*f(y+k2/2, t+h/2)
    k4 = h*f(y+k3, t+h)
    y = y + (1/6)*(k1+2*k2+2*k3+k4)
    return y     
    
#%% Q1 (b) - Compare the exact, forward Euler and Runge-Kutta solutions

"Parameters"
Omega0 = 2*np.pi 
dt = 0.001
tmax =5.

# numpy arrays for time and y ODE sets
tlist = np.arange(0.0, tmax, dt) # list of time intervals
npts = len(tlist) # number of points
yEuler = np.zeros((npts,3)) # 2D array for Euler method results
y1Euler = yEuler[1,:] # initial row to pass to Euler solver
yRK = np.zeros((npts,3)) # 2D array for RK method results
y1RK = yRK[1,:] # initial row to pass to Euler solver


"Call ODE Solver for Euler method"
for i in range(1,npts):   # loop over time
    y1Euler = EulerForward(OBE_CW,y1Euler,tlist[i-1],dt) 
    yEuler[i,:]= y1Euler


"Call ODE Solver for Runge-Kutta method"
for i in range(1,npts):   # loop over time
    y1RK = RungeKutta(OBE_CW,y1RK,tlist[i-1],dt) 
    yRK[i,:]= y1RK
    

"Exact Solution for excited state population"
yexact = [m.sin(Omega0*tlist[i]/2)**2 for i in range(npts)]
  

#%% Q1 (b) Graphics

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.dpi']= 120

#CW excitation using the RWA
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(tlist, yexact, 'b', label = 'Exact solution', linewidth = 2.5)
ax.plot(tlist, yEuler[:,2], 'g', label = 'Forward Euler', linewidth = 2.5)
ax.plot(tlist, yRK[:,2], 'r', label = "Runge-Kutta", linewidth = 2.5, linestyle = 'dashed')
ax.set(xlabel='$t/t_p$', ylabel='$n_e$', title="$\Delta$t = 0.001")
ax.grid()
plt.legend(loc='center left')
plt.show() 
#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q1b.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 


#%% Q1 (c) - Reduce the step size (h=0.01)
dt = 0.01

# numpy arrays for time and y ODE sets
tlist = np.arange(0.0, tmax, dt) # list of time intervals
npts = len(tlist) # number of points
yEuler = np.zeros((npts,3)) # 2D array for Euler method results
y1Euler = yEuler[1,:] # initial row to pass to Euler solver
yRK = np.zeros((npts,3)) # 2D array for RK method results
y1RK = yRK[1,:] # initial row to pass to Euler solver


"Call ODE Solver for Euler method"
for i in range(1,npts):   # loop over time
    y1Euler = EulerForward(OBE_CW,y1Euler,tlist[i-1],dt) 
    yEuler[i,:]= y1Euler


"Call ODE Solver for Runge-Kutta method"
for i in range(1,npts):   # loop over time
    y1RK = RungeKutta(OBE_CW,y1RK,tlist[i-1],dt) 
    yRK[i,:]= y1RK
    

"Exact Solution for excited state population"
yexact = [m.sin(Omega0*tlist[i]/2)**2 for i in range(npts)]

#%% Q1 (c) Graphics

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(tlist, yexact, 'b', label = 'Exact solution', linewidth = 2.5)
ax.plot(tlist, yEuler[:,2], 'g', label = 'Forward Euler', linewidth = 2.5)
ax.plot(tlist, yRK[:,2], 'r', label = "Runge-Kutta", linewidth = 2.5, linestyle = 'dashed')
ax.set(xlabel='$t/t_p$', ylabel='$n_e$', title="$\Delta$t = 0.01")
ax.grid()
plt.legend(loc='lower left')
plt.show() 
#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q1c.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 

#%% Q2 - Rabi flop

"Parameters"
#on-resonance excitation and no dephasing
delta0L = 0
gammad = 0

toff = 5 # time offset
pulseArea = 2.*np.pi
Omega0=pulseArea/(np.sqrt(np.pi)) 
dt = 0.01
tmax =10.

# numpy arrays for time and y ODE set
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist) # number of points
yRabi = np.zeros((npts,3)) # 2D array for results
y1Rabi = yRabi[1,:] # initial row to pass to solver
pulse = np.zeros(npts)

"Call ODE solver"
for i in range(1,npts):   # loop over time
    y1Rabi = RungeKutta(OBE_RWA,y1Rabi,tlist[i-1],dt) 
    yRabi[i,:]= y1Rabi
    pulse[i] = pulse_gauss(tlist[i-1])

#%% Q2 Graphics

plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.dpi']= 120

#RWA Rabi Flop
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(tlist, yRabi[:,2], 'm', linewidth=2.5, label='$n_e$')
ax.plot(tlist, yRabi[:,1], 'b', linewidth=2.4, label='Im[$u$]')
ax.plot(tlist, pulse, 'g', linewidth=2.4, label='$\widetilde{\Omega}(t)$')
ax.set(xlabel='$t/t_p$', ylabel='$n_e$')
ax.grid()
plt.legend(loc='best')
plt.show()
#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q2.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 

#%% Q3 (a) - Full-wave (no RWA)

"Parameters"
pulseArea = 2.*np.pi # nominal 2π pulse
Omega0 = pulseArea/(np.sqrt(np.pi)) 
phi = 0
gammad = 0
toff= 5

dt = 0.01
tmax = 10.

# numpy arrays for time
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist)

"Call ODE solver for RWA equations"
# RWA (independent of wL)
yRWA = np.zeros((npts,3))
y1RWA = yRWA[1,:]

for i in range(1,npts):   # loop over time
    y1RWA = RungeKutta(OBE_RWA,y1RWA,tlist[i-1],dt) 
    yRWA[i,:]= y1RWA

"Call ODE solver for full OBEs"
sols = []
wLList = [Omega0, 2.*Omega0, 8.*Omega0]

# solve OBEs for three different values of wL (w0=wL)
for i in range(3):
    w0, wL = wLList[i], wLList[i]
    yFull = np.zeros((npts,3))
    y1Full = yFull[1,:]

    for i in range(1,npts):   # loop over time
        y1Full = RungeKutta(OBE_full,y1Full,tlist[i-1],dt) 
        yFull[i,:]= y1Full
    
    sols.append(yFull[:])
    
    
#%% Q3 (a) Graphics - seperate plot for each value of wL

multList = [1, 2, 8] # Omega0 multipliers

for i in range(3):
    yFull = sols[i]
    
    # create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    mult = multList[i]
    # plot data on the first subplot
    ax1.plot(tlist, yFull[:,1], 'b', linewidth=2.5, label='Full wave')
    #ax1.plot(tlist, yFull[:,0], 'orange', linewidth=2.5, label='$Re[u]$')
    ax1.plot(tlist, yRWA[:,1], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
    #ax1.plot(tlist, yRWA[:,0], 'm', linewidth=2.5, label='$RWA Re[u]$', linestyle = 'dashed')
    ax1.legend(loc='right')
    ax1.set(xlabel='$t/t_p$', ylabel='Im[$u(t/t_p)]$', title = f'$\omega_L = {mult}\Omega_0$')
    ax1.grid('True')


    # plot data on the second subplot
    ax2.plot(tlist, yFull[:,2], 'm', linewidth=2.5, label='Full wave')
    ax2.plot(tlist, yRWA[:,2], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
    ax2.set(xlabel='$t/t_p$', ylabel='$n_e$')
    ax2.legend(loc='right')
    ax2.grid('True')

    # adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()
    
    
#%% Q3 (a) u subplots

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13))

# Omega0
ax1.plot(tlist, sols[0][:,1], 'b', linewidth=2.5, label='Full wave')
ax1.plot(tlist, yRWA[:,1], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
ax1.legend(loc='upper right')
ax1.set(xlabel='$t/t_p$', ylabel='Im[u]', title = f'$\omega_L = \Omega_0$')
ax1.grid('True')

# 2 Omega0
ax2.plot(tlist, sols[1][:,1], 'b', linewidth=2.5, label='Full wave')
ax2.plot(tlist, yRWA[:,1], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
ax2.legend(loc='upper right')
ax2.set(xlabel='$t/t_p$', ylabel='Im[u]', title = f'$\omega_L = 2\Omega_0$')
ax2.grid('True')

# 8 Omega0
ax3.plot(tlist, sols[2][:,1], 'b', linewidth=2.5, label='Full wave')
ax3.plot(tlist, yRWA[:,1], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
ax3.legend(loc='upper right')
ax3.set(xlabel='$t/t_p$', ylabel='Im[u]', title = f'$\omega_L = 8\Omega_0$')
ax3.grid('True')


# adjust layout to prevent overlap
plt.tight_layout()

plt.show()
#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q3a1.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 

#%% Q3 (a) ne subplots

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13))

# Omega0
ax1.plot(tlist, sols[0][:,2], 'm', linewidth=2.5, label='Full wave')
ax1.plot(tlist, yRWA[:,2], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
ax1.set(xlabel='$t/t_p$', ylabel='$n_e$', title = f'$\omega_L = \Omega_0$')
ax1.legend(loc='right')
ax1.grid('True')

# 2 Omega0
ax2.plot(tlist, sols[1][:,2], 'm', linewidth=2.5, label='Full wave')
ax2.plot(tlist, yRWA[:,2], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
ax2.set(xlabel='$t/t_p$', ylabel = '$n_e$', title = f'$\omega_L = 2\Omega_0$')
ax2.legend(loc='right')
ax2.grid('True')

# 8 Omega0
ax3.plot(tlist, sols[2][:,2], 'm', linewidth=2.5, label='Full wave')
ax3.plot(tlist, yRWA[:,2], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
ax3.set(xlabel='$t/t_p$', ylabel = '$n_e$', title = f'$\omega_L = 8\Omega_0$')
ax3.legend(loc='right')
ax3.grid('True')

# adjust layout to prevent overlap
plt.tight_layout()

plt.show()
#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q3a2.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 


#%% Q3 (a) - changing the phase to φ = π/2

"Parameters"
pulseArea = 2.*np.pi # nominal 2π pulse
Omega0 = pulseArea/(np.sqrt(np.pi)) 
phi = np.pi/2.
gammad = 0
toff= 5

dt = 0.01
tmax = 10.

# numpy arrays for time
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist)

"Call ODE solver for RWA equations"
# RWA (independent of wL)
yRWA = np.zeros((npts,3))
y1RWA = yRWA[1,:]

for i in range(1,npts):   # loop over time
    y1RWA = RungeKutta(OBE_RWA,y1RWA,tlist[i-1],dt) 
    yRWA[i,:]= y1RWA

"Call ODE solver for full OB equations"
sols = []
wLList = [Omega0, 2.*Omega0, 8.*Omega0]

# solve OBEs for three different values of wL (w0=wL)
for i in range(3):
    w0, wL = wLList[i], wLList[i]
    yFull = np.zeros((npts,3))
    y1Full = yFull[1,:]

    for i in range(1,npts):   # loop over time
        y1Full = RungeKutta(OBE_full,y1Full,tlist[i-1],dt) 
        yFull[i,:]= y1Full
    
    sols.append(yFull[:])
    
"Plot results"    
multList = [1, 2, 8]
for i in range(3):
    yFull = sols[i]
    
    # create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    mult = multList[i]
    # plot data on the first subplot
    ax1.plot(tlist, yFull[:,1], 'b', linewidth=2.5, label='Full wave')
    #ax1.plot(tlist, yFull[:,0], 'orange', linewidth=2.5, label='$Re[u]$')
    ax1.plot(tlist, yRWA[:,1], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
    #ax1.plot(tlist, yRWA[:,0], 'm', linewidth=2.5, label='$RWA Re[u]$', linestyle = 'dashed')
    ax1.legend(loc='right')
    ax1.set(xlabel='$t/t_p$', ylabel='Im[$u(t/t_p)]$', title = f'$\omega_L = {mult}\Omega_0$')
    ax1.grid('True')


    # plot data on the second subplot
    ax2.plot(tlist, yFull[:,2], 'm', linewidth=2.5, label='Full wave')
    ax2.plot(tlist, yRWA[:,2], 'g', linewidth=2.5, label='RWA', linestyle = 'dashed')
    ax2.set(xlabel='$t/t_p$', ylabel='$n_e$')
    ax2.legend(loc='right')
    ax2.grid('True')

    # adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()

#%% Q3 (b) - fixed wL

"Paramaters"
pulseArea = 2.*np.pi
Omega0 = 2.*np.pi/(np.sqrt(np.pi))
w0, wL = 2.*Omega0, 2.*Omega0 # fix wL = 2*Omega0 for pulse area of 2pi
phi = 0
gammad = 0
toff= 5

# u quickly varying- adjust time step
dt = 0.001
tmax = 10.
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist) # number of points

"Solve full OBEs"
# solve OBEs for three different pulse areas
pulseAreaList = [np.pi/2., 4.*np.pi, 16.*np.pi] # pulse areas
sols = []

for i in range(3):
    Omega0 = pulseAreaList[i]/(np.sqrt(np.pi)) 

    y = np.zeros((npts,3))
    y1 = y[1,:]

    for i in range(1,npts):   # loop over time
        y1 = RungeKutta(OBE_full,y1,tlist[i-1],dt) 
        y[i,:]= y1
    
    sols.append(y)

print(sols)


#%% Q3 (b) Graphics

for i in range(3):
    yFull = sols[i]
    
    # create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    # plot data on the first subplot
    ax1.plot(tlist, yFull[:,1], 'b', linewidth=2.5, label='$Im[u]$')
    ax1.plot(tlist, yFull[:,0], 'orange', linewidth=2.5, label='$Re[u]$')
    ax1.legend(loc='upper left')
    ax1.set(xlabel='$t/t_p$', ylabel='$u(t)$')
    ax1.grid('True')


    # plot data on the second subplot
    ax2.plot(tlist, yFull[:,2], 'm', linewidth=2.5, label='$n_e$')
    ax2.set(xlabel='$t/t_p$', ylabel='$n_e$')
    ax2.legend(loc='upper left')
    ax2.grid('True')

    # adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()
    

#%% Q3 (b) ne plots

fig, (ax1) = plt.subplots(1, 1, figsize=(9, 7))
ax1.plot(tlist, sols[2][:,2], 'y', linewidth=2.5, label='$\pi/2$')
ax1.plot(tlist, sols[1][:,2], 'm', linewidth=2.5, label='$4\pi$')
ax1.plot(tlist, sols[0][:,2], 'b', linewidth=2.5, label='$16\pi$')
ax1.set(xlabel='$t/t_p$', ylabel='$n_e$')
ax1.legend(loc='upper left')
ax1.grid('True')


# adjust layout to prevent overlap
plt.tight_layout()

plt.show()

#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q3b.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 
    
#%% Q3 (c)

"Parameters"
pulseArea = 2.*np.pi
Omega0 = 2.*np.pi/(np.sqrt(np.pi))
wL, w0 = 2.*Omega0, 2.*Omega0 # fix wL = 2*Omega0 for pulse area of 2pi
phi = 0.
tp = 1.
gammad = 0.2 / tp
toff= 5

# u quickly varying- adjust time step
dt = 0.001
tmax = 50.
toff = 5.
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist) # number of points

"Prepare arrays"
pulseAreaList = [np.pi/2., 4.*np.pi, 16.*np.pi] # pulse areas
uwList, OmegawList, wList = [], [], [] # arrays for Fourier transforms

for i in range(3):
    
    pulseArea = pulseAreaList[i]
    Omega0 = pulseArea/(np.sqrt(np.pi))
    
    y = np.zeros((npts,3))
    y1 = y[1,:]
    
    "Call ODE solver for full OBEs"
    for i in range(1,npts):   # loop over time
        y1 = RungeKutta(OBE_full,y1,tlist[i-1],dt) 
        y[i,:]= y1
    
    # make list of complex u (using y[0] and y[1])
    ut = y[:,0] + 1j*y[:,1]
    
    "Fourier transforms"
    uw = np.fft.fft(ut) # fft of u(t)
    uw = np.fft.fftshift(uw) # shift the zero-frequency component to the center of the spectrum
    uw = np.abs(uw)/np.max(np.abs(uw[int(len(uw)/2):])) # normalize
    uwList.append(uw)
    
    # get pulse values in list
    Omegat = pulse_full(tlist) # calculate Omega(t)
    Omegaw = np.fft.fft(Omegat) # fft of Omega(t)
    Omegaw = np.fft.fftshift(Omegaw) # shift frequency axis from 0-fs to -fs/2 - fs/2
    Omegaw = np.abs(Omegaw)/np.max(np.abs(Omegaw)) # normalize 
    OmegawList.append(Omegaw)
    
        
    # use fftfreq to get frequency intervals
    # np.fft.fftfreq(n,d) where n = window length, d = sample spacing
    w = np.fft.fftfreq(npts, dt) / (wL) # fft of dt
    w = np.fft.fftshift(w)*2.*np.pi # shift the zero-frequency component to the center of the spectrum
    wList.append(w)


#%% Q3 (c) graphics

for i in range(3):
    uw = uwList[i]
    Omegaw = OmegawList[i]
    w = wList[i]
    
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(w, uw, 'b', linewidth=2.5, label='$|u(w)|$')
    ax.plot(w, Omegaw, 'orange', linewidth=2.5, label='$\Omega(w)$')
    ax.set(xlabel='$w/w_L$', ylabel='$|u(w)|$', title='Polarization power spectrum')
    ax.grid()
    ax.set_xlim(0,6)
    plt.legend()
    ax.set_ylim(0,1.1)
    
    plt.show() 

#%% Q3 (c) graphics - combined

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(wList[0], uwList[0], 'b', linewidth=2.5, label='$\pi/2$')
ax.plot(wList[1], uwList[1], 'orange', linewidth=2.5, label='$4\pi$')
ax.plot(wList[2], uwList[2], 'g', linewidth=2.5, label='$16\pi$')
ax.plot(w, Omegaw, 'm', linewidth=2.5, label='$\Omega(w)$', linestyle='dashed')
ax.set(xlabel='$w/w_L$', ylabel='$|u(w)|$')
ax.grid()
ax.set_xlim(0,6)
plt.legend()
ax.set_ylim(0,1.1)

plt.show() 

#plt.savefig('./Desktop/ENPH479/Assignment1plots/Q3c.pdf', format='pdf', dpi=1200,bbox_inches = 'tight') 

#%% Q4 

"Parameters"
pulseArea = 2.*np.pi
Omega0 = 2.*np.pi/(np.sqrt(np.pi))
w0, wL = 2.*Omega0, 2.*Omega0 # fix wL = 2*Omega0 for pulse area of 2pi
phi = 0
gammad = 0
toff= 5

# u quickly varying- adjust time step
dt = 0.01
tmax = 10.
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist) # number of points

"Use 3(b), pulse area = 4pi, as an example"
pulseArea = 4.*np.pi
Omega0 = pulseArea/(np.sqrt(np.pi)) 
y = np.zeros((npts,3))
y1 = y[1,:]

"Call original ODE solver"
start = timeit.default_timer()
for i in range(1,npts):   # loop over time
    y1 = RungeKutta(OBE_full,y1,tlist[i-1],dt) 
    y[i,:]= y1

stop = timeit.default_timer()
time = stop - start

# read in ode solver
from scipy.integrate import odeint
y0=[0. , 0. , 0.] # initial condition

"Call SciPy ODE solver"
start = timeit.default_timer()
vals = odeint(OBE_full, y0, tlist)
pop = vals[:,2]
stop = timeit.default_timer()
time2 = stop - start

print("Original code: ", time, "s")
print("Using SciPy:)", time2, "s")

"Plot results"
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 7))
ax1.plot(tlist, y[:,0], label = "Original", linewidth=2.5)
ax1.plot(tlist, vals[:,0], label="SciPy", linestyle='dashed', linewidth=2.5)
ax1.set(xlabel='$t/t_p$', ylabel='$Re[u]$')
ax1.legend()
plt.show()


#%% The end







