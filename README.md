#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 22:39:11 2025

@author: shahid
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

# =====================
#  Parameters (WITH UNITS)
# =====================
# Supernova feedback
eta_sn = 1e-5  # dimensionless
sfr = 3.0 * u.M_sun/u.yr  # Star formation rate

# AGN feedback
epsilon_rad = 0.1  # dimensionless
mbh_dot = 1e-4 * u.M_sun/u.yr  # Black hole accretion rate

# Time parameters
t_max = 10 * u.Gyr  # Total simulation time
n_steps = 1000

# =====================
#  Unit Conversions
# =====================
# Convert all to CGS units (g, s, erg)
c = const.c.to(u.cm/u.s)  # cm/s
c_squared = (c**2).to(u.erg/u.g)  # erg/g

# Convert mass rates to g/s
sfr_g = sfr.to(u.g/u.s)
mbh_dot_g = mbh_dot.to(u.g/u.s)

# Create time array in seconds
time = np.linspace(0, t_max.to(u.s).value, n_steps) * u.s

# =====================
#  Feedback Calculation (FIXED)
# =====================
def calculate_feedback(time, sfr, mbh_dot):
    """Calculate feedback energy with proper unit handling"""
    # Initialize energy arrays with units
    energy_sn = np.zeros_like(time.value) * u.erg
    energy_agn = np.zeros_like(time.value) * u.erg
    total_energy = np.zeros_like(time.value) * u.erg
    
    for i in range(1, len(time)):
        dt = (time[i] - time[i-1]).to(u.s)  # Time step with units
        
        # Supernova energy: η * (SFR in g/s) * c² * dt
        energy_sn[i] = energy_sn[i-1] + eta_sn * sfr * c_squared * dt
        
        # AGN energy: ε * (accretion rate in g/s) * c² * dt
        energy_agn[i] = energy_agn[i-1] + epsilon_rad * mbh_dot * c_squared * dt
        
        # Total energy
        total_energy[i] = energy_sn[i] + energy_agn[i]
    
    return energy_sn, energy_agn, total_energy

# Run calculation
energy_sn, energy_agn, total_energy = calculate_feedback(
    time, 
    sfr_g, 
    mbh_dot_g
)

# =====================
#  Plot Results
# =====================
plt.figure(figsize=(12, 6))

# Cumulative energy plot
plt.subplot(1, 2, 1)
plt.plot(time.to(u.Gyr), total_energy.to(u.erg), 
         label='Total Feedback Energy', lw=2)
plt.plot(time.to(u.Gyr), energy_sn.to(u.erg), 
         '--', label='Supernova Contribution')
plt.plot(time.to(u.Gyr), energy_agn.to(u.erg), 
         '--', label='AGN Contribution')

plt.xlabel('Time (Gyr)', fontsize=12)
plt.ylabel('Cumulative Energy (erg)', fontsize=12)
plt.title('Galactic Feedback Energy Evolution', fontsize=14)
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Energy rate plot
plt.subplot(1, 2, 2)
sn_rate = (eta_sn * sfr_g * c_squared).to(u.erg/u.s)
agn_rate = (epsilon_rad * mbh_dot_g * c_squared).to(u.erg/u.s)

plt.bar(['Supernovae', 'AGN'], [sn_rate.value, agn_rate.value], 
        color=['tab:blue', 'tab:orange'])
plt.ylabel('Energy Injection Rate (erg/s)', fontsize=12)
plt.title('Feedback Energy Rates', fontsize=14)
plt.yscale('log')

plt.tight_layout()
plt.show()

# Units Check Verification 
print(sfr_g.unit)         # g / s
print(mbh_dot_g.unit)     # g / s
print(c_squared.unit)     # erg / g
print(time.unit)          # s
print(energy_sn.unit)     # erg
plt.savefig('Galactic_Feedback_Energy_Evolution.png')  # Save the plot as PNG
