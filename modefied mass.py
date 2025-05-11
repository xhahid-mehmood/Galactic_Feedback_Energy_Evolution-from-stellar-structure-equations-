#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 22:49:56 2025

@author: shahid
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

# Set up dark matter and visible matter profiles
def hernquist_mass(r, M_v, a):
    """Visible matter (Hernquist profile)"""
    return M_v * (r**2) / (r + a)**2

def nfw_mass(r, rho_0, r_s):
    """NFW dark matter profile"""
    x = r / r_s
    return 4 * np.pi * rho_0 * r_s**3 * (np.log(1 + x) - x/(1 + x))

def total_mass(r, M_v, a, rho_0, r_s):
    """Total mass = visible + dark matter"""
    return hernquist_mass(r, M_v, a) + nfw_mass(r, rho_0, r_s)

# Galaxy parameters (Milky Way-like)
params = {
    'M_visible': 1e11 * u.M_sun,    # Visible mass
    'a': 2.0 * u.kpc,               # Visible matter scale radius
    'rho_0': 0.02 * u.M_sun/u.pc**3,# NFW characteristic density
    'r_s': 20.0 * u.kpc,            # NFW scale radius
    'R_vir': 200 * u.kpc             # Virial radius
}

# Convert units
params['rho_0'] = params['rho_0'].to(u.M_sun/u.kpc**3)

# Radial grid
r = np.logspace(-1, 3, 500) * u.kpc  # 0.1 pc to 1000 kpc

# Calculate mass components
M_visible = hernquist_mass(r, params['M_visible'], params['a'])
M_dark = nfw_mass(r, params['rho_0'], params['r_s'])
M_total = total_mass(r, params['M_visible'], params['a'], 
                    params['rho_0'], params['r_s'])

# Plotting
plt.figure(figsize=(12, 6))

# Mass profile plot
plt.subplot(1, 2, 1)
plt.loglog(r, M_visible, label='Visible Matter (Hernquist)')
plt.loglog(r, M_dark, '--', label='Dark Matter (NFW)')
plt.loglog(r, M_total, '-', label='Total Mass')

plt.axvline(params['r_s'].value, color='gray', linestyle=':', 
           label=r'NFW $r_s$ (20 kpc)')
plt.axvline(params['R_vir'].value, color='black', linestyle='--', 
           label=r'Virial Radius (200 kpc)')

plt.xlabel('Radius [kpc]', fontsize=12)
plt.ylabel(r'Enclosed Mass [M$_\odot$]', fontsize=12)
plt.title('Mass Distribution in Galaxy', fontsize=14)
plt.legend()
#plt.grid(True, which="both", ls="--")

# Rotation curve plot
plt.subplot(1, 2, 2)
v_visible = np.sqrt(const.G * M_visible / r).to(u.km/u.s)
v_dark = np.sqrt(const.G * M_dark / r).to(u.km/u.s)
v_total = np.sqrt(const.G * M_total / r).to(u.km/u.s)

plt.semilogx(r, v_visible, label='Visible Matter')
plt.semilogx(r, v_dark, '--', label='Dark Matter')
plt.semilogx(r, v_total, '-', label='Total')

plt.xlabel('Radius [kpc]', fontsize=12)
plt.ylabel('Circular Velocity [km/s]', fontsize=12)
plt.title('Galaxy Rotation Curve', fontsize=14)
plt.legend()
#plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()

# Print dark matter fraction
dm_fraction = (M_dark[-1]/(M_visible[-1] + M_dark[-1])).value*100
print(f"Dark matter fraction at virial radius: {dm_fraction:.1f}%")
plt.savefig('/media/shahid/OS/PhD Nust/Astronomy & Astrophysics/Project Presentation/Mass_Distribution_in_Galaxy.png')  # Save the plot as PNG
