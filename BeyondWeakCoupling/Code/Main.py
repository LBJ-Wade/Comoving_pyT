import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import time
import sys


from background_inputs import background_inputs
from model import model
from solver import solver


############################
############################
#Importing and interpolating background functions

n_back = 500 #Number of points for the background

N_load = np.linspace(-5, 10, n_back)
H_load = np.ones(n_back) #Hubble scale set to unity
rho_load = 1*np.ones(n_back)
m2_load = 1*np.ones(n_back)

mu_load = 1*np.ones(n_back)
kappa1_load = 1*np.ones(n_back)
kappa2_load = 1*np.ones(n_back)

background = background_inputs(N_load, H_load, rho_load, mu_load, m2_load, kappa1_load, kappa2_load)
interpolated = background.output()




############################
############################
#Compute the power spectra as function of time
############################
############################


# Nspan = np.linspace(-5, 10, 500)
# Nfield = 2
# Rtol, Atol = [1e-3, 1e-5, 1e-3], [1e-100, 1e-100, 1e-6]

# mdl = model(N = Nspan, Nfield = Nfield, interpolated = interpolated)


# N_exit = 0
# k = mdl.k_mode(N_exit)
# print("k = {}".format(k))

# DeltaN = 5 # number of efolds before horizon crossing
# Ni, Nf, Nsample = N_exit - DeltaN, 10, 1000 # sets initial and final efolds for transport equation integration
# N = np.linspace(Ni, Nf, Nsample)
# s = solver(Nspan = N, Nfield = Nfield, interpolated = interpolated, Rtol = Rtol, Atol = Atol)


# start_time = time.time()
# rescaling = k**3/2/np.pi**2
# SigmaAB = s.SigmaAB_solution(k = k, part = "Re") * rescaling #to have the dimensionless power spectrum
# print("The power spectra computation took", time.time() - start_time, "sec to run")


# #Plotting the power spectra
# plt.semilogy(N, np.absolute(SigmaAB[0, 0]), label = "$XX = \\varphi\\varphi$")
# plt.semilogy(N, np.absolute(SigmaAB[1, 1]), label = "$XX = \sigma\sigma$")

# plt.axvline(x = N_exit, ls = "--", color = "grey")
# plt.xlabel("$N$", fontsize = 15)
# #plt.ylim(1e-5, 1e8)
# #plt.title("$m^2/H^2 = 10, \\rho/H = 15$", fontsize = 15)
# plt.grid(True)
# plt.legend()
# plt.show()




############################
############################
#Compute the spectra/bispectra simultaneously
############################
############################

#sys.exit()
#Define background span and torelance
start_time = time.time()
Nspan = np.linspace(-5, 10, 500)
Nfield = 2
Rtol, Atol = [1e-3, 1e-3, 1e-3], [1e-6, 1e-3, 1e-180]

mdl = model(N = Nspan, Nfield = Nfield, interpolated = interpolated)


#Define the various modes
N_exit = 0
kt = mdl.k_mode(N_exit)
print("mode k = ", kt)
k1 = 1*kt
k2 = 1*kt
k3 = 1*kt
#print("k1 = {}, k2 = {}, k3 = {}".format(k1, k2, k3))


#Define the span for the spectra/bispectra
DeltaN = 10 # number of efolds before horizon crossing
Ni, Nf, Nsample = N_exit - DeltaN, 10, 1000 # sets initial and final efolds for transport equation integration
N = np.linspace(Ni, Nf, Nsample)
s = solver(Nspan = N, Nfield = Nfield, interpolated = interpolated, Rtol = Rtol, Atol = Atol)

N1 = mdl.efold_mode(k1, Nf, N_exit)
N2 = mdl.efold_mode(k2, Nf, N_exit)
N3 = mdl.efold_mode(k3, Nf, N_exit)
print("N1 = {}, N2 = {}, N3 = {}".format(N1, N2, N3))

start_time = time.time()
f = s.f_solution(k1 = k1, k2 = k2, k3 = k3)
print("The spectra/bispectra computation took", time.time() - start_time, "sec to run")


S = 1/(2*np.pi)**4/((f[0][0, 0][-1]*k1**3/2/np.pi**2 + f[1][0, 0][-1]*k2**3/2/np.pi**2 + f[2][0, 0][-1]*k3**3/2/np.pi**2)/3)**2 * (k1*k2*k3)**2 * f[6][0, 0, 0]

#print(f[0][0, 0][-1])
#print(S[-1])


P1 = np.absolute(f[0][0, 0]) * k1**3/2/np.pi**2
P2 = np.absolute(f[1][0, 0]) * k2**3/2/np.pi**2
P3 = np.absolute(f[2][0, 0]) * k3**3/2/np.pi**2

#plt.semilogy(N, P1/P1[-1], label = "$\mathcal{P}_{k_1}/\mathcal{P}^{\star}$")
#plt.semilogy(N, P2/P1[-1], label = "$\mathcal{P}_{k_2}/\mathcal{P}^{\star}$")
#plt.semilogy(N, P3/P1[-1], label = "$\mathcal{P}_{k_3}/\mathcal{P}^{\star}$")

print("Power spectrum : " + str(f[0][0, 0][-1]))
print("3pt function : " + str(f[6][0, 0, 0][-1]))

plt.semilogy(N, np.absolute(f[0][0, 0]), label = "$\\varphi\\varphi$")
plt.semilogy(N, np.absolute(f[6][0, 0, 0]), label = "$\\varphi\\varphi\\varphi$")




#plt.semilogy(N, np.absolute(S), label = "$S(k_1, k_2, k_3)$")
print("My program took", time.time() - start_time, "to run")

#Cross/entropic correlations
# plt.semilogy(N, np.absolute(f[6][1, 1, 1]), label = "$\cal{F}\cal{F}\cal{F}$")
# plt.semilogy(N, np.absolute(f[6][1, 1, 3]), label = "$\cal{F}\cal{F}\pi_{\cal{F}}$")
# plt.semilogy(N, np.absolute(f[6][1, 3, 3]), label = "$\cal{F}\pi_{\cal{F}}\pi_{\cal{F}}$")
# plt.semilogy(N, np.absolute(f[6][3, 3, 3]), label = "$\pi_{\cal{F}}\pi_{\cal{F}}\pi_{\cal{F}}$")

plt.semilogy(N, np.absolute(f[6][0, 0, 1]), label = "$\\varphi\\varphi\sigma$")
# plt.semilogy(N, np.absolute(f[6][1, 1, 0]), label = "$\cal{F}\cal{F}\zeta$")

# plt.semilogy(N, np.absolute(f[6][3, 0, 0]), label = "$\pi_{\cal{F}}\zeta\zeta$")
# plt.semilogy(N, np.absolute(f[6][3, 3, 0]), label = "$\pi_{\cal{F}}\pi_{\cal{F}}\zeta$")





#Plotting the bispectra
#plt.semilogy(N, np.absolute(S), label = "$\\frac{\langle\zeta\zeta\zeta\\rangle (k_1k_2k_3)^2}{A_s^2 (2\pi)^2}$")
# plt.semilogy(N, np.absolute(f[6][0, 0, 0]), label = "$\langle\zeta\zeta\zeta\\rangle$")
# plt.semilogy(N, np.absolute(f[6][0, 0, 2]), label = "$\langle\zeta\zeta\pi_{\zeta}\\rangle$")
# plt.semilogy(N, np.absolute(f[6][0, 2, 2]), label = "$\langle\zeta\pi_{\zeta}\pi_{\zeta}\\rangle$")
# plt.semilogy(N, np.absolute(f[6][2, 2, 2]), label = "$\langle\pi_{\zeta}\pi_{\zeta}\pi_{\zeta}\\rangle$")



# plt.semilogy(N, np.absolute(f[6][1, 1, 1]), label = "$\langle\cal{F}\cal{F}\cal{F}\\rangle$")

# plt.semilogy(Nspan, np.absolute(f[6][3, 1, 1]))
# plt.semilogy(Nspan, np.absolute(f[6][1, 1, 3]))
# plt.semilogy(Nspan, np.absolute(f[6][1, 3, 1]), label = "$\cal{F}\pi_{\cal{F}}\cal{F}$")

# plt.semilogy(Nspan, np.absolute(f[6][3, 3, 1]))
# plt.semilogy(Nspan, np.absolute(f[6][3, 1, 3]))
# plt.semilogy(Nspan, np.absolute(f[6][1, 3, 3]), label = "$\langle\cal{F}\pi_{\cal{F}}\pi_{\cal{F}}\\rangle$")

# plt.semilogy(Nspan, np.absolute(f[6][3, 3, 3]), label = "$\pi_{\cal{F}}\pi_{\cal{F}}\pi_{\cal{F}}$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 0, 3]))#, label = "$\zeta\zeta\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 3, 0]))#, label = "$\zeta\pi_{\cal{F}}\zeta$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 0, 0]), label = "$\pi_{\cal{F}}\zeta\zeta$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 3, 3]))#, label = "$\zeta\zeta\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 0, 3]))#, label = "$\zeta\pi_{\cal{F}}\zeta$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 3, 0]), label = "$\pi_{\cal{F}}\pi_{\cal{F}}\zeta$")



# plt.semilogy(Nspan, np.absolute(f[6][0, 0, 1]))#, label = "$\zeta\zeta\cal{F}$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 1, 0]))#, label = "$\zeta\zeta\cal{F}$")
# plt.semilogy(Nspan, np.absolute(f[6][1, 0, 0]), label = "$\zeta\zeta\cal{F}$")


#plt.semilogy(Nspan, np.absolute(f[6][0, 1, 1]), label = "$\langle\zeta\cal{F}\cal{F}\\rangle$")


#plt.semilogy(N, np.abs(np.absolute(f[6][0, 0, 0]))*1e40, label = "$\langle\zeta\zeta\zeta\\rangle$")
#plt.semilogy(N, P1, label = "$\langle\zeta\zeta\\rangle$")



#plt.semilogy(Nspan, np.absolute(f[6][0, 1, 3]))
#P = np.absolute(f[0][0, 0]) * rescaling
# plt.semilogy(N, np.absolute(f[0][0, 0]), label = "$\langle\zeta\zeta\\rangle$")
# #plt.semilogy(N, np.absolute(f[0][2, 2]), label = "$\langle\pi_{\zeta}\pi_{\zeta}\\rangle$")
# plt.semilogy(N, np.absolute(f[0][1, 1]), label = "$\langle\cal{F}\cal{F}\\rangle$")
# plt.semilogy(N, np.absolute(f[0][3, 3]), label = "$\langle\pi_{\cal{F}}\pi_{\cal{F}}\\rangle$")
# plt.semilogy(N, np.absolute(f[0][0, 1]), label = "$\langle\zeta\cal{F}\\rangle$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 0, 2]))#, label = "$\zeta\zeta\pi_{\zeta}$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 2, 0]))#, label = "$\zeta\pi_{\zeta}\zeta$")
# plt.semilogy(Nspan, np.absolute(f[6][2, 0, 0]), label = "$\pi_{\zeta}\zeta\zeta$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 2, 2]))#, label = "$\zeta\pi_{\zeta}\pi_{\zeta}$")
# plt.semilogy(Nspan, np.absolute(f[6][2, 0, 2]))#, label = "$\pi_{\zeta}\zeta\pi_{\zeta}$")
# plt.semilogy(Nspan, np.absolute(f[6][2, 2, 0]), label = "$\pi_{\zeta}\pi_{\zeta}\zeta$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 0, 3]))#, label = "$\zeta\zeta\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 3, 0]))#, label = "$\zeta\pi_{\cal{F}}\zeta$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 0, 0]), label = "$\pi_{\cal{F}}\zeta\zeta$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 3, 3]))#, label = "$\zeta\pi_{\cal{F}}\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 0, 3]))#, label = "$\pi_{\cal{F}}\zeta\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 3, 0]), label = "$\pi_{\cal{F}}\pi_{\cal{F}}\zeta$")

# plt.semilogy(Nspan, np.absolute(f[6][1, 1, 3]))#, label = "$\cal{F}\cal{F}\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[6][1, 3, 1]))#, label = "$\cal{F}\pi_{\cal{F}}\cal{F}$")
# plt.semilogy(Nspan, np.absolute(f[6][3, 1, 1]), label = "$\pi_{\cal{F}}\cal{F}\cal{F}$")

# plt.semilogy(Nspan, np.absolute(f[6][0, 0, 1]))#, label = "$\zeta\zeta\cal{F}$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 1, 0]))#, label = "$\zeta\cal{F}\zeta$")
# plt.semilogy(Nspan, np.absolute(f[6][1, 0, 0]), label = "$\cal{F}\zeta\zeta$")


plt.xlabel("$N$", fontsize = 15)
#plt.ylabel("Comoving pyT", fontsize = 15)
plt.title("$N_1 = ${}, $N_2 = ${}, $N_3 = ${}".format(round(N1, 2), round(N2, 2), round(N3, 2)))
plt.axvline(x = N_exit, ls = "--", color = "grey", label = "Horizon crossing")
#plt.title("$f_{NL}^{eq} = 1.8 \\times 10^{-2}$", fontsize = 15)
plt.legend(loc = "upper center", frameon = False, ncol = 2)
plt.grid()
plt.show()















