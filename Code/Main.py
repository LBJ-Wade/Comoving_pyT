import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import time


from background_inputs import background_inputs
from model import model
from solver import solver


############################
############################
#Importing and interpolating background functions
N_load = np.load("/home/dwerth/Documents/PhD/Python/DoubleQuad/time_dependent_functions/N.npy")
H_load = np.load("/home/dwerth/Documents/PhD/Python/DoubleQuad/time_dependent_functions/H.npy")
etaperp_load = np.sqrt(np.load("/home/dwerth/Documents/PhD/Python/DoubleQuad/time_dependent_functions/etaperp2.npy"))
Omega_load = 0
m2_load = np.load("/home/dwerth/Documents/PhD/Python/DoubleQuad/time_dependent_functions/m2.npy")
Vabc_load = 0
Rasbs_load = 0
Rabcs_load = 0
Rasbs_c_load = 0

background = background_inputs(N_load, H_load, etaperp_load, Omega_load, m2_load, Vabc_load, Rasbs_load, Rabcs_load, Rasbs_c_load)
interpolated = background.output()


############################
############################
#Plotting the background evolution

N = np.linspace(1, 70, 1000)
Nfield = 2
mdl = model(N = N, Nfield = Nfield, interpolated = interpolated)


N_exit = N[0] + 5
k = mdl.k_mode(N_exit)
print("The computed k mode is k = {}".format(k))
plt.axvline(x = N_exit, ls = "--", color = "grey")
#plt.semilogy(N, 1/(mdl.a_f(N)*mdl.H_f(N)), label = "$(aH)^{-1}$")
#plt.semilogy(N, mdl.epsilon_f(N), label = "$\epsilon$")
plt.semilogy(N, mdl.m2_f(N)[0], label = "$m^2/H^2$")
plt.semilogy(N, mdl.dm2_f(N)[0], label = "$m^2/H^2$")
#plt.semilogy(N, mdl.etaperp_f(N), label = "$\eta_\perp = \omega_1/H$")
#plt.semilogy(N, np.absolute(mdl.eta_f(N)), label = "$|\eta|$")
#plt.semilogy(N, np.sqrt(1/(2*mdl.epsilon_f(N))), label = "$1/\sqrt{2\epsilon M_p^2}$")
#plt.semilogy(N, mdl.H_f(N), label = "$H^{-1}$")
#plt.semilogy(N, mdl.a_f(N)/mdl.H_f(N)/(1 + mdl.a_f(N)*mdl.H_f(N)/k), label = "$\\frac{a}{H(1+aH/k)}$")
#plt.semilogy(N, mdl.a_f(N)/mdl.H_f(N), label = "$a/H$")
#plt.semilogy(N, k/mdl.H_f(N)**2, label = "$k/H^2$")
#plt.semilogy(N, mdl.omega1_f(N)**3, label = "$\omega_1$")
plt.semilogy(N, -4/mdl.H_f(N) * np.sqrt(2/mdl.epsilon_f(N)) * (mdl.omega1_f(N)**2)/mdl.Mp)

plt.xlabel("$N$", fontsize = 15)
plt.legend()
plt.xlim(N[0], N[-1])
plt.grid(True)
#plt.ylim(100, 1e9)
plt.show()


############################
############################
#Compute the power spectra
"""
Nspan = np.linspace(0, 70, 1000)
Nfield = 2
Rtol, Atol = [1e-3, 1e-5, 1e-3], [1e-50, 1e-100, 1e-6]

mdl = model(N = Nspan, Nfield = Nfield, interpolated = interpolated)
s = solver(Nspan = Nspan, Nfield = Nfield, interpolated = interpolated, Rtol = Rtol, Atol = Atol)

N_exit = Nspan[0] + 5
k = mdl.k_mode(N_exit)

start_time = time.time()
rescaling = k**3/2/np.pi**2
SigmaAB = s.SigmaAB_solution(k = k, part = "Re") * rescaling #to have the dimensionless power spectrum
print("The power spectra computation took", time.time() - start_time, "sec to run")
SigmaAB = SigmaAB/mdl.normalization(N_exit)


#Plotting the power spectra
plt.semilogy(Nspan, np.absolute(SigmaAB[0, 0]), label = "$XX = \zeta\zeta$")
plt.semilogy(Nspan, np.absolute(SigmaAB[1, 1]), label = "$XX = \cal{F}\cal{F}$")
plt.semilogy(Nspan, np.absolute(SigmaAB[0, 1]), label = "$XX = \zeta\cal{F}$")
plt.semilogy(Nspan, np.absolute(SigmaAB[2, 2]), label = "$XX = \pi_{\zeta}\pi_{\zeta}$")
plt.semilogy(Nspan, np.absolute(SigmaAB[3, 3]), label = "$XX = \pi_{\cal{F}}\pi_{\cal{F}}$")
plt.semilogy(Nspan, np.absolute(SigmaAB[2, 3]), label = "$XX = \pi_\zeta\pi_{\cal{F}}$")
plt.semilogy(Nspan, np.absolute(SigmaAB[0, 2]), label = "$XX = \zeta\pi_\zeta$")
plt.semilogy(Nspan, np.absolute(SigmaAB[1, 3]), label = "$XX = \cal{F}\pi_{\cal{F}}$")

plt.axvline(x = N_exit, ls = "--", color = "grey")
plt.xlabel("$N$", fontsize = 15)
plt.xlim(Nspan[0] - 3, Nspan[-1])
plt.ylabel("Re$(\Sigma^{XX})/\cal{P}_{\zeta}$", fontsize = 15)
plt.grid(True)
plt.legend()
plt.show()
"""


############################
############################
#Compute the spectra/bispectra simultaneously

Nspan = np.linspace(1, 70, 1000)
#Nspan = np.linspace(1, 1.5, 3)
Nfield = 2
Rtol, Atol = [1e-3, 1e-3, 1e-3], [1e-6, 1e-3, 1e-50]

mdl = model(N = Nspan, Nfield = Nfield, interpolated = interpolated)
s = solver(Nspan = Nspan, Nfield = Nfield, interpolated = interpolated, Rtol = Rtol, Atol = Atol)

N_exit = Nspan[0] + 4
kt = mdl.k_mode(N_exit)
alpha = 0.1
beta = 0.1#1./3.
k1 = kt#/4*(1 + alpha + beta)
k2 = kt#/4*(1 - alpha + beta)
k3 = kt#/2*(1 - beta)
rescaling = k1**3/2/np.pi**2
print("k1 = {}, k2 = {}, k3 = {}".format(k1, k2, k3))

start_time = time.time()
f = s.f_solution(k1 = k1, k2 = k2, k3 = k3)
print("The spectra/bispectra computation took", time.time() - start_time, "sec to run")


S = 1/(2*np.pi)**4/(f[0][0, 0][-1]*k1**3/2/np.pi**2)**2 * (k1*k2*k3)**2 * f[6][0, 0, 0]



#Plotting the bispectra
plt.semilogy(Nspan, np.absolute(f[6][0, 0, 0]), label = "$\zeta\zeta\zeta$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 0, 2]), "--", label = "$\zeta\zeta\pi_{\zeta}$")
# plt.semilogy(Nspan, np.absolute(f[6][0, 2, 2]), "--", label = "$\zeta\pi_{\zeta}\pi_{\zeta}$")
# plt.semilogy(Nspan, np.absolute(f[6][2, 2, 2]), "--", label = "$\pi_{\zeta}\pi_{\zeta}\pi_{\zeta}$")



# plt.semilogy(Nspan, np.absolute(f[6][1, 1, 1]), label = "$\cal{F}\cal{F}\cal{F}$")

# plt.semilogy(Nspan, np.absolute(f[6][3, 1, 1]))
# plt.semilogy(Nspan, np.absolute(f[6][1, 1, 3]))
# plt.semilogy(Nspan, np.absolute(f[6][1, 3, 1]), label = "$\cal{F}\pi_{\cal{F}}\cal{F}$")

# plt.semilogy(Nspan, np.absolute(f[6][3, 3, 1]))
# plt.semilogy(Nspan, np.absolute(f[6][3, 1, 3]))
# plt.semilogy(Nspan, np.absolute(f[6][1, 3, 3]), label = "$\cal{F}\pi_{\cal{F}}\pi_{\cal{F}}$")

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


# plt.semilogy(Nspan, np.absolute(f[6][0, 1, 1]), label = "$\zeta\cal{F}\cal{F}$")



print(f[6][0, 0, 0][-1])




#plt.semilogy(Nspan, np.absolute(f[6][0, 1, 3]))

plt.semilogy(Nspan, np.absolute(f[0][0, 0]) * rescaling, color = "C1", label = "$\cal{P}_\zeta$")
# plt.semilogy(Nspan, np.absolute(f[0][2, 2]) * rescaling, label = "$XX = \pi_{\zeta}\pi_{\zeta}$")
# plt.semilogy(Nspan, np.absolute(f[0][1, 1]) * rescaling, label = "$XX = \cal{F}\cal{F}$")
# plt.semilogy(Nspan, np.absolute(f[0][3, 3]) * rescaling, label = "$XX = \pi_{\cal{F}}\pi_{\cal{F}}$")
# plt.semilogy(Nspan, np.absolute(f[0][0, 2]) * rescaling, label = "$XX = \zeta\pi_\zeta$")
# plt.semilogy(Nspan, np.absolute(f[3][0, 2]) * rescaling, label = "$XX = \zeta\pi_\zeta$ (Im)")
# plt.semilogy(Nspan, np.absolute(f[3][1, 3]) * rescaling, label = "$XX = \cal{F}\pi_{\cal{F}}$ (Im)")

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
plt.title("$k_1 = ${}, $k_2 = ${}, $k_3 = ${}".format(round(k1, 3), round(k2, 3), round(k3, 3)))
plt.axvline(x = N_exit, ls = "--", color = "grey")
plt.legend(loc = "upper right")
plt.grid()
plt.show()
