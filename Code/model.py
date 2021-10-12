import numpy as np
from scipy.misc import derivative



class model():
	"""
	This class defines the model (background functions, M, Delta, I, A, B, C tensors and u^A_B, u^A_BC tensors)
	where the functions have been optimized to be computed once at each integration step.
	The minimal background interpolated functions should be imported using the background_inputs class. 
	The background functions are vectorized.
	"""

	def __init__(self, N, Nfield, interpolated):
		"""
		All the functions are evaluated once at N (can be vectorized for background quantities
		or just a float for tensorial quantities)
		"""
		self.N = N
		self.Nfield = Nfield
		self.Mp = 1
		self.interpolated = interpolated #(H, etaperp, Omega, m2, Vabc, Rasbs, Rabcs, Rasbs_c)


		#Evaluating the background quantities at time N
		N = self.N
		self.H = self.H_f(N)
		self.etaperp = self.etaperp_f(N)
		self.Omega = self.Omega_f(N)
		self.m2 = self.m2_f(N)
		self.Vabc = self.Vabc_f(N)
		self.Rasbs = self.Rasbs_f(N)
		self.Rabcs = self.Rabcs_f(N)
		self.Rasbs_c = self.Rasbs_c_f(N)
		self.a = self.a_f(N)
		self.epsilon = self.epsilon_f(N)
		self.eta = self.eta_f(N)
		self.dvsig = self.dvsig_f(N)
		self.omega1 = self.omega1_f(N)
		self.u1 = self.u1_f(N)
		self.Omega2 = self.Omega2_f(N)
		self.dm2 = self.dm2_f(N)
		self.dOmega = self.dOmega_f(N)

		self.scale = self.scale_f(N)
		self.dscale = self.dscale_f(N)


	############################
	############################
	#Defining time-dependent background functions (as functions of the number of efolds)
	############################
	############################

	############################
	#Required as extern inputs
	def H_f(self, N):
		"""
		Hubble parameter as a function of the number of efolds
		"""
		return self.interpolated[0](N)

	def etaperp_f(self, N):
		"""
		Etaperp (bending) as a function of the number of efolds
		"""
		return self.interpolated[1](N)

	def Omega_f(self, N):
		"""
		Omega antisymmetric square matrix of size (Nfield-1 * Nfield-1)
		"""
		return np.zeros((self.Nfield-1, self.Nfield-1))

	def m2_f(self, N): 
		"""
		Square mass matrix for the entropic modes 
		(depends on the potential, Omega, omega1, epsilon, H, Riemann tensor)
		Needs to be adapted from the m2 format
		"""
		return np.eye(self.Nfield-1) * self.interpolated[3](N)

	def Vabc_f(self, N):
		"""
		Potential Derivative with respect to three entropic directions
		"""
		return np.zeros((self.Nfield-1, self.Nfield-1, self.Nfield-1))

	def Rasbs_f(self, N):
		"""
		Riemann tensor with two indices corresponding to adiabatic directions
		"""
		return np.zeros((self.Nfield-1, self.Nfield-1))

	def Rabcs_f(self, N):
		"""
		Riemann tensor with one index corresponding to adiabatic direction
		"""
		return np.zeros((self.Nfield-1, self.Nfield-1, self.Nfield-1))

	def Rasbs_c_f(self, N):
		"""
		Covariant derivative of Riemann tensor with indices corresponding to adiabatic directions
		"""
		return np.zeros((self.Nfield-1, self.Nfield-1, self.Nfield-1))


	############################
	#Deduced background functions
	def a_f(self, N):
		"""
		Scale factor as a function of the number of efolds
		"""
		return np.exp(N)

	def dH_f(self, N):
		"""
		Derivative of the Hubble rate (with respect to cosmic time)
		"""
		dHdN = derivative(self.H_f, N, dx = 1e-6)
		return self.H_f(N)*dHdN

	def epsilon_f(self, N):
		"""
		Slow-roll parameter as a function of the number of efolds
		(deduced from H)
		"""
		dHdN = derivative(self.H_f, N, dx = 1e-6)
		return -dHdN/self.H_f(N)

	def eta_f(self, N):
		"""
		Second slow-roll parameter as a function of the number of efolds
		(deduced from epsilon)
		"""
		depsilondN = derivative(self.epsilon_f, N, dx = 1e-3)
		return depsilondN/self.epsilon_f(N)

	def dvsig_f(self, N):
		"""
		Total field velocity
		"""
		return np.sqrt(2 * self.Mp**2 * self.H_f(N)**2 * self.epsilon_f(N))

	def omega1_f(self, N):
		"""
		omega1 parameter (etaperp rescaled by H)
		"""
		return self.etaperp_f(N) * self.H_f(N)

	def u1_f(self, N):
		"""
		Logarithmic derivative of omega1
		(deduced from omega1)
		"""
		domega1dN = derivative(self.omega1_f, N, dx = 1e-6)
		return domega1dN/self.omega1_f(N)

	def Omega2_f(self, N):
		"""
		Squared Omega antisymmetric square matrix of size (Nfield-1 * Nfield-1)
		(deduced from Omega)
		"""
		Om = self.Omega_f(N)
		return np.dot(Om, Om)

	def dm2_f(self, N):
		"""
		Time (efolds) derivative of m2 matrix
		(deduced from m2)
		"""
		dm2dN = derivative(self.m2_f, N, dx = 1e-6)
		return np.eye(self.Nfield-1) * dm2dN

	def dOmega_f(self, N):
		"""
		Time (efolds) derivative of Omega matrix
		(deduced from Omega)
		"""
		return np.zeros((self.Nfield-1, self.Nfield-1))


	############################
	############################
	#Choosing the number of efolds before horizon exit so that it fixes the mode k, 
	#and defining the power spectrum normalization
	############################
	############################

	def k_mode(self, N_exit):
		return self.a_f(N_exit) * self.H_f(N_exit)

	def normalization(self, N_exit):
		"""
		Function that returns the single field slow-roll
		power spectrum amplitude to normalize Sigma (for the dimensionless power spectrum)
		"""
		return self.H_f(N_exit)**2/8/np.pi**2/self.epsilon_f(N_exit)/self.Mp**2

	def scale_f(self, N):
		"""
		Function that defines the rescaled a to improve performance
		"""
		k = 1
		a = self.a_f(N)
		H = self.H_f(N)
		return a/(1. + a*H/k)/H

	def dscale_f(self, N):
		"""
		Function that defines the derivative of scale function
		"""
		k = 1
		a = self.a_f(N)
		H = self.H_f(N)
		Hd = self.dH_f(N)
		return -Hd/H/H*a/(1. + a*H/k) + a/(1. + a*H/k) - a*(a*H*H/k + a*Hd/k)/(1. + a*H/k)/(1. + a*H/k)/H



	############################
	############################
	#Defining the u_AB tensor for the power spectrum calculations
	############################
	############################

	def Delta_ab(self):
		Nfield = self.Nfield
		Deltaab = np.eye(Nfield)

		Deltaab[0, 0] = 1/(2 * self.Mp**2 * self.epsilon)
		return Deltaab

	def I_ab(self):
		Nfield = self.Nfield
		Iab = np.zeros((Nfield, Nfield))

		Iab[0, 1] = self.omega1/self.Mp * np.sqrt(2/self.epsilon)#(self.dvsig * self.etaperp)/(self.Mp**2 * self.epsilon)
		Iab[1:, 1:] = self.Omega
		return Iab

	def M_ab(self, k):
		Nfield = self.Nfield

		Mab = (-(k**2)/(self.a**2))*np.eye(Nfield)
		Mab[0, 0] = (-(k**2)/(self.a**2)) * 2 * self.Mp**2 * self.epsilon
		Mab[1, 1] += -4*self.omega1**2#(2*(self.dvsig)**2 * (self.etaperp)**2)/(self.Mp**2 * self.epsilon)
		Mab[1: ,1:] += self.Omega2 - self.m2
		return Mab

	def u_AB(self, k):
		Nfield = self.Nfield
		H = self.H
		s = self.scale
		ds = self.dscale
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		uAB = np.zeros((2*Nfield, 2*Nfield))

		uAB[:Nfield, :Nfield] = -self.I_ab()/H
		uAB[:Nfield, Nfield:] = self.Delta_ab()/H /s
		uAB[Nfield:, :Nfield] = self.M_ab(k)/H *s
		uAB[Nfield:, Nfield:] = (self.I_ab()).T/H - 3*self.H*np.eye(Nfield)/H + ds/s*np.eye(Nfield)/H 
		return uAB


	############################
	############################
	#Defining the u_ABC tensor for bispectrum calculations
	############################
	############################

	def A_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2

		I1 = np.zeros((Nfield-1, Nfield-1))
		I1[0, 0] = 1
		X = self.epsilon/2 * self.m2 + self.dm2/2 + np.dot((self.epsilon*np.transpose(self.Omega) + np.transpose(self.dOmega) - np.transpose(self.m2)/self.H), self.Omega)
		Y = 2*self.epsilon*self.H**2*self.Mp**2 * self.Rasbs - self.omega1**2 * I1
		J1 = np.zeros(Nfield-1)
		J1[0] = 1
		Z = self.Vabc - 4*np.sqrt(2*self.epsilon)*self.H*self.Mp * (self.omega1*np.reshape(np.kron(J1, np.transpose(self.Rasbs)), (Nfield-1, Nfield-1, Nfield-1)) - np.tensordot(self.Omega, self.Rabcs, axes = ([1, 0]))) + 2*self.epsilon*self.H**2*self.Mp**2*self.Rasbs_c

		Aabc[0, 1, 1] += 4*(self.epsilon - self.eta) * self.omega1**2
		Aabc[0, 0, 0] += -2*self.Mp**2 * self.epsilon*(self.epsilon + self.eta) * k2k3/self.a**2
		Aabc[0, 0, 1] += -2*np.sqrt(2*self.epsilon) * self.omega1/self.H * self.Mp * k1k2/self.a**2
		Aabc[1, 1, 1] += -4/self.H * np.sqrt(2/self.epsilon) * (self.omega1**3)/self.Mp
		Aabc[0, 1, 1] += 4*(self.eta + 2*self.u1) * self.omega1**2
		##Aabc[0, 1, 1:] += -8*self.Omega[0,:]/self.H * self.omega1**2
		Aabc[1:, 1:, 0] += 2*X
		##Aabc[1:, 1:, 0] += 2*self.epsilon * self.Omega2
		Aabc[1:, 1:, 1] += -2*Y/self.H * np.sqrt(2/self.epsilon) * self.omega1/self.Mp
		##Aabc[1:, 1:, 0] += 2*self.epsilon * self.Omega2
		Aabc[1:, 1:, 0] += -self.epsilon * k1k2/self.a**2 * np.identity(Nfield-1)
		##Aabc[1:, 1:, 1:] += -4/3 * np.sqrt(2*self.epsilon) * self.H * self.Mp * np.tensordot(self.Omega, self.Rabcs, axes = ([1, 0]))
		Aabc[1:, 1:, 1:] += -1/3 * Z

		return Aabc

	def A_abc_fast(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		
		Aabc[0, 0, 0] += -2*self.Mp**2 * self.epsilon*(self.epsilon + self.eta) * k2k3
		Aabc[0, 0, 1] += -2*np.sqrt(2*self.epsilon) * self.omega1/self.H * self.Mp * k1k2
		Aabc[1:, 1:, 0] += -self.epsilon * k1k2 * np.identity(Nfield-1)

		return Aabc

	def A_abc_slow(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))

		X = self.epsilon/2 * self.m2 + self.dm2/2 + np.dot((self.epsilon*np.transpose(self.Omega) + np.transpose(self.dOmega) - np.transpose(self.m2)/self.H), self.Omega)
		I1 = np.zeros((Nfield-1, Nfield-1))
		I1[0, 0] = 1
		Y = 2*self.epsilon*self.H**2*self.Mp**2 * self.Rasbs - self.omega1**2 * I1
		J1 = np.zeros(Nfield-1)
		J1[0] = 1
		Z = self.Vabc - 4*np.sqrt(2*self.epsilon)*self.H*self.Mp * (self.omega1*np.reshape(np.kron(J1, np.transpose(self.Rasbs)), (Nfield-1, Nfield-1, Nfield-1)) - np.tensordot(self.Omega, self.Rabcs, axes = ([1, 0]))) + 2*self.epsilon*self.H**2*self.Mp**2*self.Rasbs_c

		Aabc[0, 1, 1] += 4*(self.epsilon - self.eta) * self.omega1**2
		Aabc[1, 1, 1] += -4/self.H * np.sqrt(2/self.epsilon) * (self.omega1**3)/self.Mp
		Aabc[0, 1, 1] += 4*(self.eta + 2*self.u1) * self.omega1**2
		##Aabc[0, 1, 1:] += -8*self.Omega[0,:]/self.H * self.omega1**2
		Aabc[1:, 1:, 0] += 2*X
		##Aabc[1:, 1:, 0] += 2*self.epsilon * self.Omega2
		##Aabc[1:, 1:, 1] += -2*Y/self.H * np.sqrt(2/self.epsilon) * self.omega1/self.Mp
		##Aabc[1:, 1:, 0] += -2*self.epsilon * self.Omega2
		##Aabc[1:, 1:, 1:] += 4/3 * np.sqrt(2*self.epsilon) * self.H * self.Mp * np.tensordot(self.Omega, self.Rabcs, axes = ([1, 0]))
		Aabc[1:, 1:, 1:] += -1/3 * Z

		return Aabc

	def B_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Babc = np.zeros((Nfield, Nfield, Nfield))

		I1 = np.zeros((Nfield-1, Nfield-1))
		I1[0, 0] = 1
		Y = 2*self.epsilon*self.H**2*self.Mp**2 * self.Rasbs - self.omega1**2 * I1

		Babc[0, 1, 0] += -2*np.sqrt(2/self.epsilon) * (self.epsilon - self.eta) * self.omega1/self.Mp
		Babc[1, 1, 0] += 4/self.epsilon/self.H * self.omega1**2/self.Mp**2
		Babc[0, 1, 0] += -np.sqrt(2/self.epsilon) * (self.eta + 2*self.u1) * self.omega1/self.Mp
		##Babc[0, 1:, 0] += 2*np.sqrt(2/self.epsilon) * self.omega1/self.Mp * self.Omega[0,:]/self.H
		##Babc[0, 1:, 1:] += -self.epsilon * self.Omega
		Babc[1:, 1:, 0] += Y/self.H/self.epsilon/self.Mp**2
		##Babc[0, 1:, 1:] += 2*self.epsilon*self.Omega
		##Babc[1:, 1:, 1:] += 4/3 * np.sqrt(2*self.epsilon) * self.H * self.Mp * np.transpose(self.Rabcs, (2, 0, 1))

		return Babc

	def C_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Cabc = np.zeros((Nfield, Nfield, Nfield))
		k1k3 = (k2**2 - k1**2 - k3**2)/2
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2

		Cabc[0, 0, 0] += (self.epsilon - self.eta)/2/self.epsilon/self.Mp**2
		Cabc[0, 0, 0] += (self.epsilon/2 - 2)/2/self.Mp**2 * k1k3/k2**2
		Cabc[0, 0, 0] += self.epsilon/8/self.Mp**2 * k3**2/k1**2/k2**2 * k1k2
		Cabc[0, 0, 1] += -1/2/self.H * np.sqrt(2/self.epsilon**3) * self.omega1/self.Mp**3
		Cabc[1:, 1:, 0] += self.epsilon * np.identity(Nfield-1)
		Cabc[0, 1:, 1:] += -1/self.Mp**2 * k1k3/k1**2 * np.identity(Nfield-1)

		return Cabc

	def u_ABC(self, k1, k2, k3):
		Nfield = self.Nfield
		s = self.scale
		S = np.ones((Nfield, Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		H = self.H
		uABC = np.zeros((2*Nfield, 2*Nfield, 2*Nfield))

		#Symmetrize the tensors (B and C over the first two indices)
		B123 = (self.B_abc(k1, k2, k3) + np.transpose(self.B_abc(k2, k1, k3), (1, 0, 2)))/2
		B231 = (self.B_abc(k2, k3, k1) + np.transpose(self.B_abc(k3, k2, k1), (1, 0, 2)))/2#(np.transpose(self.B_abc(k2, k3, k1), (1, 2, 0)) + np.transpose(self.B_abc(k3, k2, k1), (2, 1, 0)))/2#np.transpose(B123, (1, 2, 0))#np.transpose(B123, (2, 1, 0))
		B132 = (self.B_abc(k1, k3, k2) + np.transpose(self.B_abc(k3, k1, k2), (1, 0, 2)))/2#(np.transpose(self.B_abc(k1, k3, k2), (0, 2, 1)) + np.transpose(self.B_abc(k3, k1, k2), (2, 0, 1)))/2#np.transpose(B123, (0, 2, 1))
		
		A123 = (self.A_abc(k1, k2, k3) 
			+ np.transpose(self.A_abc(k2, k3, k1), (1, 2, 0)) 
			+ np.transpose(self.A_abc(k3, k1, k2), (2, 0, 1)) 
			+ np.transpose(self.A_abc(k2, k1, k3), (1, 0, 2)) 
			+ np.transpose(self.A_abc(k3, k2, k1), (2, 1, 0)) 
			+ np.transpose(self.A_abc(k1, k3, k2), (0, 2, 1)))/6
		
		C123 = (self.C_abc(k1, k2, k3) + np.transpose(self.C_abc(k2, k1, k3), (1, 0, 2)))/2
		C132 = (self.C_abc(k1, k3, k2) + np.transpose(self.C_abc(k3, k1, k2), (1, 0, 2)))/2#(np.transpose(self.C_abc(k1, k3, k2), (0, 2, 1)) + np.transpose(self.C_abc(k3, k1, k2), (2, 0, 1)))/2#np.transpose(C123, (0, 2, 1))
		C321 = (self.C_abc(k3, k2, k1) + np.transpose(self.C_abc(k2, k3, k1), (1, 0, 2)))/2#(np.transpose(self.C_abc(k3, k2, k1), (2, 1, 0)) + np.transpose(self.C_abc(k2, k3, k1), (2, 0, 1)))/2#np.transpose(C123, (2, 1, 0))
		
		
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):

					uABC[i, j, k] = -B231[j, k, i]/H

					uABC[i, Nfield+j, k] = -C123[i, j, k]/H/s
					uABC[i, j, Nfield+k] = -C132[i, k, j]/H/s
					uABC[Nfield+i, Nfield+j, Nfield+k] = C321[k, j, i]/H/s

					uABC[i, Nfield+j, Nfield+k] = 0

					uABC[Nfield+i, j, k] = 3.*A123[i, j, k]/H*s

					uABC[Nfield+i, Nfield+j, k] = B132[i, k, j]/H

					uABC[Nfield+i, j, Nfield+k] = B123[i, j, k]/H

		# uABC[:Nfield, :Nfield, :Nfield] = -np.transpose(B231, (1, 2, 0))/H #- B231/H
		# uABC[Nfield:, Nfield:, :Nfield] = np.transpose(B132, (0, 2, 1))/H #B132/H
		# uABC[Nfield:, :Nfield, :Nfield] = 3 * A123/H #3 * A123/H *S
		# uABC[:Nfield, Nfield:, :Nfield] = -C123/H #- C123/H /S
		# uABC[:Nfield, :Nfield, Nfield:] = -np.transpose(C132, (0, 2, 1))/H #- C132/H /S
		# uABC[Nfield:, Nfield:, Nfield:] = np.transpose(C321, (2, 1, 0))/H #C321/H /S
		# uABC[Nfield:, :Nfield, Nfield:] = B123/H #B123/H
		
		return uABC