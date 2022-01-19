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
		self.interpolated = interpolated #(H, rho, mu, m2, kappa1, kappa2)


		#Evaluating the background quantities at time N
		N = self.N
		self.H = self.H_f(N)
		self.rho = self.rho_f(N)
		self.mu = self.mu_f(N)
		self.m2 = self.m2_f(N)
		self.kappa1 = self.kappa1_f(N)
		self.kappa2 = self.kappa2_f(N)


		self.mu1 = self.mu1_f(N)
		self.mu2 = self.mu2_f(N)
		self.mu3 = self.mu3_f(N)
		self.mu4 = self.mu4_f(N)
		self.mu5 = self.mu5_f(N)

		self.a = self.a_f(N)
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

	def rho_f(self, N):
		"""
		rho as a function of the number of efolds
		"""
		return self.interpolated[1](N)

	def mu_f(self, N):
		"""
		mu as a function of the number of efolds
		"""
		return self.interpolated[2](N)

	def m2_f(self, N):
		"""
		m2_f as a function of the number of efolds
		"""
		return self.interpolated[3](N)

	def kappa1_f(self, N):
		"""
		kappa1 as a function of the number of efolds
		"""
		return self.interpolated[4](N)

	def kappa2_f(self, N):
		"""
		kappa2 as a function of the number of efolds
		"""
		return self.interpolated[5](N)


	def mu1_f(self, N):
		"""
		mu1 as a function of the number of efolds
		"""
		return self.interpolated[6](N)


	def mu2_f(self, N):
		"""
		mu2 as a function of the number of efolds
		"""
		return self.interpolated[7](N)


	def mu3_f(self, N):
		"""
		mu3 as a function of the number of efolds
		"""
		return self.interpolated[8](N)


	def mu4_f(self, N):
		"""
		mu4 as a function of the number of efolds
		"""
		return self.interpolated[9](N)


	def mu5_f(self, N):
		"""
		mu5 as a function of the number of efolds
		"""
		return self.interpolated[10](N)



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
		return 0#-dHdN/self.H_f(N)

	def eta_f(self, N):
		"""
		Second slow-roll parameter as a function of the number of efolds
		(deduced from epsilon)
		"""
		#depsilondN = derivative(self.epsilon_f, N, dx = 1e-3)
		return 0#depsilondN/self.epsilon_f(N)

	def dvsig_f(self, N):
		"""
		Total field velocity
		"""
		return 0#np.sqrt(2 * self.Mp**2 * self.H_f(N)**2 * self.epsilon_f(N))

	def omega1_f(self, N):
		"""
		omega1 parameter (etaperp rescaled by H)
		"""
		return 0#self.etaperp_f(N) * self.H_f(N)

	def u1_f(self, N):
		"""
		Logarithmic derivative of omega1
		(deduced from omega1)
		"""
		#domega1dN = derivative(self.omega1_f, N, dx = 1e-6)
		return 0#domega1dN/self.omega1_f(N)

	def Omega2_f(self, N):
		"""
		Squared Omega antisymmetric square matrix of size (Nfield-1 * Nfield-1)
		(deduced from Omega)
		"""
		#Om = self.Omega_f(N)
		return 0#np.dot(Om, Om)

	def dm2_f(self, N):
		"""
		Time (efolds) derivative of m2 matrix
		(deduced from m2)
		"""
		#dm2dN = derivative(self.m2_f, N, dx = 1e-6)
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

	def efold_mode(self, k_exit, N_end, N_exit):
		"""
		Function that takes k_exit and computes the number of efolds
		before inflation end at which the mode exits the horizon.
		The approximation that H is constant was made.
		"""
		return N_end - np.log(k_exit/self.H_f(N_exit))


	def k_mode(self, N_exit):
		return self.a_f(N_exit) * self.H_f(N_exit)

	def normalization(self, N_exit):
		"""
		Function that returns the single field slow-roll
		power spectrum amplitude to normalize Sigma (for the dimensionless power spectrum)
		"""
		return 1#self.H_f(N_exit)**2/8/np.pi**2/self.epsilon_f(N_exit)/self.Mp**2

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

		Deltaab[0, 0] = 1
		return Deltaab

	def I_ab(self):
		Nfield = self.Nfield
		Iab = np.zeros((Nfield, Nfield))

		Iab[0, 1] = self.rho
		return Iab

	def M_ab(self, k):
		Nfield = self.Nfield

		Mab = (-(k**2)/(self.a**2))*np.eye(Nfield)
		#Mab[0, 0] = -((k**2)/(self.a**2) + self.m2)

		Mab[0, 0] = (-(k**2)/(self.a**2))
		Mab[1, 1] += -self.m2 - self.rho**2
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

		Aabc[0, 0, 0] += -2*self.mu1 + 2*self.mu4*k1k2/self.a**2

		Aabc[1, 1, 1] += self.kappa2*self.rho - 2*self.mu - self.kappa1*self.rho**2
		Aabc[0, 0, 1] += self.kappa1*k1k2/self.a**2

		return Aabc

	def A_abc_fast(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		

		Aabc[0, 0, 0] = + 2*self.mu4*k1k2

		Aabc[0, 0, 1] += self.kappa1*k1k2


		return Aabc

	def A_abc_slow(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))

		Aabc[0, 0, 0] += -2*self.mu1

		Aabc[1, 1, 1] += self.kappa2*self.rho - 2*self.mu + self.kappa1*self.rho**2

		return Aabc

	def B_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Babc = np.zeros((Nfield, Nfield, Nfield))
		k1k2 = (k3**2 - k1**2 - k2**2)/2

		Babc[0, 0, 0] += -2*self.mu2 + 2*self.mu5*k1k2/self.a**2

		Babc[0, 1, 1] += -2*self.kappa1*self.rho - self.kappa2


		return Babc

	def C_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Cabc = np.zeros((Nfield, Nfield, Nfield))
		k1k3 = (k2**2 - k1**2 - k3**2)/2
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2


		Cabc[0, 0, 0] += -2*self.mu3

		Cabc[0, 0, 1] += self.kappa1


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

		# print("blablabla")
		# for i in range(2*Nfield):
		# 	for j in range(2*Nfield):
		# 		print(uABC[0, i, j])
		# print("blablabla")
		
		return uABC