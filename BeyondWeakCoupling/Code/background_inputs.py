import numpy as np
from scipy.interpolate import interp1d

class background_inputs():
	"""
	This class takes as inputs the mimimal background arrays and gives interpolated functions.
	This class needs to be executed just once at the beginning of the numerical integration
	"""

	def __init__(self, N_load, H_load, rho_load, mu_load, m2_load, kappa1_load, kappa2_load, mu1_load, mu2_load, mu3_load, mu4_load, mu5_load):

		#Importing pre-computed background quantities
		self.N_load = N_load
		self.H_load = H_load
		self.rho_load = rho_load
		self.mu_load = mu_load
		self.m2_load = m2_load
		self.kappa1_load = kappa1_load
		self.kappa2_load = kappa2_load

		self.mu1_load = mu1_load
		self.mu2_load = mu2_load
		self.mu3_load = mu3_load
		self.mu4_load = mu4_load
		self.mu5_load = mu5_load

		#Creating continuous functions out of the imported functions
		self.H_f = interp1d(self.N_load, self.H_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate") #cubic
		self.rho_f = interp1d(self.N_load, self.rho_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.mu_f = interp1d(self.N_load, self.mu_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.m2_f = interp1d(self.N_load, self.m2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.kappa1_f = interp1d(self.N_load, self.kappa1_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.kappa2_f = interp1d(self.N_load, self.kappa2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")


		self.mu1_f = interp1d(self.N_load, self.mu1_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.mu2_f = interp1d(self.N_load, self.mu2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.mu3_f = interp1d(self.N_load, self.mu3_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.mu4_f = interp1d(self.N_load, self.mu4_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.mu5_f = interp1d(self.N_load, self.mu5_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")

		#Creating interpolating list
		self.interpolated = [self.H_f, self.rho_f, self.mu_f, self.m2_f, self.kappa1_f, self.kappa2_f, self.mu1_f, self.mu2_f, self.mu3_f, self.mu4_f, self.mu5_f]

	def output(self):
		"""
		Gives the interpolated continous functions that can be evaluated at any N
		"""
		return self.interpolated
