import numpy as np
from scipy.interpolate import interp1d

class background_inputs():
	"""
	This class takes as inputs the mimimal background arrays and gives interpolated functions.
	This class needs to be executed just once at the beginning of the numerical integration
	"""

	def __init__(self, N_load, H_load, etaperp_load, Omega_load, m2_load, Vabc_load, Rasbs_load, Rabcs_load, Rasbs_c_load):

		#Importing pre-computed background quantities
		self.N_load = N_load
		self.H_load = H_load
		self.etaperp_load = etaperp_load
		self.Omega_load = Omega_load
		self.m2_load = m2_load
		self.Vabc_load = Vabc_load
		self.Rasbs_load = Rasbs_load
		self.Rabcs_load = Rabcs_load
		self.Rasbs_c_load = Rasbs_c_load

		#Creating continuous functions out of the imported functions
		self.H_f = interp1d(self.N_load, self.H_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.etaperp_f = interp1d(self.N_load, self.etaperp_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.Omega_f = 0
		self.m2_f = interp1d(self.N_load, self.m2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.Vabc_f = 0
		self.Rasbs_f = 0
		self.Rabcs_f = 0
		self.Rasbs_c_f = 0

		#Creating interpolating list
		self.interpolated = [self.H_f, self.etaperp_f, self.Omega_f, self.m2_f, self.Vabc_f, self.Rasbs_f, self.Rabcs_f, self.Rasbs_c_f]

	def output(self):
		"""
		Gives the interpolated continous functions that can be evaluated at any N
		"""
		return self.interpolated
