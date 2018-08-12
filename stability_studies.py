"""
Module: stability_studies.py
------------------------------------------

Studies the stability properties of the linear advection scheme
"""

import basis
import element
import math
import cmath
import numpy
import matplotlib.pyplot as plt
import parameters
import solution
import mesh
import solve_explicit_global
import os.path


def get_w_val_closest_abs_dist(w_vals, w_ref_real, w_ref_imag):

	"""
	Return the w_val value that is closest to the reference w_ref value. By closest,
	we will return the w (eigenvalue) with the smallest magnitude difference on the complex plane 
	"""

	w_val_index = None
	w_val_abs_diff = None

	for w_val in w_vals:

		if w_val_index is None:
			w_val_index = w_vals.index(w_val)
			w_val_abs_diff = math.sqrt((w_val.real - w_ref_real)**2. + (w_val.imag - w_ref_imag)**2.) 

		else:
			diff = math.sqrt((w_val.real - w_ref_real)**2. + (w_val.imag - w_ref_imag)**2.) 
			if diff < w_val_abs_diff:
				w_val_index = w_vals.index(w_val)
				w_val_abs_diff = diff

	return w_vals[w_val_index]


def filter_eigenvalues_into_physical_mode(w_real_physical_mode_list, w_imag_physical_mode_list, w_vals, k_list, k_index):

	"""
	Filter the eigenvalues to only choose the one that corresponds to the physical mode

	We will essentially look at the approximation (a linear approximation) for what the next
	eigenvalue in the mode curve should be. Then, the eigenvalue that comes closest to 
	this linear approximation will be used for that mode curve.

	We will use the dispersion curves to figure out which eigenvalue is associated 
	with which mode. Then, knowing this, we can add to the dissipation curve as well


	:param w_real_physical_mode_list: The list holding the physical mode dispersion results (w_real)
	:param w_imag_modes_lists: The list holding the physical mode dissipation results (w_imag)
	:param w_vals: The eigenvalues for this current k (wavenumber). 
	:param k_list: The list holding all the k values (used for the linear interpolation)
	:param k_index: The index for which wavenumber we are at in the list (k_list). Will only be able to
		use linear extrapolation when k_index > 1 (assuming 0 based indexing)
	"""

	if k_index == 0 or k_index == 1:
		
		# Choose the mode that is closest to the theoretical (for small k, should follow 
		# theoertical quite closely). This is done when linear extrapolation is not an option

		w_theo_real = k_list[k_index] * parameters.CONST_BETA
		w_theo_imag = 0.0
		w_val_closest = get_w_val_closest_abs_dist(w_vals, w_theo_real, w_theo_imag)

		w_real_physical_mode_list.append(w_val_closest.real)
		w_imag_physical_mode_list.append(w_val_closest.imag)

	else:

		# With more than 2 elements in the list, can now do linear extrapolation

		# Get the linearly extrapolated values
			
		k3 = k_list[k_index]
		k2 = k_list[k_index-1]
		k1 = k_list[k_index-2]

		w2_real = w_real_physical_mode_list[-1]
		w1_real = w_real_physical_mode_list[-2]

		w2_imag = w_imag_physical_mode_list[-1]
		w1_imag = w_imag_physical_mode_list[-2]

		# Linearly extrapolated value
		linearly_extrapolated_w_real = ((k3-k2)/(k2-k1))*(w2_real-w1_real) + w2_real
		linearly_extrapolated_w_imag = ((k3-k2)/(k2-k1))*(w2_imag-w1_imag) + w2_imag

		w_val_closest = get_w_val_closest_abs_dist(w_vals, linearly_extrapolated_w_real, linearly_extrapolated_w_imag)

		w_real_physical_mode_list.append(w_val_closest.real)
		w_imag_physical_mode_list.append(w_val_closest.imag)


def study_dispersion_dissipation():

	"""
	Study the dispersion and dissipation properties of the scheme
	"""

	# =================================
	#          Preprocessing
	# =================================

	print "Generate Basis Functions"
	b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)
	knots_smooth = b_spline_basis.get_smoothed_knots()
	b_spline_basis_smooth = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)

	print "Generate Reference Element"
	reference_element = element.ReferenceElement(b_spline_basis)
	reference_element_smooth = element.ReferenceElement(b_spline_basis_smooth)

	print "Generate Mesh"
	patches, patch_faces = mesh.generate_mesh(reference_element)
	patches_smooth, patch_faces_smooth = mesh.generate_mesh(reference_element_smooth)


	# =================================
	#         Stability Study
	# =================================

	# Look at the first patch (uniform or smooth)
	patch = patches_smooth[0]

	A, B, C = patch.A_k, patch.B_k, patch.C_k
	n_eig = A.shape[0]  # Number of eigenvalues

	h = patch.x_vertices[1] - patch.x_vertices[0]
	
	# go up till 63ish
	k_min = 0
	k_max = int(parameters.CONST_NUM_BASIS * math.pi * (1./h))
	num_k = 5*k_max
	k_vals = numpy.linspace(k_min, k_max, num_k)


	# Theoretical dispersion results (for only the physical mode)
	w_real_theo_physical_mode_list = [parameters.CONST_BETA * k for k in k_vals]  

	# Theoretical Dissipation results (for only the physical mode)
	w_imag_theo_physical_mode_list = [0.0 for k in k_vals]  

	# Dispersion results (only the physical mode)
	w_real_physical_mode_list = []

	# Dissipation results (only the physical mode)
	w_imag_physical_mode_list = []


	k_index = 0
	for k in k_vals:

		cos_kh = math.cos(k*h)
		sin_kh = math.sin(k*h)

		A_k = complex(cos_kh, -sin_kh) * A
		B_k = B
		C_k = complex(cos_kh, sin_kh) * C

		M = A_k + B_k + C_k

		# Get the eigenvalues
		l_eigenvalue = numpy.linalg.eigvals(M)

		# Get the w values (temporal frequency) for all modes
		w_vals = [ x * complex(0,1) for x in l_eigenvalue]

		filter_eigenvalues_into_physical_mode(w_real_physical_mode_list, w_imag_physical_mode_list, 
			w_vals, k_vals, k_index)
		k_index += 1


	# Normalize the dispersion data
	k_vals_hat = [k*float(h)/(parameters.CONST_NUM_BASIS) for k in k_vals]
	w_real_physical_mode_list = [w_real*float(h)/parameters.CONST_NUM_BASIS for w_real in w_real_physical_mode_list]
	w_real_theo_physical_mode_list = [w_real_theo*float(h)/parameters.CONST_NUM_BASIS for w_real_theo in w_real_theo_physical_mode_list]

	# Plot the dissipation and dispersion curves for each mode

	# Dispersion Curves	
	plt.figure(1)

	plt.plot(k_vals_hat, w_real_physical_mode_list, marker='o', markersize=1)		
	plt.plot(k_vals_hat, w_real_theo_physical_mode_list, linestyle="--")
	
	plt.xlabel("k * h / n_dof_element")
	plt.ylabel("Real(w) * h / n_dof_element")
	
	plt.grid()
	plt.legend()

	
	# Dissipation Curves
	plt.figure(2)
	
	plt.plot(k_vals_hat, w_imag_physical_mode_list)	
	plt.plot(k_vals_hat, w_imag_theo_physical_mode_list, linestyle="--")
	
	plt.xlabel("k")
	plt.ylabel("Imag(w)")

	plt.grid()
	plt.legend()
	

	plt.show(block=True)


	# Output the data into a file
	output_file_dir = "Results/Dispersion_Dissipation_Results/Tau1.0/Smoothed_Knots"

	P = parameters.CONST_P
	nBasis = parameters.CONST_NUM_BASIS

	file_name = "%s_P%d_NBasis%d" % ("DispersionDissipationResults", P, nBasis)
	file_rel_path = os.path.join(output_file_dir, file_name)

	with open(file_rel_path, "w") as fp:

		fp.write("Data Format: k_val_normalized w_real_normalized w_imag \n")
		fp.write("num_pts %d \n" % (len(k_vals_hat)))

		for i in range(len(k_vals_hat)):

			fp.write("%.14e %.14e %.14e \n" % (k_vals_hat[i], w_real_physical_mode_list[i], 
				w_imag_physical_mode_list[i]))
		

def study_spectra():

	"""
	Study the stability properties of the scheme by looking at the spectra
	"""

	b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)
	#knots_smooth = b_spline_basis.get_smoothed_knots()
	#b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)
	
	reference_element = element.ReferenceElement(b_spline_basis)
	patches, patch_faces = mesh.generate_mesh(reference_element)
	solution.set_initial_condition(patches)


	# Discretization matrix
	Lh = solve_explicit_global.assemble_Lh(patches)
	
	w = numpy.linalg.eigvals(Lh)
	w_real = numpy.real(w)
	w_imag = numpy.imag(w)

	plt.figure(1)
	plt.scatter(w_real, w_imag, c='b')
	
	plt.xlabel("Real")
	plt.ylabel("Imaginary")
	
	plt.grid()
	plt.legend()

	plt.show(block=True)


	# Print the data to a file

	# Output the data into a file
	output_file_dir = "Results/Spectra_Distribution/Tau0.0/Uniform_Knots"

	file_name = "%s_P%d_NPatch%d_NBasis%d" % ("SpectraDistribution", parameters.CONST_P, 
		parameters.CONST_NUM_PATCH, parameters.CONST_NUM_BASIS)

	file_rel_path = os.path.join(output_file_dir, file_name)

	with open(file_rel_path, "w") as fp:

		fp.write("w_real w_imag \n")
		for i in range(w.shape[0]):
			fp.write("%.14e %.14e \n" % (w_real[i], w_imag[i]))




def study_spectral_radius():

	"""
	Study the growth of the spectral radius for the discretization matrix
	"""

	p_vals = [2,3,4,5,6,7]

	n_levels = [1, 3, 5, 7, 9, 11]

	num_patches_vals = []
	spectral_radius_vals = []
	for i in range(len(p_vals)):
		spectral_radius_vals.append([])

	for p_val in p_vals:

		parameters.CONST_P = p_val

		# Setup 1
		b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)
		knots_smooth = b_spline_basis.get_smoothed_knots()
		b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)
		reference_element = element.ReferenceElement(b_spline_basis)

		for i in n_levels:
			
			print "p: %d, i: %d" % (p_val, i)

			parameters.CONST_NUM_PATCH = i*parameters.CONST_NUM_PATCH_ML0

			# Setup 2
			patches, patch_faces = mesh.generate_mesh(reference_element)
			solution.set_initial_condition(patches)

			# Discretization matrix. Get the spectral radius
			Lh = solve_explicit_global.assemble_Lh(patches)
			eig_vals = numpy.linalg.eigvals(Lh)

			spectral_radius = 0
			for eig_val in eig_vals:
				modulus_eig_val = math.sqrt(eig_val.real**2.0 + eig_val.imag**2.0)
				if modulus_eig_val > spectral_radius:
					spectral_radius = modulus_eig_val

			if p_vals.index(p_val) == 0:
				num_patches_vals.append(parameters.CONST_NUM_PATCH)
			spectral_radius_vals[p_vals.index(p_val)].append(spectral_radius)


	plt.figure(1)
	
	for p_val in p_vals:
		i = p_vals.index(p_val)
		curve_label = "P : %d" % (p_val)
		plt.plot(num_patches_vals, spectral_radius_vals[i], marker='o', markersize=5, label=curve_label)	
	
	plt.xlabel("Num Patches")
	plt.ylabel("Spectral radius")

	plt.grid()
	plt.legend()
	
	plt.show(block=True)


	# Output the data into a file
	output_file_dir = "Results/Spectral_Radius_Results/Tau0.5/Smoothed_Knots"

	for p_val in p_vals:

		file_name = "%s_P%d" % ("SpectralRadius", p_val)
		file_rel_path = os.path.join(output_file_dir, file_name)

		spectral_radius_vals_p = spectral_radius_vals[p_vals.index(p_val)]

		with open(file_rel_path, "w") as fp:

			fp.write("num_patches spectral_radius \n")

			for i in range(len(spectral_radius_vals_p)):
				fp.write("%.14e %.14e \n" % (num_patches_vals[i], spectral_radius_vals_p[i]))


if __name__ == "__main__":
	study_spectra()
	#study_dispersion_dissipation()
	#study_spectral_radius()


