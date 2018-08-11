"""
Module: convergence_order_test.py
------------------------------------------

Run several different P values and mesh levels to do a convergence order test

NOTE: This module will perform the tests by varying the CONST_P and CONST_ML 
values in the parameters.py module. Probably not the best way to do this since
these values in parameters.py should be constants
"""

CONST_CONVERGENCE_TEST_P_VALUES = [1,2,3]
CONST_CONVERGENCE_TEST_ML_VALUES = [0,1,2,3,4]

REFINEMENT_TYPE = "PATCHES"  # KNOT/PATCHES

import parameters
import simulation


def test_convergence_orders():

	# Do not output the plots at the end of a simulation
	parameters.CONST_OUTPUT_PLOT = False

	for p in CONST_CONVERGENCE_TEST_P_VALUES:
		for ml in CONST_CONVERGENCE_TEST_ML_VALUES:

			# Modify the values in parameters.py
			parameters.CONST_ML = ml
			parameters.CONST_P = p
			
			# Perform the specified type of refinement
			if REFINEMENT_TYPE == "PATCHES":
				parameters.CONST_NUM_PATCH = parameters.CONST_NUM_PATCH_ML0 * (2**(parameters.CONST_ML))  # Update the number of patches
				parameters.CONST_NUM_BASIS = 12
			elif REFINEMENT_TYPE == "KNOT":
				parameters.CONST_NUM_PATCH = 2
				parameters.CONST_NUM_BASIS = parameters.CONST_NUM_BASIS_ML0 * (2**(parameters.CONST_ML))  # Update the number of basis functions
			else:
				raise ValueError("Unkown REFINEMENT_TYPE")

			# Run the simulation
			simulation.simulate()


if __name__ == "__main__":
	test_convergence_orders()
