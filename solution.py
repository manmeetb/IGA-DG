"""
Module: solution.py
------------------------------------------

Set the solution information (such as the initial condition)
"""


import parameters
import patch


def set_initial_condition(patches):

	"""
	Set the initial condition on each patch by using a L2 projection

	:param patches: The list of patches that make up the mesh
	"""

	for patch in patches:
		patch.set_initial_condition(parameters.CONST_FUNCTION_IC)

