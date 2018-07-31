"""
Module: parameters.py
------------------------------------------

Parameters for the simulation
"""

import math

# =======================================
#           Patch Parameters
# =======================================

CONST_ML = 4 # The mesh level
CONST_P = 3  # The order on each patch

CONST_REFINEMENT_TYPE = "KNOT" # Patches / KNOT
CONST_NUM_PATCH_ML0 = 4  # The number of patches on the zeroth mesh level
CONST_NUM_BASIS_ML0 = 6  # The number of basis functions on the zeroth mesh level

if CONST_REFINEMENT_TYPE == "PATCHES":
	CONST_NUM_PATCH = CONST_NUM_PATCH_ML0 * (2**(CONST_ML))  # The number of patches
	CONST_NUM_BASIS = 12
elif CONST_REFINEMENT_TYPE == "KNOT":
	CONST_NUM_PATCH = 2
	CONST_NUM_BASIS = CONST_NUM_BASIS_ML0 * (2**(CONST_ML))  # The number of basis functions on each patch

CONST_BASIS_TYPE = "SMOOTHED" # UNIFORM / SMOOTHED


# =======================================
#        Physical Domain Parameters
# =======================================

CONST_X_RANGE = [0, 1]  # The 1D physical domain


# =======================================
#         1D Advection Parameters
# =======================================

CONST_BETA = 2.0
CONST_TAU = 0.0


# =======================================
#          Flow Solve Parameters
# =======================================

CONST_NUM_TIME_STEPS = 1000
CONST_DELTA_T = 0.0005
CONST_TIME_STEP_METHOD = "EXPLICIT_RK4_LOW_STORAGE"  # Time Stepping Method: EULER/EXPLICIT_RK4_LOW_STORAGE


# =======================================
#          Initial Condition
# =======================================

k_wave_number = 3
CONST_FUNCTION_IC = lambda x: math.sin(k_wave_number * 2 * math.pi * (x - CONST_X_RANGE[0])/(CONST_X_RANGE[1] - CONST_X_RANGE[0]))


# =======================================
#            Post Processing
# =======================================
CONST_ERROR_OUTPUT_FILE_DIR= "Results/Convergence_Orders/Smoothed_Knots/Knot_Refinement"
CONST_NUM_PATCH_PLOT_PTS = 25
CONST_OUTPUT_PLOT = True

