import os

import galsim.Image as Image
import numpy as np

from astropy.io import ascii
from astropy.time import Time

######################################################################################################
# Default values
######################################################################################################
gain = 1.0
pixel_scale = 0.11  # arcsec / pixel
diameter = 2.36  # meters
obscuration = 0.32
collecting_area = 3.757e4  # cm^2, from Cycle 7
exptime = 139.8  # s
dark_current = 0.015  # e-/pix/s
nonlinearity_beta = -6.0e-7
reciprocity_alpha = 0.0065
read_noise = 8.5  # e-
n_dithers = 6
nborder = 4  # number of border pixels used for reference pixels.
# Physical pixel size
pixel_scale_mm = 0.01  # mm
stray_light_fraction = 0.1
n_sca = 18
n_pix_tot = 4096
n_pix = 4088
jitter_rms = 0.014
charge_diffusion = 0.1

# Maxinum allowed angle from the telecope solar panels to the sun in degrees.
max_sun_angle = 36.0

# Which bands should use the long vs short pupil plane files for the PSF.
# F184, K213
longwave_bands = ["F184", "K213"]
# R062, Z087, Y106, J129, H158, W146, SNPrism, Grism_0thOrder, Grism_1stOrder.
# Note that the last three are not imaging bands.
non_imaging_bands = ["Grism_0thOrder", "Grism_1stOrder", "SNPrism"]
shortwave_bands = [
    "R062",
    "Z087",
    "Y106",
    "J129",
    "H158",
    "W146",
] + non_imaging_bands

band_name_map = {
    "F062": "R062",
    "F087": "Z087",
    "F106": "Y106",
    "F129": "J129",
    "F158": "H158",
    "F146": "W146",
    "F213": "K213",
    "Prism": "SNPrism",
}

# These are from https://roman.gsfc.nasa.gov/science/WFI_technical.html, as of October, 2023
thermal_backgrounds = {
    "R062": 0.00,  # e-/pix/s
    "Z087": 0.00,
    "Y106": 0.00,
    "J129": 0.00,
    "H158": 0.04,
    "F184": 0.17,
    "K213": 4.52,
    "W146": 0.98,
    "SNPrism": 0.00,
    "Grism_0thOrder": 0.00,
    "Grism_1stOrder": 0.00,
}

persistence_coefficients = (
    np.array(
        [
            0.045707683,
            0.014959818,
            0.009115737,
            0.00656769,
            0.005135571,
            0.004217028,
            0.003577534,
            0.003106601,
        ]
    )
    / 100.0
)

# IPC kernel is unnormalized at first.  We will normalize it.
ipc_kernel = np.array(
    [
        [0.001269938, 0.015399776, 0.001199862],
        [0.013800177, 1.0, 0.015600367],
        [0.001270391, 0.016129619, 0.001200137],
    ]
)
ipc_kernel /= np.sum(ipc_kernel)
ipc_kernel = Image(ipc_kernel)

# parameters in the fermi model = [ A, x0, dx, a, r, half_well]
# The following parameters are for H4RG-lo, the conservative model for low influence level x.
# The info and implementation can be found in roman_detectors.applyPersistence() and roman_detectors.fermi_linear().
persistence_fermi_parameters = np.array(
    [0.017, 60000.0, 50000.0, 0.045, 1.0, 50000.0]
)

######################################################################################################
# [TODO] Temporary implementation for accessing roman-technical-information repo
######################################################################################################
roman_tech_repo_path = (
    "/hpc/home/yf194/Work/projects/roman-technical-information/"
)
FPSPerformance_path = os.path.join(
    roman_tech_repo_path, "data", "WideFieldInstrument", "FPSPerformance"
)
dark_current_summary = os.path.join(
    FPSPerformance_path, "WFI_Dark_current_summary.ecsv"
)
# Dark current
# Columns in the summary file: ['SCU', 'SCA', 'Dark Current - Median', 'Dark Current - Mean', 'Percentage Passing Requirement']
# The 18th (counting from 0) row: All detectors (MAP)
try:
    data = ascii.read(dark_current_summary)
    dark_current = data[18]["Dark Current - Median"]
except RuntimeError as e:
    print(
        f" {e} Failed to fetch WFI_Dark_current_summary.ecsv, use default value for dark_current"
    )
thermal_backgrounds_summary = os.path.join(
    roman_tech_repo_path,
    "data",
    "WideFieldInstrument",
    "Imaging",
    "Backgrounds",
    "internal_thermal_backgrounds.ecsv",
)
try:
    data = ascii.read(thermal_backgrounds_summary)
    for i in range(len(data["filter"])):
        band_name = (
            band_name_map[data["filter"][i]]
            if data["filter"][i] in band_name_map
            else data["filter"][i]
        )
        thermal_backgrounds[band_name] = data[i]["rate"]
except RuntimeError as e:
    print(
        f" {e} Failed to fetch internal_thermal_backgrounds.ecsv, use default value for thermal_backgrounds"
    )
# persistence_summary = os.path.join(
#     FPSPerformance_path, "WFI_Persistence_summary.ecsv"
# )
# persistence_exp_fits_summary = os.path.join(
#     FPSPerformance_path, "WFI_Persistence_exp_fits.ecsv"
# )

######################################################################################################
# Default configuration parameters
######################################################################################################

read_pattern = {
    3: [
        [1],
        [2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17],
        [18],
    ],
    4: [
        [1],
        [2],
        [3, 4],
        [5, 6, 7, 8, 9],
        [10 + x for x in range(8)],
        [18 + x for x in range(8)],
        [26 + x for x in range(8)],
        [34 + x for x in range(10)],
        [44],
    ],
    5: [
        [1],
        [2],
        [3, 4],
        [5, 6, 7, 8, 9, 10],
        [11 + x for x in range(6)],
        [17 + x for x in range(6)],
        [23 + x for x in range(6)],
        [29 + x for x in range(6)],
        [35 + x for x in range(8)],
        [43 + x for x in range(10)],
        [53 + x for x in range(10)],
        [63],
    ],
    6: [
        [1],
        [2],
        [3, 4],
        [5, 6, 7, 8, 9, 10],
        [11 + x for x in range(6)],
        [17 + x for x in range(6)],
        [23 + x for x in range(8)],
        [31 + x for x in range(8)],
        [39 + x for x in range(8)],
        [47 + x for x in range(8)],
        [55 + x for x in range(8)],
        [63 + x for x in range(10)],
        [73 + x for x in range(10)],
        [83 + x for x in range(10)],
        [93],
    ],
    7: [
        [1],
        [2],
        [3, 4],
        [5 + x for x in range(8)],
        [13 + x for x in range(8)],
        [21 + x for x in range(10)],
        [31 + x for x in range(10)],
        [41 + x for x in range(12)],
        [53 + x for x in range(12)],
        [65 + x for x in range(12)],
        [77 + x for x in range(12)],
        [89 + x for x in range(12)],
        [101 + x for x in range(12)],
        [113 + x for x in range(12)],
        [125],
    ],
    8: [
        [1],
        [2],
        [3, 4],
        [5 + x for x in range(8)],
        [13 + x for x in range(9)],
        [22 + x for x in range(10)],
        [32 + x for x in range(12)],
        [44 + x for x in range(16)],
        [60 + x for x in range(16)],
        [76 + x for x in range(16)],
        [92 + x for x in range(16)],
        [108 + x for x in range(16)],
        [124 + x for x in range(16)],
        [140 + x for x in range(16)],
        [156],
    ],
    9: [
        [1],
        [2],
        [3, 4],
        [5 + x for x in range(8)],
        [13 + x for x in range(8)],
        [21 + x for x in range(10)],
        [31 + x for x in range(12)],
        [43 + x for x in range(16)],
        [59 + x for x in range(16)],
        [75 + x for x in range(16)],
        [91 + x for x in range(16)],
        [107 + x for x in range(16)],
        [123 + x for x in range(32)],
        [155 + x for x in range(32)],
        [187],
    ],
    10: [
        [1],
        [2],
        [3, 4],
        [5 + x for x in range(8)],
        [13 + x for x in range(12)],
        [25 + x for x in range(16)],
        [41 + x for x in range(16)],
        [57 + x for x in range(16)],
        [73 + x for x in range(16)],
        [89 + x for x in range(16)],
        [105 + x for x in range(32)],
        [137 + x for x in range(32)],
        [169 + x for x in range(32)],
        [201 + x for x in range(32)],
        [233],
    ],
    11: [
        [1],
        [2],
        [3, 4],
        [5 + x for x in range(16)],
        [21 + x for x in range(16)],
        [37 + x for x in range(16)],
        [53 + x for x in range(32)],
        [85 + x for x in range(32)],
        [117 + x for x in range(32)],
        [149 + x for x in range(32)],
        [181 + x for x in range(32)],
        [213 + x for x in range(32)],
        [245 + x for x in range(32)],
        [277 + x for x in range(32)],
        [309],
    ],
    12: [
        [1],
        [2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21],
        [22, 23, 24, 25],
        [26, 27, 28, 29],
        [30, 31, 32, 33],
        [34, 35, 36, 37],
        [38, 39, 40, 41],
        [42 + x for x in range(8)],
        [50 + x for x in range(8)],
        [58],
    ],
    13: [
        [1],
        [2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21],
        [22 + x for x in range(8)],
        [30 + x for x in range(8)],
        [38 + x for x in range(8)],
        [46 + x for x in range(8)],
        [54 + x for x in range(8)],
        [62 + x for x in range(8)],
        [70 + x for x in range(8)],
        [78],
    ],
    14: [
        [1],
        [2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15],
        [16 + x for x in range(8)],
        [24 + x for x in range(8)],
        [32 + x for x in range(8)],
        [40 + x for x in range(8)],
        [48 + x for x in range(8)],
        [56 + x for x in range(8)],
        [64 + x for x in range(8)],
        [72 + x for x in range(8)],
        [80 + x for x in range(16)],
        [96],
    ],
    15: [
        [1],
        [2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21],
        [22 + x for x in range(12)],
        [34 + x for x in range(12)],
        [46 + x for x in range(12)],
        [58 + x for x in range(12)],
        [70 + x for x in range(12)],
        [82 + x for x in range(12)],
        [94 + x for x in range(12)],
        [106 + x for x in range(12)],
        [118],
    ],
    16: [
        [1],
        [2, 3],
        [4, 5],
        [6 + x for x in range(8)],
        [14 + x for x in range(12)],
        [26 + x for x in range(12)],
        [38 + x for x in range(12)],
        [50 + x for x in range(12)],
        [62 + x for x in range(12)],
        [74 + x for x in range(12)],
        [86 + x for x in range(12)],
        [98 + x for x in range(16)],
        [114 + x for x in range(16)],
        [130 + x for x in range(16)],
        [146],
    ],
    17: [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8, 9, 10, 11, 12],
        [13 + x for x in range(8)],
        [21 + x for x in range(12)],
        [33 + x for x in range(16)],
        [49 + x for x in range(16)],
        [65 + x for x in range(16)],
        [81 + x for x in range(16)],
        [97 + x for x in range(16)],
        [113 + x for x in range(16)],
        [129 + x for x in range(32)],
        [161 + x for x in range(32)],
        [193],
    ],
    # note: these MA tables are not part of the PRD and are intended
    # only to support DMS testing.  In particular, 110 is a
    # 16-resultant imaging table needed to demonstrate ramp fitting
    # performance.
    # both 109 and 110 have CRDS darks for both spectroscopic and
    # imaging modes
    109: [
        [1],
        [2, 3],
        [5, 6, 7],
        [10, 11, 12, 13],
        [15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26],
        [27, 28, 29, 30, 31, 32],
        [33, 34, 35, 36, 37, 38],
        [39, 40, 41, 42, 43],
        [44],
    ],
    110: [
        [1],
        [2, 3, 4],
        [5, 6, 7],
        [8, 9, 10],
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
        [20, 21, 22],
        [23, 24, 25],
        [26, 27, 28],
        [29, 30, 31],
        [32, 33, 34],
        [35, 36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44],
    ],
}

default_parameters_dictionary = {
    "instrument": {
        "name": "WFI",
        "detector": "WFI07",
        "optical_element": "F184",
    },
    "ephemeris": {
        "time": Time("2026-01-01").mjd,
        "spatial_x": 0.0,
        "spatial_y": 0.0,
        "spatial_z": 0.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "velocity_z": 0.0,
    },
    "exposure": {
        "start_time": Time("2026-01-01T00:00:00"),
        "type": "WFI_IMAGE",
        "ma_table_number": 4,
        "read_pattern": read_pattern[4],
        # Changing the default MA table to be 4 (C2A_IMG_HLWAS)
        # as MA table 1 (DEFOCUS_MOD) is not supported
    },
    "pointing": {
        "target_ra": 270.0,
        "target_dec": 66.0,
        "target_aperture": "WFI_CEN",
        "pa_aperture": 0.0,
    },
    "velocity_aberration": {"scale_factor": 1.0},
    "wcsinfo": {
        "aperture_name": "WFI_CEN",
        "ra_ref": 270.0,
        "dec_ref": 66.0,
        "v2_ref": 0,
        "v3_ref": 0,
        "roll_ref": 0,
        "vparity": -1,
        "v3yangle": -60.0,
        # I don't know what vparity and v3yangle should really be,
        # but they are always -1 and -60 in existing files.
    },
}
