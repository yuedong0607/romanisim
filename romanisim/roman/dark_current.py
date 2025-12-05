import os
import asdf
import crds
import galsim
import roman_datamodels
from astropy import units as u
from astropy.io import ascii

from .parameters import default_parameters_dictionary, nborder, roman_tech_repo_path
from .gain import gain

__all__ = ["DarkCurrent"]

# Default dark current
dark_current = 0.015  # e-/pix/s

# Update dark current value with one from roman-technical-information
# Columns in the summary file: ['SCU', 'SCA', 'Dark Current - Median', 'Dark Current - Mean', 'Percentage Passing Requirement']
# The 18th (counting from 0) row: All detectors (MAP)
dark_current_summary = os.path.join(
    roman_tech_repo_path, "data", "WideFieldInstrument", "FPSPerformance", "WFI_Dark_current_summary.ecsv"
)
try:
    data = ascii.read(dark_current_summary)
    dark_current = data[18]["Dark Current - Median"]
except RuntimeError as e:
    print(
        f" {e} Failed to fetch WFI_Dark_current_summary.ecsv, use default value for dark_current"
    )


class DarkCurrent(object):
    def __init__(self, usecrds=False, metadata=None, rng=None, seed=None):
        self.dark_rate = dark_current
        self.gain = gain
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)

        if rng is None and seed is None:
            self.seed = 45
        if rng is None:
            self.rng = galsim.BaseDeviate(seed)
        else:
            self.rng = galsim.BaseDeviate(rng)

    def _get_crds_model(self, metadata=None):
        image_mod = roman_datamodels.datamodels.ImageModel.create_fake_data()
        meta = image_mod.meta
        meta["wcs"] = None
        for key in default_parameters_dictionary.keys():
            meta[key].update(default_parameters_dictionary[key])

        if metadata:
            for key in metadata.keys():
                meta[key].update(metadata[key])

        ref_file = crds.getreferences(
            image_mod.get_crds_parameters(),
            reftypes=["dark", "gain"],
            observatory="roman",
        )
        with asdf.open(ref_file["dark"]) as f:
            self.dark_rate = f["roman"]["dark_slope"][
                nborder:-nborder, nborder:-nborder
            ].copy()
        with asdf.open(ref_file["gain"]) as f:
            self.gain = f["roman"]["data"][nborder:-nborder, nborder:-nborder].copy()
        self.dark_rate * u.DN / u.s
        self.dark_rate *= self.gain
        if isinstance(self.dark_rate, u.Quantity):
            self.dark_rate = self.dark_rate.to(u.electron / u.s).value

    def apply(self, img, exptime):
        if not self.usecrds:
            total_dark_current = self.dark_rate * exptime
            dark_noise = galsim.DeviateNoise(
                galsim.PoissonDeviate(self.rng, total_dark_current)
            )
            img.addNoise(dark_noise)
        else:
            workim = img * 0
            workim += self.dark_rate * exptime
            workim.addNoise(galsim.PoissonNoise(self.rng))
            img += workim

        return img
