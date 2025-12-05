import asdf
import crds
import galsim
import numpy as np
import roman_datamodels

from .parameters import default_parameters_dictionary, nborder

__all__ = ["ReadNoise"]

# Default read noise value
read_noise = 8.5  # e-

class ReadNoise(object):
    def __init__(self, usecrds=False, metadata=None, rng=None, seed=None):
        self.read_noise = read_noise
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)

        if rng is None and seed is None:
            self.seed = 45
        if rng is None:
            self.rng = galsim.GaussianDeviate(seed)
        else:
            self.rng = galsim.GaussianDeviate(rng)

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
            reftypes=["readnoise"],
            observatory="roman",
        )
        with asdf.open(ref_file["readnoise"]) as f:
            self.read_noise = f["roman"]["data"][
                nborder:-nborder, nborder:-nborder
            ].copy()
        # self.read_noise *= u.DN

    def apply(self, img, n_reads=1.0):
        if not self.usecrds:
            gn = galsim.GaussianNoise(self.rng, sigma=self.read_noise)
            img.addNoise(gn)
        else:
            # The read noise is averaged down like 1/\sqrt{n_reads},
            # where n_reads is the number of reads contributing to
            # the resultant.
            noise = np.zeros(img.array.shape, dtype="f4")
            self.rng.generate(noise)
            noise = noise * self.read_noise / (n_reads**0.5)
            img.array += noise
            return img
