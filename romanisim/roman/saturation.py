import asdf
import crds
import numpy as np
import roman_datamodels

from astropy import units as u

from .parameters import default_parameters_dictionary, nborder

__all__ = ["Saturation"]


class Saturation(object):
    def __init__(self, usecrds=False, metadata=None, saturation_level=100000):
        self.usecrds = usecrds
        self.metadata = metadata
        self.saturation_level = saturation_level
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)

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
            reftypes=["saturation"],
            observatory="roman",
        )
        with asdf.open(ref_file["saturation"]) as f:
            self.saturation_map = f["roman"]["data"][
                nborder:-nborder, nborder:-nborder
            ].copy()
        self.saturation_map *= u.DN

    def apply(self, img):
        if not self.usecrds:
            saturation_array = np.ones_like(img.array) * self.saturation_level
            where_sat = np.where(img.array > saturation_array)
            img.array[where_sat] = saturation_array[where_sat]
        else:
            # The CRDS saturation references is in DN
            # Resultants exceeding the saturation level are clipped at
            # the saturation level and marked as saturated.

            # [from roman_imsim] this maybe should be better applied at
            # read time? it's not actually clear to me what the right
            # thing to do is in detail.
            if not isinstance(img, u.Quantity):
                img *= u.DN
            img = np.clip(img, 0 * u.DN, self.saturation_map, out=img)

            # m = resultants >= saturation
            # dq[m] |= parameters.dqbits['saturated']
            # return resultants, dq

        return img
