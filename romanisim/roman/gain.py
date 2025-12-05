import asdf
import crds
import roman_datamodels

from .parameters import default_parameters_dictionary, nborder

__all__ = ["Gain"]

# Default gain value
gain = 1.0


class Gain(object):
    def __init__(self, usecrds=False, metadata=None):
        self.gain = gain
        self.usecrds = usecrds
        self.metadata = metadata
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
            reftypes=["gain"],
            observatory="roman",
        )
        with asdf.open(ref_file["gain"]) as f:
            self.gain = f["roman"]["data"][
                nborder:-nborder, nborder:-nborder
            ].copy()

    def apply(self, img):
        img_arr = img.array
        img_arr /= self.gain
        img.array = img_arr
        return img
