import asdf
import crds
import numpy as np
import roman_datamodels

from scipy import ndimage

from .parameters import default_parameters_dictionary, ipc_kernel

__all__ = ["IPC"]


class IPC(object):
    def __init__(self, usecrds=False, metadata=None):
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)
        else:
            self.ipc_kernel = ipc_kernel

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
            reftypes=["ipc"],
            observatory="roman",
        )
        with asdf.open(ref_file["ipc"]) as f:
            self.ipc_kernel = f["roman"]["data"]
            self.ipc_kernel /= np.sum(self.ipc_kernel)

    def apply(self, img, edge_treatment="extend", fill_value=None):
        if not self.usecrds:
            img.applyIPC(
                self.ipc_kernel,
                edge_treatment=edge_treatment,
                fill_value=fill_value,
            )
        else:
            if not fill_value:
                fill_value = 0.0
            img_arr = ndimage.convolve(
                img.array, self.ipc_kernel, mode="constant", cval=fill_value
            )
            img.array = img_arr
        return img
