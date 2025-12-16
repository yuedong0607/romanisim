import asdf
import crds
import numpy as np
import roman_datamodels

from astropy import units as u

from .gain import gain
from .parameters import default_parameters_dictionary, nborder

__all__ = ["NLfunc", "Nonlinearity"]

# Default nonlinearity beta value
nonlinearity_beta = -6.0e-7


# def print_ram_usage(message=""):
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()

#     print(f"{message}, RSS (Resident Set Size): {mem_info.rss / 1024 / 1024:.2f} MB")
#     print(f"{message}, VMS (Virtual Memory Size): {mem_info.vms / 1024 / 1024:.2f} MB")


def NLfunc(x):
    return x + nonlinearity_beta * (x**2)


class Nonlinearity(object):
    def __init__(self, usecrds=False, metadata=None):
        self.gain = gain
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)

    def _get_crds_model(self, metadata=None):
        # Inverse linearity reference files are used to apply the
        # effect of classical non-linearity when constructing
        # L1 files, and linearity reference files are used to
        # remove it when constructing L2 files.
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
            reftypes=["inverselinearity", "gain"],
            observatory="roman",
        )
        # self.crds_model = roman_datamodels.datamodels.InverselinearityRefModel(
        #     ref_file
        # )
        with asdf.open(ref_file["inverselinearity"]) as f:
            self.crds_coeffs = self._repair_coefficients(
                coeffs=f["roman"]["coeffs"][
                    :, nborder:-nborder, nborder:-nborder
                ].copy(),
                dq=f["roman"]["dq"][nborder:-nborder, nborder:-nborder].copy(),
            )

        with asdf.open(ref_file["gain"]) as f:
            self.gain = f["roman"]["data"][
                nborder:-nborder, nborder:-nborder
            ].copy()

    def _repair_coefficients(self, coeffs, dq):
        """Fix cases of zeros and NaNs in non-linearity coefficients.

        This function replaces suspicious-looking non-linearity coefficients
        with identity transformation coefficients from a non-linearity
        perspective; all coefficients are zero except for the linear term,
        which is set to 1.

        This function doesn't try to make sure that the derivative of the
        correction is greater than 1, which we would expect for a non-linearity
        correction.

        Parameters
        ----------
        coeffs : np.ndarray[ncoeff, ny, nx] (float)
            Nonlinearity coefficients, starting with the constant term and
            increasing in power.

        dq : np.ndarray[n_resultant, ny, nx]
            Data Quality array

        Returns
        -------
        coeffs : np.ndarray[ncoeff, ny, nx] (float)
            "repaired" coefficients with NaNs and weird coefficients replaced
            with linear values with slopes of unity.

        dq : np.ndarray[n_resultant, ny, nx]
            DQ array marking pixels with improper non-linearity coefficients
        """
        res = coeffs.copy()

        nocorrection = np.zeros(coeffs.shape[0], dtype=coeffs.dtype)
        nocorrection[1] = 1.0  # "no correction" is just normal linearity.
        # For NaN, all zero, or flagged pixels, reset to no correction.
        m = (
            np.any(~np.isfinite(coeffs), axis=0)
            | np.all(coeffs == 0, axis=0)
            | (dq != 0)
        )
        res[:, m] = nocorrection[:, None]

        # [TODO] deal with dq
        # lin_dq_array = np.zeros(coeffs.shape[1:], dtype=np.uint32)
        # lin_dq_array[m] = parameters.dqbits["no_lin_corr"]
        # dq = np.bitwise_or(dq, lin_dq_array)
        # return res, dq
        return res

    def _evaluate_nl_polynomial(self, counts, coeffs, reversed=False):
        """Correct the observed DN for non-linearity.

        As electrons accumulate, they make it harder for the device to count
        future electrons due to classical non-linearity.  This function
        converts observed DN to what would have been seen absent
        non-linearity, using the provided non-linearity coefficients.

        Parameters
        ----------
        counts : np.ndarray[ny, nx] (float)
            Number of DN already in pixel
        coeffs : np.ndarray[ncoeff, ny, nx] (float)
            Coefficients of the non-linearity correction polynomials
        reversed : bool
            If True, the coefficients are in reversed order, which is the
            order that np.polyval wants them.  One can maybe save a little
            time reversing them once ahead of time.

        Returns
        -------
        corrected : np.ndarray[nx, ny] (float)
            The corrected number of DN
        """
        if reversed:
            cc = coeffs
        else:
            cc = coeffs[::-1, ...]

        if isinstance(counts, u.Quantity):
            unit = counts.unit
            counts = counts.value
        else:
            unit = None

        res = np.polyval(cc, counts)

        if unit is not None:
            res = res * unit

        return res

    def apply(self, img, electrons=False, reversed=False):
        """Compute the correction of DN to linearized DN.

        Alternatively, when electrons = True, rescale these to DN,
        correct the DN, and scale them back to electrons using
        the gain.

        Parameters
        ----------
        img : galsim.Image
            The observed img

        electrons : bool
            Set to True for 'img' being in electrons, with coefficients
            designed for DN. Accordingly, the gain needs to be removed and
            reapplied.

        reversed : bool
            If True, the coefficients are in reversed order, which is the
            order that np.polyval wants them.  One can maybe save a little
            time reversing them once ahead of time.
        """
        if not self.usecrds:
            img.applyNonlinearity(NLfunc=NLfunc)
        else:
            img_arr = img.array
            if electrons:
                img_arr = self.gain * self._evaluate_nl_polynomial(
                    img_arr / self.gain, self.crds_coeffs, reversed
                )
            else:
                img_arr = self._evaluate_nl_polynomial(
                    img_arr, self.crds_coeffs, reversed
                )
            img.array = img_arr
        return img
