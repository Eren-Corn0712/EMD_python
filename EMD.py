from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def normalize_time(t):
    d = np.diff(t)
    assert np.all(d != 0), "All time domain values needs to be unique"
    return (t - t[0]) / np.min(d)


class EMD:
    def __init__(self, **kwargs,
                 ):
        self.extrema_detection = kwargs.get("extrema_detection", "simple")  # simple, parabol
        self.DTYPE = kwargs.get("DTYPE", np.float64)
        self.MAX_ITERATION = int(kwargs.get("MAX_ITERATION", 1000))

    def find_extrema(self, S, T):
        # Finds indexes of zero-crossings
        if self.extrema_detection == "parabol":
            return self._find_extrema_parabol(T, S)
        elif self.extrema_detection == "simple":
            return self._find_extrema_simple(T, S)
        else:
            raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")

    def _find_extrema_parabol(self, T, S):
        indzer = self._find_zero_crossing(S)
        pass

    def _find_extrema_simple(self, T, S):
        indzer = self._find_zero_crossing(S)
        pass

    def _find_zero_crossing(self, S):
        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1 * S2 < 0)[0]
        if np.any(S == 0):
            indz = np.nonzero(S == 0)[0]

            if np.any(np.diff(indz) == 1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2.0)

            indzer = np.sort(np.append(indzer, indz))

        return indzer

    def __call__(self, S, T=None, max_imf=-1):
        return self.emd(S, T=T, max_imf=max_imf)

    def emd(self, S, T, max_imf):
        if T is not None and len(S) != len(T):
            raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(len(S), len(T)))

        if T is None or self.extrema_detection == "simple":
            T = np.arange(start=0, stop=len(S), dtype=S.dtype)

        T = normalize_time(T)
        self.DTYPE = S.dtype
        N = len(S)

        residue = S.astype(self.DTYPE)  # initial Signnal
        imf = np.zeros(len(S), dtype=self.DTYPE)
        imf_old = np.nan

        # Create arrays
        imfNo = 0
        extNo = -1
        IMF = np.empty((imfNo, N))  # Numpy container for IMF
        finished = False

        while not finished:

            # residuum = Original Signal - IMFs
            residue[:] = S - np.sum(IMF[:imfNo], axis=0)
            imf = residue.copy()
            mean = np.zeros(len(S), dtype=self.DTYPE)

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when |#zero - #ext| <=1

            while True:
                n += 1
                if n >= self.MAX_ITERATION:
                    break

                xt_res = self.find_extrema(T, imf)


if __name__ == "__main__":
    # EMD options
    max_imf = -1
    DTYPE = np.float64

    # Signal options
    N = 400
    tMin, tMax = 0, 2 * np.pi
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)

    S = np.sin(20 * T * (1 + 0.2 * T)) + T ** 2 + np.sin(13 * T)
    S = S.astype(DTYPE)
    print("Input S.dtype: " + str(S.dtype))

    # Prepare and run EMD
    emd = EMD()
    emd.FIXE_H = 5
    emd.nbsym = 2
    emd.spline_kind = "cubic"
    emd.DTYPE = DTYPE

    imfs = emd.emd(S, T, max_imf)
    imfNo = imfs.shape[0]

    # Plot results
    c = 1
    r = np.ceil((imfNo + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S, "r")
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r, c, num + 2)
        plt.plot(T, imfs[num], "g")
        plt.xlim((tMin, tMax))
        plt.ylabel("Imf " + str(num + 1))

    plt.tight_layout()
    plt.show()
