################################################################################
from collections.abc import Sequence
import numpy as np
from h5py import File
from scipy.sparse import csc_array, coo_matrix, sparray
from numpy.typing import NDArray
from tqdm import tqdm
from multiprocessing import Pool

################################################################################


class Composition(Sequence):
    N: NDArray[np.int64]
    Z: NDArray[np.int64]
    A: NDArray[np.int64]
    num_isotopes: int
    mass_shells: NDArray[np.int64]
    times: NDArray[np.float64]
    shape: tuple[int, int]
    file_path: str
    verbose: bool
    n_procs: int = 1

    def __init__(self, filepath: str):
        self.file_path = filepath
        self.verbose = False

        with File(self.file_path, "r") as hf:
            self._Z = np.array(hf["Z"], dtype=int)
            self._A = np.array(hf["A"], dtype=int)
            self.mass_shells = np.array(
                [int(shell[6:]) for shell in hf.keys() if shell.startswith("shell_")]
            )

            sh = "shell_0001"
            if 1 not in self.mass_shells:
                raise ValueError(
                    f"Mass shell 1 not found in {self.file_path}. Available mass shells: {self.mass_shells}"
                )
            self.times = np.array(
                [
                    hf[f"{sh}/{tstr}"].attrs["time"]
                    for tstr in hf[sh]
                    if tstr.startswith("it_")
                ]
            )

        self._N = self._A - self._Z
        self.Z = np.arange(max(self._Z) + 1)
        self.N = np.arange(max(self._N) + 1)
        self.A = np.arange(max(self._A) + 1)
        self.num_isotopes = len(self._Z)

        self.shape = (
            len(self.times),
            len(self.mass_shells),
            max(self.N) + 1,
            max(self.Z) + 1,
        )

    def __getitem__(self, idx) -> NDArray[np.float64] | np.float64:
        # make sure idx is a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        # fill in missing indices
        idt, ids, N, Z = (*idx, slice(None), slice(None), slice(None))[:4]

        # convert slices to indices
        itime = np.arange(len(self.times))[idt]
        ishell = np.arange(len(self.mass_shells))[ids]
        #  make sure indices are arrays
        itime = np.atleast_1d(itime)
        ishell = np.atleast_1d(ishell)

        tqdm_kw = dict(
            disable=not self.verbose, leave=False, ncols=0, desc="Loading abundances"
        )
        if self.n_procs <= 1:
            if len(itime > 1):
                itime = tqdm(itime, total=len(itime), **tqdm_kw)
            elif len(ishell > 1):
                ishell = tqdm(ishell, total=len(ishell), **tqdm_kw)

            Y = [[self._get(it, ish, N, Z) for ish in ishell] for it in itime]
        else:
            with Pool(self.n_procs) as p:

                def get(arg):
                    return self._get(*arg)

                Y = list(
                    tqdm(
                        p.imap(
                            get, [(it, ish, N, Z) for it in itime for ish in ishell]
                        ),
                        total=len(itime) * len(ishell),
                        **tqdm_kw,
                    )
                )
                Y = np.array(Y).reshape(len(itime), len(ishell), *Y[0].shape)
        return np.squeeze(Y)

    def _get(
        self,
        it: int,
        ish: int,
        N: NDArray[np.int64] | int | slice,
        Z: NDArray[np.int64] | int | slice,
    ) -> NDArray[np.float64]:
        Y = self.get_sparse_NZ(it, ish)
        Y = Y[N, Z]
        if isinstance(Y, csc_array):
            Y = Y.toarray()
        return Y

    def get_sparse_NZ(self, it: int, ish: int) -> csc_array:
        sh = f"shell_{ish+1:04d}"
        itstr = f"it_{it+1:06d}"
        with File(self.file_path, "r") as hf:
            _Y = np.array(hf[f"{sh}/{itstr}/Y"])
            _iAZ = np.array(hf[f"{sh}/{itstr}/iAZ"]) - 1  # fortran to python idx
        _Z = self._Z[_iAZ]
        _N = self._N[_iAZ]
        return csc_array((_Y, (_N, _Z)), shape=self.shape[2:])

    def get_sparse_AZ(self, it: int, ish: int) -> csc_array:
        sh = f"shell_{ish+1:04d}"
        itstr = f"it_{it+1:06d}"
        with File(self.file_path, "r") as hf:
            _Y = np.array(hf[f"{sh}/{itstr}/Y"])
            _iAZ = np.array(hf[f"{sh}/{itstr}/iAZ"]) - 1  # fortran to python idx
        _Z = self._Z[_iAZ]
        _A = self._A[_iAZ]
        return csc_array((_Y, (_A, _Z)), shape=(self.A.max() + 1, self.Z.max() + 1))

    def __len__(self):
        return len(self.times)

    def __repr__(self):
        return f"Composition({self.file_path})"

    def __str__(self):
        return f"Composition({self.file_path})"

    def __iter__(self):
        for it in range(len(self.times)):
            yield self[it]

    @property
    def old_format(self):
        return OldFormat(self)

    @property
    def AZ(self):
        return AZ(self)

    def spec_abundance(
        self, A: int, Z: int, reload: bool = False
    ) -> NDArray[np.float64]:

        dsetname = f"specific_Y/A{A:03d}_Z{Z:03d}"
        with File(self.file_path, "r") as hf:
            if dsetname in hf and not reload:
                return np.array(hf[dsetname])
        yy = self._spec_abundance(A, Z)
        with File(self.file_path, "a") as hf:
            dset = hf.require_dataset(dsetname, shape=yy.shape, dtype=yy.dtype)
            dset[...] = yy
        return yy

    def _spec_abundance(self, A: int, Z: int) -> NDArray[np.float64]:
        i_AZ = np.where((self._A == A) & (self._Z == Z))[0]
        yy = np.zeros((len(self.times), len(self.mass_shells)))
        with File(self.file_path, "r") as hf:
            for it in tqdm(
                range(len(self.times)),
                desc=f"Loading Y(A={A}, Z={Z})",
                ncols=0,
                leave=False,
                disable=not self.verbose,
            ):
                for ish in self.mass_shells:
                    dset_name = f"shell_{ish:04d}/it_{it+1:06d}/iAZ"
                    if dset_name not in hf:
                        continue
                    _iAZ = np.array(hf[dset_name]) - 1
                    if np.any(msk := _iAZ == i_AZ):
                        yy[it, ish - 1] = hf[f"shell_{ish:04d}/it_{it+1:06d}/Y"][msk]
        return yy


class OldFormat(Sequence):
    def __init__(self, comp):
        self.comp = comp

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        idt, ids, iAZ = (*idx, slice(None), slice(None))[:3]
        _N = self.comp._N[iAZ]
        _Z = self.comp._Z[iAZ]
        return self.comp.__getitem__((idt, ids, _N, _Z))

    def __len__(self):
        return len(self.comp)

    def __repr__(self):
        return f"OldFormat({self.comp})"

    def __str__(self):
        return f"OldFormat({self.comp})"

    def __iter__(self):
        return iter(self.comp)


class AZ(Composition):

    def __init__(self, comp):
        self.file_path = comp.file_path
        self.N = comp.N
        self.Z = comp.Z
        self.A = comp.A
        self.verbose = comp.verbose
        self.num_isotopes = comp.num_isotopes
        self.mass_shells = comp.mass_shells
        self._Z = comp._Z
        self._N = comp._N
        self._A = comp._A
        self.times = comp.times
        self.shape = (
            len(self.times),
            len(self.mass_shells),
            max(self.A) + 1,
            max(self.Z) + 1,
        )

    def _get(
        self,
        it: int,
        ish: int,
        A: NDArray[np.int64] | int | slice,
        Z: NDArray[np.int64] | int | slice,
    ):
        Y = self.get_sparse_AZ(it, ish)
        if A is None:
            A = slice(None)
        if Z is None:
            Z = slice(None)
        Y = Y[A, Z]
        if isinstance(Y, csc_array):
            Y = Y.toarray()
        return Y


def init_new_comp_file(
    path: str,
    A: NDArray[np.int64],
    Z: NDArray[np.int64],
    n_shells: int,
):
    with File(path, "w") as hf:
        hf.create_dataset("Z", data=Z)
        hf.create_dataset("A", data=A)
        for ish in range(n_shells):
            shgrp = hf.create_group(f"shell_{ish+1:04d}")


def write_composition(
    sparse_data: sparray,
    file_path: str,
    ish: int,
    it: int,
    time: float,
):

    "Write a sparse composition dataset given a csc_array"
    with File(file_path, "r") as hf:
        As = hf["A"][()]
        Zs = hf["Z"][()]

    # ycoo = sparse_data.tocoo() # this does not work for some reason
    ycoo = coo_matrix(sparse_data)
    A = ycoo.row
    Z = ycoo.col
    iAZ = np.array(
        [np.where((As == aa) & (Zs == zz))[0][0] + 1 for aa, zz in zip(A, Z)]
    )
    yy = ycoo.data

    grp_path = f"shell_{ish+1:04d}/it_{it+1:06d}"
    with File(file_path, "a") as hf:
        grp = hf.require_group(grp_path)
        grp.attrs["time"] = time
        grp.create_dataset("Y", data=yy)
        grp.create_dataset("iAZ", data=iAZ)
