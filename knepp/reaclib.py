################################################################################

import re
import numpy as np

# from .util import tqdm
from tqdm import tqdm

################################################################################

_elnames = ("n h he li be b c n o f ne na mg al si p s cl ar k "
            "ca sc ti v cr mn fe co ni cu zn ga ge as se br kr rb sr y "
            "zr nb mo tc ru rh pd ag cd in sn sb te i xe cs ba la ce pr "
            "nd pm sm eu gd tb dy ho er tm yb lu hf ta w re os ir pt au "

            "hg tl pb bi po at rn fr ra ac th pa u np pu am cm bk cf es "
            "fm md no lr rf db sg bh hs mt ds rg cn nh fl mc lv ts og").split()

_fmt = r"([-+ ]\d\.\d{6}[eE][-+]\d{2})"
_coeff_fmt = re.compile(4*_fmt + " *\n" + 3*_fmt + " *\n")

_header_fmt = re.compile(
    r"^ {5}" # Five spaces
    r"(.{5})(.{5})(.{5})(.{5})(.{5})(.{5}) {8}"   # 6 nuclides, each 5 characters
    r"(.{4})" # Set label (4 characters)
    r"([nrsw ])" # Resonance/weak flag (1 character)
    r"([v ]) {3}" # Reverse flag (1 character, optional)
    r"([-+ ]\d\.\d{5}[eE][-+]\d{2})" # Q value in scientific notation
    r" *$" # Ten trailing spaces
)


################################################################################

class Nuclide:
    def __init__(self, name: str):
        self.name = name.strip().lower()
        if self.name == "h1":
            self.name = "p"
        elif self.name == "h2":
            self.name = "d"
        elif self.name == "h3":
            self.name = "t"

        if self.name == 'n':
            self.A = 1
            self.Z = 0
            self.el = 'n'
        elif self.name == 'p':
            self.A = 1
            self.Z = 1
            self.el = 'H'
        elif self.name == 'd':
            self.A = 2
            self.Z = 1
            self.el = 'H'
        elif self.name == 't':
            self.A = 3
            self.Z = 1
            self.el = 'H'
        else:
            try:
                self.el = "".join([dd for dd in self.name if dd.isalpha()])
                self.A = int("".join([dd for dd in self.name if dd.isdigit()]))
                self.Z = _elnames.index(self.el)
            except:
                raise ValueError(f"Error: {self.name} is not a valid nuclide name.")

        self.N = self.A - self.Z

    def __eq__(self, other):
        return (self.A == other.A) and (self.Z == other.Z)

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

neutron = Nuclide("n")
proton = Nuclide("p")
deuterium = Nuclide("d")
tritium = Nuclide("t")
alpha = Nuclide("he4")


################################################################################

class Reaction:
    def __init__(
        self,
        chapter: int,
        header: str,
        coeffs: str,
    ):
        self.chapter = chapter
        self.header = header.strip()
        nucs, self.label, self.rwflag, self.revflag, self.qval = self._read_header(header)
        self.reacs, self.prods = self._devide_nuclides(nucs, chapter)
        self.coeffs = self._read_coeffs(coeffs)
        self.type = self._get_type()

    @staticmethod
    def _read_header(header: str) -> tuple[
        tuple[str, ...],
        str, str, str, float
    ]:
        """
        given the reaclib reaction header string returns:
            - tuple containing the nuclides
            - the reaction label
            - the resonance/weak flag
            - the reverse flag
            - the Q value
        """
        match = _header_fmt.match(header)
        if match is None:
            breakpoint()
            raise ValueError(f"Invalid reaction header: {header}")
        *nucs, labl, wflg, rflg = match.groups()[:-1]
        qval = float(match.group(10))
        return tuple(nucs), labl, wflg, rflg, qval

    @staticmethod
    def _devide_nuclides(
        nucs: tuple[str, ...],
        chapter: int
    ) -> tuple[tuple[Nuclide,...], tuple[Nuclide, ...]]:
        assert len(nucs) == 6
        match chapter, nucs:
            case 1, (r1, p1, *_):
                reacs, prods = (r1, ), (p1, )
            case 2, (r1, p1, p2, *_):
                reacs, prods = (r1, ), (p1, p2)
            case 3, (r1, p1, p2, p3, *_):
                reacs, prods = (r1, ), (p1, p2, p3)
            case 4, (r1, r2, p1, *_):
                reacs, prods = (r1, r2), (p1, )
            case 5, (r1, r2, p1, p2, *_):
                reacs, prods = (r1, r2), (p1, p2)
            case 6, (r1, r2, p1, p2, p3, _):
                reacs, prods = (r1, r2), (p1, p2, p3)
            case 7, (r1, r2, p1, p2, p3, p4):
                reacs, prods = (r1, r2), (p1, p2, p3, p4)
            case 8, (r1, r2, r3, p1, *_):
                reacs, prods = (r1, r2, r3), (p1, )
            case 9, (r1, r2, r3, p1, p2, _):
                reacs, prods = (r1, r2, r3), (p1, p2)
            case 10, (r1, r2, r3, r4, p1, p2):
                reacs, prods = (r1, r2, r3, r4), (p1, p2)
            case 11, (r1, p1, p2, p3, p4, _):
                reacs, prods = (r1, ), (p1, p2, p3, p4)
            case _:
                raise ValueError(f"Invalid header config {nucs} for chapter {chapter}")

        reacs = tuple(map(Nuclide, reacs))
        prods = tuple(map(Nuclide, prods))
        return reacs, prods

    @staticmethod
    def _read_coeffs(coeffs: str) -> np.ndarray:
        match = _coeff_fmt.match(coeffs)
        if match is None:
            breakpoint()
            raise ValueError(f"Invalid reaction coefficients: {coeffs}")
        return np.array(match.groups(), dtype=float)


    def _get_type(self) -> str:
        if self.label in ['ec', 'bec']:
            return 'ec'

        if self.rwflag == 'w':
            if self.chapter == 1:
                if self.reacs[0].Z < self.prods[0].Z:
                    return 'b-'
                else:
                    return 'b+'
            if self.chapter in (2, 3, 11):
                if neutron in self.reacs and proton in self.reacs:
                    return "bnp"
                proj = sorted(self.prods, key=lambda r: r.A)[0].name
                if proj == 'he4': proj = 'a'
                return f"b{proj}"
            return 'otherweak'

        if self.is_decay:
            if alpha in self.prods:
                return 'ad'
            if neutron in self.prods:
                return 'nd'
            if proton in self.prods:
                return 'pd'
            return 'otherdecay'

        if self.chapter == 2:
            if neutron in self.prods:
                return 'gn'
            if proton in self.prods:
                return 'gp'
            if alpha in self.prods:
                return 'ga'
            return "otherphotodis"

        if self.chapter == 4:
            if neutron in self.reacs:
                return 'ng'
            if proton in self.reacs:
                return 'pg'
            if alpha in self.reacs:
                return 'ag'

        if self.chapter == 5:
            proj = sorted(self.reacs, key=lambda r: r.A)[0].name
            ejec = sorted(self.prods, key=lambda r: r.A)[0].name
            if proj == 'he4': proj = 'a'
            if ejec == 'he4': ejec = 'a'
            if len(type := f"{proj}{ejec}") == 2:
                return type

        return 'other'

    def eval_at_T(self, T: float) -> float:
        a = self.coeffs
        return np.exp(a[0]
                      + sum(a[i] * T ** ((2*i -5)/3) for i in range(1, 6))
                      + a[6]*np.log(T))

    @property
    def is_decay(self) -> bool:
        return self.chapter in (1, 2, 3, 11) and np.all(self.coeffs[1:] == 0)

    @property
    def is_weak(self) -> bool:
        return self.rwflag == 'w'

    @property
    def is_reverse(self) -> bool:
        return self.revflag == 'v'

    def __eq__(self, other):
        return (self.reacs == other.reacs
               and self.prods == other.prods
               and self.coeffs == other.coeffs)

    def __repr__(self):
        return f"<Reaction {"+".join(map(str, self.reacs))} -> {"+".join(map(str, self.prods))}>"

    def __str__(self):
        return self.__repr__()

################################################################################

class Reaclib:
    _max_n_reacs = 1000000
    def __init__(self, path: str):
        self.reactions = []
        chapter = 1
        with open(path, 'r') as f:
            n_lines = len(f.readlines())

        with open(path, 'r') as f:
            # for _ in tqdm(range(n_lines),
            #               ncols=0,
            #               total=n_lines,
            #               desc="Reading Reaclib",
            #               unit="lines",
            #     ):
            while True:
                line = f.readline()
                if not line:
                    break
                if line[0].isnumeric():
                    chapter = int(line)
                    continue
                if not line.strip():
                    continue
                header = line
                coeffs = f.readline() + f.readline()
                self.reactions.append(Reaction(chapter, header, coeffs))


    def __iter__(self):
        return iter(self.reactions)

    def __getitem__(self, key):
        return self.reactions[key]

    def __len__(self):
        return len(self.reactions)
