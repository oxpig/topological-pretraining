from .bace import BACE
from .bbbp import BBBP
from .clintox import ClinTox
from .esol import ESOL
from .freesolv import FreeSolv
from .hiv import HIV
from .lipo import Lipo
from .muv import MUV
from .sider import SIDER
from .tox21 import Tox21
from .toxcast import ToxCast

from pathlib import Path

class MoleculeNet:

    def __init__(self, root: str, compression: bool = True):
        self.root = root
        self.suffix = 'csv.gz' if compression else 'csv'
        self.compression = compression

    @property
    def BACE(self):
        csv = Path(self.root) / f'bace.{self.suffix}'
        return BACE(csv=csv, compression=self.compression)
    
    @property
    def BBBP(self):
        csv = Path(self.root) / f'bbbp.{self.suffix}'
        return BBBP(csv=csv, compression=self.compression)
    
    @property
    def ClinTox(self):
        csv = Path(self.root) / f'clintox.{self.suffix}'
        return ClinTox(csv=csv, compression=self.compression)
    
    @property
    def ESOL(self):
        csv = Path(self.root) / f'esol.{self.suffix}'
        return ESOL(csv=csv, compression=self.compression)
    
    @property
    def FreeSolv(self):
        csv = Path(self.root) / f'freesolv.{self.suffix}'
        return FreeSolv(csv=csv, compression=self.compression)

    @property
    def HIV(self):
        csv = Path(self.root) / f'hiv.{self.suffix}'
        return HIV(csv=csv, compression=self.compression)

    @property
    def Lipo(self):
        csv = Path(self.root) / f'lipo.{self.suffix}'
        return Lipo(csv=csv, compression=self.compression)

    @property
    def MUV(self):
        csv = Path(self.root) / f'muv.{self.suffix}'
        return MUV(csv=csv, compression=self.compression)

    @property
    def SIDER(self):
        csv = Path(self.root) / f'sider.{self.suffix}'
        return SIDER(csv=csv, compression=self.compression)

    @property
    def Tox21(self):
        csv = Path(self.root) / f'tox21.{self.suffix}'
        return Tox21(csv=csv, compression=self.compression)

    @property
    def ToxCast(self):
        csv = Path(self.root) / f'toxcast.{self.suffix}'
        return ToxCast(csv=csv, compression=self.compression)

    