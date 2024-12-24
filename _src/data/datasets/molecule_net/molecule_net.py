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
        return BACE(root=self.root, compression=self.compression)
    
    @property
    def BBBP(self):
        return BBBP(root=self.root, compression=self.compression)
    
    @property
    def ClinTox(self):
        return ClinTox(root=self.root, compression=self.compression)
    
    @property
    def ESOL(self):
        return ESOL(root=self.root, compression=self.compression)
    
    @property
    def FreeSolv(self):
        return FreeSolv(root=self.root, compression=self.compression)

    @property
    def HIV(self):
        return HIV(root=self.root, compression=self.compression)

    @property
    def Lipo(self):
        return Lipo(root=self.root, compression=self.compression)
    
    @property
    def MUV(self):
        return MUV(root=self.root, compression=self.compression)

    @property
    def SIDER(self):
        return SIDER(root=self.root, compression=self.compression)
    
    @property
    def Tox21(self):
        return Tox21(root=self.root, compression=self.compression)

    @property
    def ToxCast(self):
        return ToxCast(root=self.root, compression=self.compression)

    