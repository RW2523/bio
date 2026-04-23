"""Input encoding modules for converting continuous sensor windows into spike-like signals."""

from models.encoders.identity import IdentityEncoder
from models.encoders.zcsf import ZCSFEncoder

__all__ = ["IdentityEncoder", "ZCSFEncoder"]
