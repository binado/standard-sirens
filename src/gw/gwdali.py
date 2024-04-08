from dataclasses import dataclass
from enum import Enum
from .detector import DetectorCoordinates


@dataclass
class GWDALIDetector:
    name: str
    lat: float
    lon: float
    rot: float
    shape: float

    def to_dict(self):
        return dict(name=self.name, lat=self.lat, lon=self.lon, rot=self.rot, shape=self.shape)


class GWDALIDetectors(Enum):
    LIGO_HANFORD = GWDALIDetector("aLIGO", *DetectorCoordinates.LIGO_HANFORD.value)
    LIGO_LIVINGSTON = GWDALIDetector("aLIGO", *DetectorCoordinates.LIGO_LIVINGSTON.value)
    VIRGO = GWDALIDetector("aVirgo", *DetectorCoordinates.VIRGO.value)
    KAGRA = GWDALIDetector("KAGRA", *DetectorCoordinates.KAGRA.value)
