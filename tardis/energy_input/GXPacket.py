from enum import IntEnum
import numpy as np
from numba import int64, float64
from numba.experimental import jitclass


class GXPacketStatus(IntEnum):
    BETA_DECAY = -1
    COMPTON_SCATTER = 0
    PHOTOABSORPTION = 1
    PAIR_CREATION = 2
    IN_PROCESS = 3
    END = 4


gxpacket_spec = [
    ("location_r", float64),
    ("location_theta", float64),
    ("location_phi", float64),
    ("direction_r", int64),
    ("direction_theta", float64),
    ("direction_phi", float64),
    ("energy", float64),
    ("nu", float64),
    ("status", int64),
    ("shell", int64),
    ("time_created", float64),
    ("time_current", float64),
    ("tau", float64),
    ("activity", float64),
]


@jitclass(gxpacket_spec)
class GXPacket(object):
    """Indivisible gamma-ray packet

    Parameters
    ----------
    object : _type_
        _description_
    """

    def __init__(
        self,
        location_r,
        location_theta,
        location_phi,
        direction_theta,
        direction_phi,
        energy,
        nu,
        status,
        shell,
        activity,
    ):
        self.location_r = location_r
        self.location_theta = location_theta
        self.location_phi = location_phi
        self.direction_r = 1
        self.direction_theta = direction_theta
        self.direction_phi = direction_phi
        self.energy = energy
        self.nu = nu
        self.status = status
        self.shell = shell
        self.time_created = 0
        self.time_current = 0
        # TODO: rename to tau_event
        self.tau = -np.log(np.random.random())
        self.activity = 0

    def get_location_cartesian_coords(self):
        x = (
            self.location_r
            * np.cos(self.location_phi)
            * np.sin(self.location_theta)
        )
        y = (
            self.location_r
            * np.sin(self.location_phi)
            * np.sin(self.location_theta)
        )
        z = self.location_r * np.cos(self.location_theta)
        return x, y, z

    def get_direction_cartesian_coords(self):
        x = np.cos(self.direction_phi) * np.sin(self.direction_theta)
        y = np.sin(self.direction_phi) * np.sin(self.direction_theta)
        z = np.cos(self.direction_theta)
        return x, y, z

    def get_direction_vector(self):
        return np.array(
            (self.direction_r, self.direction_theta, self.direction_phi)
        )