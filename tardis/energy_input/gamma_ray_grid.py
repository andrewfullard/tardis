import numpy as np
from astropy.coordinates import cartesian_to_spherical
import re
from nuclear.io.nndc import get_decay_radiation_database, store_decay_radiation
import pandas as pd

from tardis.energy_input.util import (
    solve_quadratic_equation,
    convert_half_life_to_astropy_units,
)
from tardis.montecarlo.montecarlo_numba.numba_config import CLOSE_LINE_THRESHOLD
from tardis.energy_input.energy_source import load_nndc_decay_data


def calculate_distance_radial(gxpacket, r_inner, r_outer):
    """
    Calculates 3D distance to shell from gamma ray position

    Parameters
    ----------
    gxpacket : GXPacket object
    r_inner : dtype float
    r_outer : dtype float

    Returns
    -------
    distance : dtype float

    """
    # determine cartesian location coordinates of gamma-ray object
    x, y, z = gxpacket.location.get_cartesian_coords
    # determine cartesian direction coordinates of gamma-ray object
    x_dir, y_dir, z_dir = gxpacket.direction.get_cartesian_coords
    # solve the quadratic distance equation for the inner and
    # outer shell boundaries
    inner_1, inner_2 = solve_quadratic_equation(
        x, y, z, x_dir, y_dir, z_dir, r_inner
    )
    outer_1, outer_2 = solve_quadratic_equation(
        x, y, z, x_dir, y_dir, z_dir, r_outer
    )
    distances = [inner_1, inner_2, outer_1, outer_2]
    # the correct distance is the shortest positive distance
    distance = min(i for i in distances if i > 0.0)

    return distance


def distance_trace(
    gxpacket,
    inner_radii,
    outer_radii,
    total_opacity,
    ejecta_epoch,
):
    """
    Traces distance traveled by gamma ray and finds distance to
    next interaction and boundary

    Parameters
    ----------
    gxpacket : GXPacket object
    inner_radii : One dimensional Numpy array, dtype float
    outer_radii : One dimensional Numpy array, dtype float
    total_opacity : dtype float
    ejecta_epoch : dtype float

    Returns
    -------
    distance_interaction : dtype float
    distance_boundary : dtype float

    """
    if gxpacket.shell < len(inner_radii):
        distance_boundary = calculate_distance_radial(
            gxpacket,
            inner_radii[gxpacket.shell],
            outer_radii[gxpacket.shell],
        )
    else:
        distance_boundary = 0.0

    distance_interaction = gxpacket.tau / total_opacity / ejecta_epoch
    return distance_interaction, distance_boundary


def move_gamma_ray(gxpacket, distance):
    """
    Moves gamma ray a distance along its direction vector

    Parameters
    ----------
    gxpacket : GXPacket object
    distance : dtype float

    Returns
    -------
    gxpacket : GXPacket object

    """
    x_old, y_old, z_old = gxpacket.location.get_cartesian_coords
    x_dir, y_dir, z_dir = gxpacket.direction.get_cartesian_coords
    # overshoot by CLOSE_LINE_THRESHOLD * distance to shell boundary
    # so that the gamma-ray is comfortably in the next shell
    y_new = y_old + distance * (1 + CLOSE_LINE_THRESHOLD) * y_dir
    z_new = z_old + distance * (1 + CLOSE_LINE_THRESHOLD) * z_dir
    x_new = x_old + distance * (1 + CLOSE_LINE_THRESHOLD) * x_dir

    r, theta, phi = cartesian_to_spherical(x_new, y_new, z_new)
    gxpacket.location.r = r.value
    gxpacket.location.theta = theta.value + 0.5 * np.pi
    gxpacket.location.phi = phi.value
    return gxpacket


def density_sampler(radii, mass_ratio):
    """
    Randomly samples the

    Parameters
    ----------
    radii : GammaRay object
    mass_ratio : dtype float

    Returns
    -------
    radius : dtype float
    index : dtype int

    """
    z = np.random.random()

    mass_ratio_sorted_indices = np.argsort(mass_ratio)
    index = mass_ratio_sorted_indices[
        np.searchsorted(mass_ratio, z, sorter=mass_ratio_sorted_indices)
    ]

    return radii[index], index


def mass_per_shell(radial_grid_size, inner_radii, outer_radii, density_profile):
    """Calculates the distribution of mass in the shells
    based on a density profile

    Parameters
    ----------
    radial_grid_size : int
        Number of radial grid cells
    inner_radii : One-dimensional Numpy Array, dtype float
        Inner radii of shells
    outer_radii : One-dimensional Numpy Array, dtype float
        Outer radii of shells
    density_profile : One-dimensional Numpy Array, dtype float
        Density of shells

    Returns
    -------
    One-dimensional Numpy Array, dtype float
        Normalized array of mass in each shell
    """
    mass = np.zeros(radial_grid_size)

    for i in range(radial_grid_size):
        if i == 0:
            mass[i] = (
                4.0
                / 3.0
                * np.pi
                * density_profile[i]
                * (outer_radii[i] - inner_radii[i]) ** 3.0
            )
        else:
            mass[i] = (
                4.0
                / 3.0
                * np.pi
                * density_profile[i]
                * (outer_radii[i] ** 3.0 - outer_radii[i - 1] ** 3.0)
            )
    return mass


def mass_distribution(
    radial_grid_size, inner_radii, outer_radii, density_profile
):
    """Calculate the mass distribution of the density profile

    Parameters
    ----------
    radial_grid_size : int64
        Number of radial grid cells
    inner_radii : ndarray
        Array of inner radii
    outer_radii : ndarray
        Array of outer radii
    density_profile : ndarray
        Array of density

    Returns
    -------
    ndarray
        Mass cumulative distribution function
    """
    shell_masses = mass_per_shell(
        radial_grid_size, inner_radii, outer_radii, density_profile
    )
    mass_cdf = np.zeros(radial_grid_size)
    mass = 0
    for i in range(radial_grid_size):
        mass += shell_masses[i]
        mass_cdf[i] = mass
    return mass_cdf / np.max(mass_cdf)


def get_shell(radius, outer_radii):
    """Returns the shell index at a given radius

    Parameters
    ----------
    radius : float
        Radius of interest
    outer_radii : One-dimensional Numpy Array, dtype float
        Outer radii of shells

    Returns
    -------
    int
        Shell index corresponding to radius
    """
    shell_inner = np.searchsorted(outer_radii, radius, side="left")

    return shell_inner


def compute_required_packets_per_shell(
    outer_radii,
    inner_radii,
    ejecta_density,
    number_of_shells,
    raw_isotope_abundance,
    number_of_packets,
):
    """Computes the number of packets required per shell
    that sum to the total number of requested packets

    Parameters
    ----------
    outer_radii : ndarray
        Outer radii of shells
    inner_radii : ndarray
        Inner radii of shells
    ejecta_density : ndarray
        Array of densities
    number_of_shells : int64
        Number of radial grid cells
    raw_isotope_abundance : pandas DataFrame
        Abundances of isotopes
    number_of_packets : int64
        Total number of simulation packets

    Returns
    -------
    pandas DataFrame
        Packets required per shell
    pandas DataFrame
        Database of decay radiation
    """
    shell_masses = mass_per_shell(
        number_of_shells, inner_radii, outer_radii, ejecta_density
    )
    shell_masses = shell_masses / np.sum(shell_masses)
    abundance_dict = {}
    for index, row in raw_isotope_abundance.iterrows():
        isotope_string = load_nndc_decay_data(index, force_update=False)
        abundance_dict[isotope_string] = row * shell_masses

    abundance_df = pd.DataFrame.from_dict(abundance_dict)

    decay_rad_db, meta = get_decay_radiation_database()

    activity_df = abundance_df.copy()
    for column in activity_df:
        isotope_meta = meta.loc[column]
        half_life = isotope_meta.loc[
            isotope_meta["key"] == "Parent T1/2 value"
        ]["value"].values[0]
        half_life = convert_half_life_to_astropy_units(half_life)
        atomic_mass = float(re.findall("\d+", column)[0])
        activity_factor = np.log(2) / atomic_mass / half_life
        activity_df[column] = activity_df[column] * activity_factor

    total_activity = activity_df.to_numpy().sum()
    packet_per_shell_df = activity_df.copy()

    for column in packet_per_shell_df:
        packet_per_shell_df[column] = round(
            packet_per_shell_df[column] * number_of_packets / total_activity
        )
        packet_per_shell_df[column] = packet_per_shell_df[column].astype(int)

    return packet_per_shell_df, decay_rad_db
