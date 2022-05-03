import numpy as np
import pandas as pd
from nuclear.ejecta import Ejecta
from numba import njit
import radioactivedecay as rd

def decay_nuclides(shell_mass, initial_composition, epoch):
    """Decay model

    Parameters
    ----------
    shell_mass : float64
        Mass of the shell in grams
    initial_composition : DataFrame
        Initial ejecta composition
    epoch : float
        Time in days

    Returns
    -------
    DataFrame
        New composition at time epoch
    """    
    decay_model = Ejecta(shell_mass, initial_composition)

    new_fractions = decay_model.decay(epoch)
    return new_fractions

@njit
def sample_mass(masses, inner_radius, outer_radius):
    """Samples location weighted by mass

    Parameters
    ----------
    masses : array
        Shell masses
    inner_radius : array
        Inner radii
    outer_radius : array
        Outer radii

    Returns
    -------
    float
        Sampled radius
    int
        Sampled shell index
    """    
    norm_mass = masses / np.sum(masses)
    cdf = np.cumsum(norm_mass)
    shell = np.searchsorted(cdf, np.random.random())

    z = np.random.random()
    radius = (z * inner_radius[shell] ** 3.0
                + (1.0 - z) * outer_radius[shell] ** 3.0
            ) ** (1.0 / 3.0)

    return radius, shell

@njit
def create_energy_cdf(energy, intensity):
    """Creates a CDF of given intensities

    Parameters
    ----------
    energy :  One-dimensional Numpy Array, dtype float
        Array of energies
    intensity :  One-dimensional Numpy Array, dtype float
        Array of intensities

    Returns
    -------
    One-dimensional Numpy Array, dtype float
        Sorted energy array
    One-dimensional Numpy Array, dtype float
        CDF where each index corresponds to the energy in
        the sorted array
    """
    energy.sort()
    sorted_indices = np.argsort(energy)
    sorted_intensity = intensity[sorted_indices]
    norm_intensity = sorted_intensity / np.sum(sorted_intensity)
    cdf = np.cumsum(norm_intensity)

    return energy, cdf


@njit
def sample_energy_distribution(energy_sorted, cdf):
    """Randomly samples a CDF of energies

    Parameters
    ----------
    energy_sorted : One-dimensional Numpy Array, dtype float
        Sorted energy array
    cdf : One-dimensional Numpy Array, dtype float
        CDF

    Returns
    -------
    float
        Sampled energy
    """
    index = np.searchsorted(cdf, np.random.random())

    return energy_sorted[index]


def setup_input_energy(nuclear_data, source):
    """Sets up energy distribution and CDF for a
    source of decay radiation.

    Parameters
    ----------
    nuclear_data : Pandas dataframe
        Dataframe of nuclear decay properties
    source : str
        Type of decay radiation

    Returns
    -------
    One-dimensional Numpy Array, dtype float
        Sorted energy array
    One-dimensional Numpy Array, dtype float
        CDF where each index corresponds to the energy in
        the sorted array
    """
    intensity = nuclear_data.query("type==" + source)["intensity"].values
    energy = nuclear_data.query("type==" + source)["energy"].values
    energy_sorted, cdf = create_energy_cdf(energy, intensity)

    return energy_sorted, cdf


def intensity_ratio(nuclear_data, source_1, source_2):
    """Determined the ratio of intensities between two
    sources of decay radiation

    Parameters
    ----------
    nuclear_data : pandas.Dataframe
        Dataframe of nuclear decay properties
    source_1 : str
        Type of decay radiation to compare
    source_2 : str
        Type of decay radiation to compare

    Returns
    -------
    float
        Fractional intensity of source_1
    float
        Fractional intensity of source_2
    float
        Number of decay products per decay
    """
    intensity_1 = nuclear_data.query("type==" + source_1)["intensity"].values
    intensity_2 = nuclear_data.query("type==" + source_2)["intensity"].values
    total_intensity = np.sum(intensity_1) + np.sum(intensity_2)
    scale_factor = total_intensity / 100
    return (
        np.sum(intensity_1) / total_intensity,
        np.sum(intensity_2) / total_intensity,
        scale_factor,
    )

def ni56_chain_energy(taus, time_start, time_end, number_ni56, ni56_lines, co56_lines):
    """Calculate the energy from the Ni56 chain

    Parameters
    ----------
    taus : array float64
        Mean half-life for each isotope
    time_start : float
        Start time in days
    time_end : float
        End time in days
    number_ni56 : int
        Number of Ni56 atoms at time_start
    ni56_lines : DataFrame
        Ni56 lines and intensities
    co56_lines : DataFrame
        Co56 lines and intensities

    Returns
    -------
    float
        Total energy from Ni56 decay
    """    
    total_ni56 = -taus["Ni56"] * (np.exp(-time_end / taus["Ni56"]) - np.exp(-time_start / taus["Ni56"]))
    total_co56 = -taus["Co56"] * (np.exp(-time_end / taus["Co56"]) - np.exp(-time_start / taus["Co56"]))

    total_energy = pd.DataFrame()
    
    total_energy["Ni56"] = number_ni56 * (
            (ni56_lines.energy * 1000 * ni56_lines.intensity).sum()
            / taus["Ni56"]
            *
            total_ni56
    )

    total_energy["Co56"] = number_ni56 * (
            (co56_lines.energy * 1000 * co56_lines.intensity).sum()
            / (taus["Ni56"] - taus["Co56"])
            * (total_ni56 - total_co56)
    )

    return total_energy

def ni56_chain_energy_choice(taus, time_start, time_end, number_ni56, ni56_lines, co56_lines, isotope):
    """Calculate the energy from the Ni56 or Co56 chain

    Parameters
    ----------
    taus : array float64
        Mean half-life for each isotope
    time_start : float
        Start time in days
    time_end : float
        End time in days
    number_ni56 : int
        Number of Ni56 atoms at time_start
    ni56_lines : DataFrame
        Ni56 lines and intensities
    co56_lines : DataFrame
        Co56 lines and intensities
    isotope : string
        Isotope chain to calculate energy for

    Returns
    -------
    float
        Total energy from decay chain
    """    
    total_ni56 = -taus["Ni56"] * (np.exp(-time_end / taus["Ni56"]) - np.exp(-time_start / taus["Ni56"]))
    total_co56 = -taus["Co56"] * (np.exp(-time_end / taus["Co56"]) - np.exp(-time_start / taus["Co56"]))

    if isotope == "Ni56":
        total_energy = number_ni56 * (
                (ni56_lines.energy * 1000 * ni56_lines.intensity).sum()
                / taus["Ni56"]
                *
                total_ni56
        )
    else:
        total_energy = number_ni56 * (
                (co56_lines.energy * 1000 * co56_lines.intensity).sum()
                / (taus["Ni56"] - taus["Co56"])
                * (total_ni56 - total_co56)
        )

    return total_energy

def get_all_isotopes(abundances):
    """Get the possible isotopes present over time 
    for a given starting abundance

    Parameters
    ----------
    abundances : DataFrame
        Current isotope abundances

    Returns
    -------
    list
        List of isotope names
    """    
    progenitors = [f"{rd.utils.Z_DICT[i[0]]}-{i[1]}" for i in abundances.T.columns]

    isotopes = set(progenitors)
    check = True

    while check == True:
        progeny = set(isotopes)

        for i in isotopes:
            for p in rd.Nuclide(i).progeny():
                if p != 'SF':
                    progeny.add(p) 
        
        if progeny == isotopes:
            check = False
        else:
            isotopes |= progeny

    isotopes = [i.replace("-", "") for i in isotopes]
    return isotopes