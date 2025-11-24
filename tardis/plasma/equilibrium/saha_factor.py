import warnings

import numpy as np
import pandas as pd
from scipy import interpolate

from tardis.plasma.equilibrium.indexing import (
    calculate_block_ids_from_dataframe,
)


def calculate_saha_factor_lte(
    g_electron, beta_radiation, partition_function, ionization_data
):
    saha_factors = np.empty(
        (
            partition_function.shape[0]
            - partition_function.index.get_level_values(0).unique().size,
            partition_function.shape[1],
        )
    )

    block_ids = calculate_block_ids_from_dataframe(partition_function)

    for i, start_id in enumerate(block_ids[:-1]):
        end_id = block_ids[i + 1]
        current_block = partition_function.values[start_id:end_id]
        current_saha_factors = current_block[1:] / current_block[:-1]
        saha_factors[start_id - i : end_id - i - 1] = current_saha_factors

    broadcast_ionization_energy = ionization_data.reindex(
        partition_function.index
    ).dropna()
    saha_factor_index = broadcast_ionization_energy.index
    broadcast_ionization_energy = broadcast_ionization_energy.values

    saha_factor_coefficient = (
        2
        * g_electron
        * np.exp(np.outer(broadcast_ionization_energy, -beta_radiation))
    )

    return pd.DataFrame(
        saha_factors * saha_factor_coefficient, index=saha_factor_index
    )


def _set_chi_0(chi_0_species, ionization_data):
    if chi_0_species == (20, 2):
        chi_0 = 1.9020591570241798e-11
    else:
        chi_0 = ionization_data.loc[chi_0_species]
    return chi_0


def calculate_radiation_field_correction(
    dilution_factor,
    ionization_data,
    beta_rad,
    electron_temperature,
    radiation_temperature,
    beta_electron,
    chi_0_species,
):
    chi_0 = _set_chi_0(chi_0_species, ionization_data)
    departure_coefficient = (
        1.0 / dilution_factor
    )  # see Equation 13 and explanations on page 451 lower right in ML 93
    radiation_field_correction = -np.ones((len(ionization_data), len(beta_rad)))
    less_than_chi_0 = (ionization_data < chi_0).values
    factor_a = electron_temperature / (
        departure_coefficient * dilution_factor * radiation_temperature
    )
    radiation_field_correction[~less_than_chi_0] = factor_a * np.exp(
        np.outer(
            ionization_data.values[~less_than_chi_0],
            beta_rad - beta_electron,
        )
    )
    radiation_field_correction[less_than_chi_0] = 1 - np.exp(
        np.outer(ionization_data.values[less_than_chi_0], beta_rad)
        - beta_rad * chi_0
    )
    radiation_field_correction[less_than_chi_0] += factor_a * np.exp(
        np.outer(ionization_data.values[less_than_chi_0], beta_rad)
        - chi_0 * beta_electron
    )

    radiation_correction_df = pd.DataFrame(
        radiation_field_correction,
        columns=np.arange(len(radiation_temperature)),
        index=ionization_data.index,
    )
    return radiation_correction_df


def get_zeta_values(zeta_data, ion_index, t_rad):
    zeta_t_rad = zeta_data.columns.values.astype(np.float64)
    zeta_values = zeta_data.loc[ion_index].values.astype(np.float64)
    zeta = interpolate.interp1d(
        zeta_t_rad, zeta_values, bounds_error=False, fill_value=np.nan
    )(t_rad)
    zeta = zeta.astype(float)

    if np.any(np.isnan(zeta)):
        warnings.warn(
            f"t_rads outside of zeta factor interpolation"
            f" zeta_min={zeta_data.columns.values.min():.2f} zeta_max={zeta_data.columns.values.max():.2f} "
            f"- replacing with zeta = 1.0"
        )
        zeta[np.isnan(zeta)] = 1.0

    return zeta


def calculate_phi_saha_nebular(
    radiative_temperature,
    dilution_factor,
    zeta_data,
    electron_temperature,
    radiation_field_correction,
    g_electron,
    beta_rad,
    partition_function,
    ionization_data,
):
    phi_lte = calculate_saha_factor_lte(
        g_electron, beta_rad, partition_function, ionization_data
    )
    zeta = get_zeta_values(zeta_data, phi_lte.index, radiative_temperature)
    phis = (
        phi_lte
        * dilution_factor
        * ((zeta * radiation_field_correction) + dilution_factor * (1 - zeta))
        * (electron_temperature / radiative_temperature) ** 0.5
    )
    return phis
