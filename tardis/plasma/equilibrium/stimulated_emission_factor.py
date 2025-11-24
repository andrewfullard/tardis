import numpy as np


def get_g_lower(g, lines_lower_level_index):
    g_lower = np.array(g.iloc[lines_lower_level_index], dtype=np.float64)
    _g_lower = g_lower[np.newaxis].T
    return _g_lower


def get_g_upper(g, lines_upper_level_index):
    g_upper = np.array(g.iloc[lines_upper_level_index], dtype=np.float64)
    _g_upper = g_upper[np.newaxis].T
    return _g_upper


def get_metastable_upper(metastability, lines_upper_level_index):
    _meta_stable_upper = metastability.values[lines_upper_level_index][
        np.newaxis
    ].T
    return _meta_stable_upper


def calculate_stimulated_emission_factor(
    g,
    level_number_density,
    lines_lower_level_index,
    lines_upper_level_index,
    metastability,
):
    n_lower = level_number_density.values.take(
        lines_lower_level_index, axis=0, mode="raise"
    )
    n_upper = level_number_density.values.take(
        lines_upper_level_index, axis=0, mode="raise"
    )
    g_lower = get_g_lower(g, lines_lower_level_index)
    g_upper = get_g_upper(g, lines_upper_level_index)
    meta_stable_upper = get_metastable_upper(
        metastability, lines_upper_level_index
    )

    # In theory the factor should be 1 for n_lower = 0, but in practice the opacity is reduced to 0 anyway
    stimulated_emission_factor = np.zeros(n_lower.shape, dtype=np.float64)

    n_lower_zero_mask = n_lower == 0.0
    stimulated_emission_factor[~n_lower_zero_mask] = 1 - (
        (g_lower * n_upper)[~n_lower_zero_mask]
        / (g_upper * n_lower)[~n_lower_zero_mask]
    )

    # the following line probably can be removed as well
    stimulated_emission_factor[np.isneginf(stimulated_emission_factor)] = 0.0
    stimulated_emission_factor[
        meta_stable_upper & (stimulated_emission_factor < 0)
    ] = 0.0
    return stimulated_emission_factor
