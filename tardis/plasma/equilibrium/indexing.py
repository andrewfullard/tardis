import numpy as np
import pandas as pd


def calculate_lower_ion_level_index(
    level_number_density: pd.DataFrame,
):
    """Calculate index for lower ion levels (ion_number == 0).

    Parameters
    ----------
    lte_level_number_density : pd.DataFrame
        DataFrame containing level number densities with a MultiIndex
        that includes 'ion_number'.

    Returns
    -------
    pd.Series
        Boolean Series indicating rows corresponding to lower ion levels.
    """
    return level_number_density.index.get_level_values("ion_number") == 0


def calculate_upper_ion_population_index(
    ion_number_density: pd.DataFrame,
):
    """Calculate index for upper ion populations (ion_number > 0).

    Parameters
    ----------
    lte_ion_number_density : pd.DataFrame
        DataFrame containing ion number densities with a MultiIndex
        that includes 'ion_number'.

    Returns
    -------
    pd.Series
        Boolean Series indicating rows corresponding to upper ion populations.
    """
    return ion_number_density.index.get_level_values("ion_number") > 0


def calculate_block_ids_from_dataframe(dataframe):
    block_start_id = (
        np.where(np.diff(dataframe.index.get_level_values(0)) != 0.0)[0] + 1
    )
    return np.hstack(([0], block_start_id, [len(dataframe)]))


def calculate_lines_lower_level_index(levels, lines):
    levels_index = pd.Series(
        np.arange(len(levels), dtype=np.int64), index=levels
    )
    lines_index = lines.index.droplevel("level_number_upper")
    return np.array(levels_index.loc[lines_index])


def calculate_lines_upper_level_index(levels, lines):
    levels_index = pd.Series(
        np.arange(len(levels), dtype=np.int64), index=levels
    )
    lines_index = lines.index.droplevel("level_number_lower")
    return np.array(levels_index.loc[lines_index])


def initialize_indices(levels, partition_function):
    indexer = pd.Series(
        np.arange(partition_function.shape[0]),
        index=partition_function.index,
    )
    return indexer.loc[levels.droplevel(2)].values
