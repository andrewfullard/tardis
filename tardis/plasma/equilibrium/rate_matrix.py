import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from tardis.plasma.equilibrium.rates.collisional_ionization_rates import (
    CollisionalIonizationRateSolver,
)
from tardis.plasma.equilibrium.rates.photoionization_rates import (
    AnalyticPhotoionizationRateSolver,
    EstimatedPhotoionizationRateSolver,
)


class RateMatrix:
    def __init__(
        self,
        rate_solvers: list,
        levels: pd.DataFrame,
    ):
        """Constructs the rate matrix from an arbitrary number of rate solvers.

        Parameters
        ----------
        rate_solvers : list
            List of rate solver objects.
        levels : pd.DataFrame
            DataFrame of energy levels.
        """
        self.rate_solvers = rate_solvers
        self.levels = levels

    def solve(
        self,
        radiation_field,
        thermal_electron_energy_distribution,
        beta_sobolevs=None,
    ):
        """Construct the compiled rate matrix dataframe.

        Parameters
        ----------
        radiation_field : RadiationField
            Radiation field containing radiative temperature.
        thermal_electron_energy_distribution : ThermalElectronEnergyDistribution
            Distribution of electrons in the plasma, containing electron energies,
            temperatures and number densities.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rate matrices indexed by atomic number and ion number,
            with each column being a cell.
        """
        required_arg = {
            "radiative": [
                radiation_field,
            ],
            "electron": [
                thermal_electron_energy_distribution.temperature,
            ],
            "scaled_radiative": [radiation_field, beta_sobolevs],
        }

        rates_df_list = [
            solver.solve(*required_arg[arg])
            for solver, arg in self.rate_solvers
        ]
        # Extract all indexes
        all_indexes = set()
        for df in rates_df_list:
            all_indexes.update(df.index)

        # Create a union of all indexes
        all_indexes = sorted(all_indexes)

        # Reindex each dataframe to ensure consistent indices
        rates_df_list = [
            df.reindex(all_indexes, fill_value=0) for df in rates_df_list
        ]

        # Multiply rates by electron number density where appropriate
        rates_df_list = [
            rates_df * thermal_electron_energy_distribution.number_density
            if solver_arg_tuple[1] == "electron"
            else rates_df
            for solver_arg_tuple, rates_df in zip(
                self.rate_solvers, rates_df_list
            )
        ]

        rates_df = sum(rates_df_list)

        grouped_rates_df = rates_df.groupby(
            level=("atomic_number", "ion_number")
        )

        rate_matrices = pd.DataFrame(
            index=grouped_rates_df.groups.keys(), columns=rates_df.columns
        )

        for species_id, rates in grouped_rates_df:
            number_of_levels = self.levels.energy.loc[species_id].count()
            for shell in range(len(rates.columns)):
                matrix = coo_matrix(
                    (
                        rates[shell],
                        (
                            rates.index.get_level_values(
                                "level_number_destination"
                            ),
                            rates.index.get_level_values("level_number_source"),
                        ),
                    ),
                    shape=(number_of_levels, number_of_levels),
                )
                matrix_array = matrix.toarray()
                np.fill_diagonal(matrix_array, -np.sum(matrix_array, axis=0))
                matrix_array[0, :] = 1
                rate_matrices.loc[species_id, shell] = matrix_array

        rate_matrices.index.names = ["atomic_number", "ion_number"]

        return rate_matrices


class IonRateMatrix:
    def __init__(
        self,
        radiative_ionization_rate_solver: AnalyticPhotoionizationRateSolver
        | EstimatedPhotoionizationRateSolver,
        collisional_ionization_rate_solver: CollisionalIonizationRateSolver,
    ):
        """Constructs the ionization rate matrix from radiative and collisional
        ionization rate solvers.

        Parameters
        ----------
        radiative_ionization_rate_solver : AnalyticPhotoionizationRateSolver | EstimatedPhotoionizationRateSolver
            Solver for radiative ionization and recombination rates.
        collisional_ionization_rate_solver : CollisionalIonizationRateSolver
            Solver for collisional ionization and recombination rates.
        """
        self.radiative_ionization_rate_solver = radiative_ionization_rate_solver
        self.collisional_ionization_rate_solver = (
            collisional_ionization_rate_solver
        )

    def __calculate_total_grouped_rates(self, rates_df):
        """Helper function to calculate the total rates from the
        photoionization and recombination rates.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame of rates indexed by atomic number and ion number,
            with each column being a cell.

        Returns
        -------
        pd.DataFrame
            A DataFrame of grouped total rates indexed by atomic number and ion number,
            with each column being a cell.
        """
        return (
            rates_df.groupby(
                level=(
                    "atomic_number",
                    "ion_number",
                    "ion_number_source",
                    "ion_number_destination",
                )
            )
            .sum()
            .groupby(level=("atomic_number"))
        )

    def __construct_rate_matrix(self, rate, cell, ion_states):
        """Construct a sparse rate matrix from the rates.

        Parameters
        ----------
        rate : pd.DataFrame
            Rate DataFrame indexed by atomic number and ion number
        cell : int
            Cell index
        ion_states : int
            Number of ion states for the atomic number

        Returns
        -------
        coo_matrix
            A sparse matrix representing the ionization rate for the given cell.
        """
        return coo_matrix(
            (
                rate[cell],
                (
                    rate.index.get_level_values("ion_number_destination"),
                    rate.index.get_level_values("ion_number_source"),
                ),
            ),
            shape=(ion_states, ion_states),
        )

    def __build_rate_matrices(
        self,
        photoion_rates_df,
        recomb_rates_df,
        collisional_ionization_rates_df,
        collision_recombination_rates_df,
        charge_conservation=False,
    ):
        """Build rate matrices from photoionization, recombination, and collisional rates.

        Parameters
        ----------
        photoion_rates_df : pd.DataFrame
            Photoionization rates DataFrame.
        recomb_rates_df : pd.DataFrame
            Recombination rates DataFrame.
        collisional_ionization_rates_df : pd.DataFrame
            Collisional ionization rates DataFrame.
        collision_recombination_rates_df : pd.DataFrame
            Collisional recombination rates DataFrame.
        charge_conservation : bool, optional
            Whether to include a charge conservation row in the rate matrix.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rate matrices indexed by atomic number,
            with each column being a cell. Each entry is a numpy array.
        """
        grouped_photoion_rates_df = self.__calculate_total_grouped_rates(
            photoion_rates_df
        )
        grouped_recomb_rates_df = self.__calculate_total_grouped_rates(
            recomb_rates_df
        )

        grouped_collisional_ionization_rates_df = (
            self.__calculate_total_grouped_rates(
                collisional_ionization_rates_df
            )
        )
        grouped_collisional_recombination_rates_df = (
            self.__calculate_total_grouped_rates(
                collision_recombination_rates_df
            )
        )

        rate_matrices = pd.DataFrame(
            index=grouped_photoion_rates_df.groups.keys(),
            columns=photoion_rates_df.columns,
        )

        for atomic_number in grouped_photoion_rates_df.groups.keys():
            photoion_rates = grouped_photoion_rates_df.get_group(atomic_number)
            recomb_rates = grouped_recomb_rates_df.get_group(atomic_number)
            coll_ion_rates = grouped_collisional_ionization_rates_df.get_group(
                atomic_number
            )
            recomb_ion_rates = (
                grouped_collisional_recombination_rates_df.get_group(
                    atomic_number
                )
            )
            ion_states = atomic_number + 1
            for shell in range(len(photoion_rates.columns)):
                photoion_matrix = self.__construct_rate_matrix(
                    photoion_rates, shell, ion_states
                )
                recomb_matrix = self.__construct_rate_matrix(
                    recomb_rates, shell, ion_states
                )
                coll_ion_matrix = self.__construct_rate_matrix(
                    coll_ion_rates, shell, ion_states
                )
                coll_recomb_matrix = self.__construct_rate_matrix(
                    recomb_ion_rates, shell, ion_states
                )

                matrix_array = (
                    photoion_matrix
                    + recomb_matrix
                    + coll_ion_matrix
                    + coll_recomb_matrix
                ).toarray()
                np.fill_diagonal(matrix_array, -np.sum(matrix_array, axis=0))
                matrix_array[1, :] = 1
                if charge_conservation:
                    charge_conservation_row = np.hstack(
                        (np.arange(0, ion_states), -1)
                    )
                    matrix_array = np.pad(matrix_array, ((0, 0), (0, 1)))
                    matrix_array = np.vstack(
                        (charge_conservation_row, matrix_array)
                    )
                rate_matrices.loc[atomic_number, shell] = matrix_array

        rate_matrices.index.names = ["atomic_number"]

        return rate_matrices

    def solve_analytic(
        self,
        radiation_field,
        thermal_electron_energy_distribution,
        lte_level_population,
        level_population,
        lte_ion_population,
        ion_population,
        partition_function,
        boltzmann_factor,
        charge_conservation=False,
    ):
        """Compute the ionization rate matrix.

        Parameters
        ----------
        radiation_field : RadiationField
            A radiation field that can compute its mean intensity.
        thermal_electron_energy_distribution : ThermalElectronEnergyDistribution
            Electron properties.
        lte_level_population : pd.DataFrame
            LTE level number density. Columns are cells.
        level_population : pd.DataFrame
            Estimated level number density. Columns are cells.
        lte_ion_population : pd.DataFrame
            LTE ion number density. Columns are cells.
        ion_population : pd.DataFrame
            Estimated ion number density. Columns are cells.
        partition_function : pd.DataFrame
            Partition function values. Index is atomic number and ion number,
            columns are cells.
        boltzmann_factor : pd.DataFrame
            Boltzmann factor values. Index is atomic number, ion number and level number,
            columns are cells.
        charge_conservation : bool, optional
            Whether to include a charge conservation row in the rate matrix.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rate matrices indexed by atomic number and ion number,
            with each column being a cell. Each entry is a numpy array.
        """
        photoion_rates_df, recomb_rates_df = (
            self.radiative_ionization_rate_solver.solve(
                radiation_field,
                thermal_electron_energy_distribution,
                lte_level_population,
                level_population,
                lte_ion_population,
                ion_population,
                partition_function,
                boltzmann_factor,
            )
        )

        # Lucy 2003 Eq 14
        level_to_ion_population_factor = lte_level_population / (
            lte_ion_population.values
            * thermal_electron_energy_distribution.number_density.value
        )

        collisional_ionization_rates_df, collision_recombination_rates_df = (
            self.collisional_ionization_rate_solver.solve(
                thermal_electron_energy_distribution,
                level_to_ion_population_factor,
                partition_function,
                boltzmann_factor,
            )
        )

        return self.__build_rate_matrices(
            photoion_rates_df,
            recomb_rates_df,
            collisional_ionization_rates_df,
            collision_recombination_rates_df,
            charge_conservation,
        )

    def solve_estimated(
        self,
        thermal_electron_energy_distribution,
        radfield_mc_estimators,
        time_simulation,
        volume,
        lte_level_population,
        level_population,
        lte_ion_population,
        ion_population,
        partition_function,
        boltzmann_factor,
        charge_conservation=False,
    ):
        """Compute the ionization rate matrix.

        Parameters
        ----------
        thermal_electron_energy_distribution : ThermalElectronEnergyDistribution
            Electron properties.
        radfield_mc_estimators : RadiationFieldMCEstimators
            Radiation field estimators from the Monte Carlo calculation.
        time_simulation : float
            The simulation time.
        volume : np.ndarray
            The volume of the cells.
        lte_level_population : pd.DataFrame
            LTE level number density. Columns are cells.
        level_population : pd.DataFrame
            Estimated level number density. Columns are cells.
        lte_ion_population : pd.DataFrame
            LTE ion number density. Columns are cells.
        partition_function : pd.DataFrame
            Partition function values. Index is atomic number and ion number,
            columns are cells.
        boltzmann_factor : pd.DataFrame
            Boltzmann factor values. Index is atomic number, ion number and level number,
            columns are cells.
        charge_conservation : bool, optional
            Whether to include a charge conservation row in the rate matrix.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rate matrices indexed by atomic number and ion number,
            with each column being a cell. Each entry is a numpy array.
        """
        # Lucy 2003 Eq 14
        level_to_ion_population_factor = lte_level_population / (
            lte_ion_population.values
            * thermal_electron_energy_distribution.number_density.value
        )

        photoion_rates_df, recomb_rates_df = (
            self.radiative_ionization_rate_solver.solve(
                thermal_electron_energy_distribution,
                radfield_mc_estimators,
                time_simulation,
                volume,
                lte_level_population,
                level_population,
                lte_ion_population,
                ion_population,
                level_to_ion_population_factor,
                partition_function,
                boltzmann_factor,
            )
        )

        collisional_ionization_rates_df, collision_recombination_rates_df = (
            self.collisional_ionization_rate_solver.solve(
                thermal_electron_energy_distribution,
                level_to_ion_population_factor,
                partition_function,
                boltzmann_factor,
            )
        )

        return self.__build_rate_matrices(
            photoion_rates_df,
            recomb_rates_df,
            collisional_ionization_rates_df,
            collision_recombination_rates_df,
            charge_conservation,
        )


class LTEIonRateMatrix:
    @staticmethod
    def _prepare_phi(phi, ion_index):
        # Check for Nans
        no_nans = pd.isnull(phi).sum().sum()
        if no_nans:
            # maybe add a warning
            phi = phi.fillna(phi.min().min())

        # Zero phi values pose a problem for the root finding algorithm. Set them to a small value.
        phi[phi == 0.0] = 1.0e-10 * phi[phi > 0.0].min().min()

        atomic_number = phi.index.get_level_values(0).values
        ion_number = phi.index.get_level_values(1).values
        new_index = pd.MultiIndex.from_arrays([atomic_number, ion_number - 1])
        phi_prep = phi.set_index(new_index).reindex(ion_index).fillna(0.0)
        return phi_prep

    def solve(self, phi, ion_index, charge_conservation=False):
        """Compute the LTE ionization rate matrix.

        Parameters
        ----------
        phi : pd.DataFrame
            Saha factor DataFrame indexed by atomic number and ion number.
        ion_index : pd.Index
            Index of all ion states.
        charge_conservation : bool, optional
            Whether to include a charge conservation row in the rate matrix.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rate matrices indexed by atomic number,
            with each column being a cell. Each entry is a numpy array.
        """
        phi_grouped = phi.groupby(level=("atomic_number"))

        rate_matrices = pd.DataFrame(
            index=phi_grouped.groups.keys(),
            columns=phi.columns,
        )

        phi_prep = self._prepare_phi(phi, ion_index)

        for atomic_number in phi_grouped.groups.keys():
            ion_states = atomic_number + 1
            for shell in range(len(phi.columns)):
                lte_diag = -phi_prep[shell].values
                lte_offdiag = (lte_diag != 0).astype(float)[:-1]

                matrix_array = np.diag(lte_diag) + np.diag(lte_offdiag, k=1)

                matrix_array[1, :] = 1
                if charge_conservation:
                    charge_conservation_row = np.hstack(
                        (np.arange(0, ion_states), -1)
                    )
                    matrix_array = np.pad(matrix_array, ((0, 0), (0, 1)))
                    matrix_array = np.vstack(
                        (charge_conservation_row, matrix_array)
                    )
                rate_matrices.loc[atomic_number, shell] = matrix_array

        rate_matrices.index.names = ["atomic_number"]

        return rate_matrices
