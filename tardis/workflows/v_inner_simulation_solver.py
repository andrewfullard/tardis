import logging

import numpy as np
from astropy import units as u

from tardis.io.atom_data.base import AtomData
from tardis.simulation.convergence import ConvergenceSolver
from tardis.spectrum.formal_integral import FormalIntegrator
from tardis.workflows.standard_simulation_solver import StandardSimulationSolver
from tardis.workflows.util import get_tau_integ

from scipy.interpolate import interp1d

# logging support
logger = logging.getLogger(__name__)

# TODO:
# Option to estimate initial v_inner from electron opacity
# Add option for desired optical depth
# Think about csvy vs other formats for specifying grid
# Handle non-explicit formats when going out of the simulation


class InnerVelocitySimulationSolver(StandardSimulationSolver):

    TAU_TARGET = np.log(2.0 / 3)

    def __init__(
        self,
        configuration,
        mean_optical_depth="rossland",
    ):
        """
        Args:
            convergence_strategy (_type_): _description_
            atom_data_path (_type_): _description_
            mean_optical_depth (str): 'rossland' or 'planck'
                Method of estimating the mean optical depth
        """
        super().__init__(configuration)
        self.mean_optical_depth = mean_optical_depth

        self.v_inner_convergence_solver = ConvergenceSolver(
            self.convergence_strategy.v_inner
        )

    def _get_convergence_status(
        self,
        t_radiative,
        dilution_factor,
        t_inner,
        v_inner,
        estimated_t_radiative,
        estimated_dilution_factor,
        estimated_t_inner,
        estimated_v_inner,
    ):
        t_radiative_converged = (
            self.t_radiative_convergence_solver.get_convergence_status(
                t_radiative.value,
                estimated_t_radiative.value,
                self.simulation_state.no_of_shells,
            )
        )

        dilution_factor_converged = (
            self.dilution_factor_convergence_solver.get_convergence_status(
                dilution_factor,
                estimated_dilution_factor,
                self.simulation_state.no_of_shells,
            )
        )

        t_inner_converged = (
            self.t_inner_convergence_solver.get_convergence_status(
                t_inner.value,
                estimated_t_inner.value,
                1,
            )
        )

        v_inner_converged = (
            self.v_inner_convergence_solver.get_convergence_status(
                v_inner.value,
                estimated_v_inner.value,
                1,
            )
        )

        if np.all(
            [
                t_radiative_converged,
                dilution_factor_converged,
                t_inner_converged,
                v_inner_converged,
            ]
        ):
            hold_iterations = self.convergence_strategy.hold_iterations
            self.consecutive_converges_count += 1
            logger.info(
                f"Iteration converged {self.consecutive_converges_count:d}/{(hold_iterations + 1):d} consecutive "
                f"times."
            )
            # If an iteration has converged, require hold_iterations more
            # iterations to converge before we conclude that the Simulation
            # is converged.
            return self.consecutive_converges_count == hold_iterations + 1

        self.consecutive_converges_count = 0
        return False
    

    def estimate_v_inner(self):
        """Compute the Rossland Mean Optical Depth,
        Estimate location where v_inner makes t=2/3 (or target)
        Extrapolate with exponential fits

        Need some way to return and inspect the optical depths for later logging"""
        pass

        tau_integ = np.log(
            get_tau_integ(
                self.plasma,
                self.simulation_state,
            )[self.mean_optical_depth]
        )

        interpolator = interp1d(
            tau_integ,
            self.simulation_state.v_inner, # Only use the active values as we only need a numerical estimate, not an index
            fill_value="extrapolate",
        )
        # TODO: Make sure eastimed_v_inner is within the bounds of the simulation!
        estimated_v_inner = interpolator(self.TAU_TARGET)

        return estimated_v_inner

    def get_convergence_estimates(self, emitted_luminosity):
        (
            estimated_t_radiative,
            estimated_dilution_factor,
        ) = (
            self.transport_solver.transport_state.calculate_radiationfield_properties()
        )

        estimated_t_inner = self.estimate_t_inner(
            self.simulation_state.t_inner,
            self.luminosity_requested,
            emitted_luminosity,
            t_inner_update_exponent=self.convergence_strategy.t_inner_update_exponent,
        )
        estimated_v_inner = self.estimate_v_inner()
        return (
            estimated_t_radiative,
            estimated_dilution_factor,
            estimated_t_inner,
            estimated_v_inner,
        )

    def check_convergence(
        self,
        estimated_t_radiative,
        estimated_dilution_factor,
        estimated_t_inner,
        estimated_v_inner,
    ):
        converged = self._get_convergence_status(
            self.simulation_state.t_radiative,
            self.simulation_state.dilution_factor,
            self.simulation_state.t_inner,
            estimated_t_radiative,
            estimated_dilution_factor,
            estimated_t_inner,
            estimated_v_inner
        )

        return converged
    
    def clip(self, property):
        """Clips a shell-dependent array to the current index"""

        return property[
            self.simulation_state.geometry.v_inner_boundary_index : self.simulation_state.geometry.v_outer_boundary_index
        ]

    def solve_plasma(
        self,
        estimated_t_radiative,
        estimated_dilution_factor,
        estimated_t_inner,
        estimated_v_inner,
    ):
        next_t_radiative = self.t_rad_convergence_solver.converge(
            self.simulation_state.t_radiative,
            estimated_t_radiative,
        )
        next_dilution_factor = self.dilution_factor_convergence_solver.converge(
            self.simulation_state.dilution_factor,
            estimated_dilution_factor,
        )
        next_v_inner = self.v_inner_convergence_solver.converge(
            self.simulation_state.v_boundary_inner,
            estimated_v_inner,
        )  # TODO: Add option to lock cycles as well

        if (
            self.iterations_executed + 1
        ) % self.convergence_strategy.lock_t_inner_cycles == 0:
            next_t_inner = self.t_inner_convergence_solver.converge(
                self.simulation_state.t_inner,
                estimated_t_inner,
            )
        else:
            next_t_inner = self.simulation_state.t_inner
        self.simulation_state.geometry.v_boundary_inner = (
                    next_v_inner  # TODO: Check reset previously masked values, should automattically happen but not sure, make sure we're setting the correct property
                )
        self.simulation_state.t_radiative = next_t_radiative
        self.simulation_state.dilution_factor = next_dilution_factor
        self.simulation_state.blackbody_packet_source.temperature = next_t_inner

        
        # TODO: Figure out how to handle the missing/extra plasma properties
        update_properties = dict(
            t_rad=self.simulation_state.t_radiative,
            w=self.simulation_state.dilution_factor,
            r_inner=self.simulation_state.r_inner.to(u.cm)
        )
        # A check to see if the plasma is set with JBluesDetailed, in which
        # case it needs some extra kwargs.

        estimators = self.transport.transport_state.radfield_mc_estimators
        if "j_blue_estimator" in self.plasma.outputs_dict:
            update_properties.update(
                t_inner=next_t_inner,
                j_blue_estimator=estimators.j_blue_estimator,
            )
        if "gamma_estimator" in self.plasma.outputs_dict:
            update_properties.update(
                gamma_estimator=estimators.photo_ion_estimator,
                alpha_stim_estimator=estimators.stim_recomb_estimator,
                bf_heating_coeff_estimator=estimators.bf_heating_estimator,
                stim_recomb_cooling_coeff_estimator=estimators.stim_recomb_cooling_estimator,
            )

        self.plasma_solver.update(**update_properties)


    def solve_spectrum(
        self,
        transport_state,
        virtual_packet_energies=None,
        integrated_spectrum_settings=None,
    ):
        # Set up spectrum solver
        self.spectrum_solver.transport_state = transport_state
        if virtual_packet_energies is not None:
            self.spectrum_solver._montecarlo_virtual_luminosity.value[
                :
            ] = virtual_packet_energies

        if integrated_spectrum_settings is not None:
            # Set up spectrum solver integrator
            self.spectrum_solver.integrator_settings = (
                integrated_spectrum_settings
            )
            self.spectrum_solver._integrator = FormalIntegrator(
                self.simulation_state, self.plasma, self.transport
            )

    def calculate_emitted_luminosity(self, transport_state):
        self.spectrum_solver.transport_state = transport_state

        output_energy = (
            self.transport.transport_state.packet_collection.output_energies
        )
        if np.sum(output_energy < 0) == len(output_energy):
            logger.critical("No r-packet escaped through the outer boundary.")

        emitted_luminosity = self.spectrum_solver.calculate_emitted_luminosity(
            self.luminosity_nu_start, self.luminosity_nu_end
        )
        return emitted_luminosity

    def solve(self):
        converged = False
        while self.completed_iterations < self.total_iterations - 1:
            transport_state, virtual_packet_energies = self.solve_montecarlo()

            emitted_luminosity = self.calculate_emitted_luminosity(
                transport_state
            )

            (
                estimated_t_radiative,
                estimated_dilution_factor,
                estimated_t_inner,
                estimated_v_inner,
            ) = self.get_convergence_estimates(emitted_luminosity)

            self.solve_plasma(
                estimated_t_radiative,
                estimated_dilution_factor,
                estimated_t_inner,
                estimated_v_inner,
            )

            converged = self.check_convergence(
                estimated_t_radiative,
                estimated_dilution_factor,
                estimated_t_inner,
                estimated_v_inner,
            )
            self.completed_iterations += 1

            if converged and self.convergence_strategy.stop_if_converged:
                break

        transport_state, virtual_packet_energies = self.solve_montecarlo(self.final_iteration_packet_count, self.virtual_packet_count
        )
        self.initialize_spectrum_solver(
            transport_state,
            virtual_packet_energies,
        )
