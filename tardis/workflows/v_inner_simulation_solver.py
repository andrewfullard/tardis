import logging

import numpy as np
from astropy import units as u

from tardis.io.atom_data.base import AtomData
from tardis.simulation.convergence import ConvergenceSolver
from tardis.spectrum.formal_integral import FormalIntegrator
from tardis.workflows.standard_simulation_solver import StandardSimulationSolver
from tardis.workflows.util import get_tau_integ
from tardis.plasma.standard_plasmas import assemble_plasma

from scipy.interpolate import interp1d
import copy

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

        self.convergence_solvers["v_inner_boundary"] = ConvergenceSolver(
            self.convergence_strategy.v_inner_boundary
        )
    
        self.property_mask = self.simulation_state.property_mask


    def estimate_v_inner(self):
        """Compute the Rossland Mean Optical Depth,
        Estimate location where v_inner makes t=2/3 (or target)
        Extrapolate with exponential fits

        Need some way to return and inspect the optical depths for later logging"""
        pass

        tau_integ = np.log(
            get_tau_integ(
                self.plasma_solver,
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

        return estimated_v_inner * u.cm/u.s

    def get_convergence_estimates(self, transport_state):

        (
            estimated_t_radiative,
            estimated_dilution_factor,
        ) = self.transport_solver.transport_state.calculate_radiationfield_properties()

        self.initialize_spectrum_solver(
            transport_state,
            None,
        )

        emitted_luminosity = self.spectrum_solver.calculate_emitted_luminosity(
            self.luminosity_nu_start, self.luminosity_nu_end
        )
        print("Emitted Luminosity:", emitted_luminosity)
        luminosity_ratios = (
            (emitted_luminosity / self.luminosity_requested).to(1).value
        )

        estimated_t_inner = (
            self.simulation_state.t_inner
            * luminosity_ratios
            ** self.convergence_strategy.t_inner_update_exponent
        )

        estimated_v_inner = self.estimate_v_inner()
        if estimated_v_inner < self.simulation_state.geometry.v_inner[0]:
            estimated_v_inner = self.simulation_state.geometry.v_inner[0]
        elif estimated_v_inner > self.simulation_state.geometry.v_inner[-1]:
            estimated_v_inner = self.simulation_state.geometry.v_inner[-1]
        print(estimated_v_inner)

        return {
            "t_radiative": estimated_t_radiative,
            "dilution_factor": estimated_dilution_factor,
            "t_inner": estimated_t_inner,
            "v_inner_boundary": estimated_v_inner,
        }

    def check_convergence(
        self,
        estimated_values,
    ):
        convergence_statuses = []

        for key, solver in self.convergence_solvers.items():

            current_value = getattr(self.simulation_state, key)
            estimated_value = estimated_values[key]
            print('Check Convergence')
            print(key, estimated_value)
            print(current_value)
            if hasattr(current_value, '__len__') and (key not in ["t_inner", "v_inner_boundary"]):
                print(key, 'has', '__len__')
                new_value = estimated_value
                current_value_expanded = np.empty(len(self.simulation_state.geometry.r_inner), dtype=current_value.dtype)
                current_value_expanded[self.simulation_state.property_mask] = current_value
                new_value_expanded = np.empty_like(current_value_expanded)
                new_value_expanded[self.new_property_mask] = new_value
                joint_mask = self.simulation_state.property_mask & self.new_property_mask
                if hasattr(current_value, 'unit'):
                    current_value_expanded = current_value_expanded * current_value.unit
                    new_value_expanded = new_value_expanded * current_value.unit
                estimated_value = new_value_expanded[joint_mask]
                current_value = current_value_expanded[joint_mask]

            
            no_of_shells = (
                self.simulation_state.no_of_shells if key not in ["t_inner", "v_inner_boundary"] else 1
            )
            
            convergence_statuses.append(
                solver.get_convergence_status(
                    current_value, estimated_value, no_of_shells
                )
            )
            print('Status:', convergence_statuses[-1])

        if np.all(convergence_statuses):
            hold_iterations = self.convergence_strategy.hold_iterations
            self.consecutive_converges_count += 1
            logger.info(
                f"Iteration converged {self.consecutive_converges_count:d}/{(hold_iterations + 1):d} consecutive "
                f"times."
            )
            print('Converged this iteration!')
            return self.consecutive_converges_count == hold_iterations + 1

        self.consecutive_converges_count = 0
        return False
    
    def clip(self, property):
        """Clips a shell-dependent array to the current index"""

        return property[
            self.simulation_state.geometry.v_inner_boundary_index : self.simulation_state.geometry.v_outer_boundary_index
        ]

    def solve_simulation_state(
        self,
        estimated_values,
    ):
        next_values = {}
        print(estimated_values)
        self.new_property_mask = self.simulation_state.property_mask
        self.old_property_mask = self.property_mask

        for key, solver in self.convergence_solvers.items():
            if (
                key in ["t_inner"]
                and (self.completed_iterations + 1)
                % self.convergence_strategy.lock_t_inner_cycles
                != 0
            ):
                next_values[key] = getattr(self.simulation_state, key)
            else:
                print('key', key)
                print(getattr(self.simulation_state, key))
                print(estimated_values[key])
                current_value = getattr(self.simulation_state, key)
                new_value = estimated_values[key]
                if hasattr(current_value, '__len__') and key not in ["t_inner", "v_inner_boundary"]:
                    print(key, 'has', '__len__')
                    current_value_expanded = np.empty(len(self.simulation_state.geometry.r_inner), dtype=current_value.dtype)
                    current_value_expanded[self.simulation_state.property_mask] = current_value
                    new_value_expanded = np.empty_like(current_value_expanded)
                    new_value_expanded[self.new_property_mask] = new_value
                    joint_mask = self.simulation_state.property_mask & self.new_property_mask
                    if hasattr(current_value, 'unit'):
                        current_value_expanded = current_value_expanded * current_value.unit
                        new_value_expanded = new_value_expanded * current_value.unit
                    new_value = new_value_expanded[joint_mask]
                    current_value = current_value_expanded[joint_mask]
                next_values[key] = solver.converge(
                    current_value, new_value
                ) # TODO: This needs to be changed to account for changing array sizes
        
        self.simulation_state.t_radiative = next_values["t_radiative"]
        self.simulation_state.dilution_factor = next_values["dilution_factor"]
        self.simulation_state.blackbody_packet_source.temperature = next_values[
            "t_inner"
        ]
        print('next v_inner', next_values["v_inner_boundary"])
        self.simulation_state.geometry.v_inner_boundary = next_values[
            "v_inner_boundary"
        ]
        self.property_mask = self.new_property_mask

        
    def solve_plasma(
        self,
        transport_state,
    ):
        # TODO: Find properties that need updating with shells

        update_properties = dict(
            t_rad=self.simulation_state.t_radiative,
            w=self.simulation_state.dilution_factor,
            r_inner=self.simulation_state.r_inner.to(u.cm),
            number_density=self.simulation_state.elemental_number_density,
            volume=self.simulation_state.volume,
            abundance=self.simulation_state.abundance,
            lines=None,
        )
        # A check to see if the plasma is set with JBluesDetailed, in which
        # case it needs some extra kwargs.
        if "j_blue_estimator" in self.plasma_solver.outputs_dict:
            update_properties.update(
                t_inner=self.simulation_state.blackbody_packet_source.temperature,
                j_blue_estimator=transport_state.radfield_mc_estimators.j_blue_estimator,
            )

        self.plasma_solver.update(**update_properties)


    def solve(self):
        converged = False
        while self.completed_iterations < self.total_iterations - 1:
            transport_state, virtual_packet_energies = self.solve_montecarlo(
                self.real_packet_count
            )

            estimated_values = self.get_convergence_estimates(
                transport_state
            )

            self.solve_simulation_state(estimated_values)

            self.solve_plasma(transport_state)

            converged = self.check_convergence(estimated_values)
            self.completed_iterations += 1
            if converged:
                print("SIMULATION CONVERGED!")
            if converged and self.convergence_strategy.stop_if_converged:
                break

        transport_state, virtual_packet_energies = self.solve_montecarlo(
            self.final_iteration_packet_count, self.virtual_packet_count
        )
        self.initialize_spectrum_solver(
            transport_state,
            virtual_packet_energies,
        )