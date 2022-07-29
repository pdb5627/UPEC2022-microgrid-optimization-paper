"""
Operational optimization problem formulation
Built using Pyomo for modeling
"""
import logging
from pyomo.environ import Var, Expression, Constraint, quicksum
from pyomo.core.expr.current import evaluate_expression
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds
from pyomo.common.tee import capture_output
from typing import Callable, Optional, Sequence, Union
from types import SimpleNamespace
from dataclasses import dataclass

from pyomo_utils import *

import numpy as np
import numpy.ma as ma
import pandas as pd
logger = logging.getLogger(__name__)

#pyomo.util.infeasible.logger.setLevel(logging.DEBUG)
neos_email_address = 'neos@mailinator.com'


def num_switches_old(model_var):
    idx = list(model_var.index_set())
    return quicksum(abs(model_var[t] - model_var[t - 1]) for t in idx[1:])


def num_switches(model_var):
    idx = list(model_var.index_set())
    return quicksum(linear_abs(model_var[t] - model_var[t - 1]) for t in idx[1:])


def initializer(data_init, converter: Callable = float):
    return lambda m, t: converter(data_init[t])


class PiecewiseFunction:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys)


@dataclass
class TwoPumpModelParameters:
    # Model metaparameters
    lookahead: int = 24  # number of periods
    delta_t: float = 1  # hr

    # Model data
    Pload: Optional[np.array] = None  # W
    P_PVavail: Optional[np.array] = None  # W

    Ppump1_max: float = 2200.  # W
    Qw1: float = 50.  # m^3 / hr

    Ppump2_max: float = 2200.  # W
    Ppump2_min_pu: float = 0.3  # pu of Ppump2_max
    Ppump2_min: Optional[float] = None  # W
    # The following four parameters are used by the default linear Qw2 and Ppump2 functions.
    # Constant efficiency is assumed in the default functions.
    eta_pump: float = 0.4  # pu
    head: float = 44.  # m
    rho: float = 1000.  # kg / m^3
    g: float = 9.81  # m / s^2
    # If the following two parameters are defined, a linear interpolation function will be used rather than
    # the default constant-efficiency function.
    Qpump2_min: Optional[float] = None  # m^3 / hr
    Qpump2_max: Optional[float] = None  # m^3 / hr

    Qw2: Optional[Callable] = None  # Function taking Ppump [W] returning Qpump [m^3 / hr]
    Ppump2: Optional[Callable] = None  # Function taking Qpump [m^3 / hr] returning Ppump [W]

    E_BSS_max: float = 9600.  # W-h
    E_BSS_lower_pu: float = 0.1  # pu of E_BSS_max
    E_BSS_lower: Optional[float] = None  # W-h
    E_BSS_upper_pu: float = 0.95  # pu of E_BSS_max
    E_BSS_upper: Optional[float] = None  # W-h
    E_BSS0_pu: float = 0.3  # pu of E_BSS_max
    E_BSS0: Optional[float] = None  # W-h
    sinv0: int = 0  # Binary. 0 = BSS/PV supplies load. 1 = Grid supplies load
    P_BSS_ch_max_pu: float = 1 / 10  # pu of E_BSS_max / 1hr
    P_BSS_ch_max: Optional[float] = None  # W
    P_BSS_disch_max_pu: float = 1 / 3  # pu of E_BSS_max / 1hr
    P_BSS_disch_max: Optional[float] = None  # W
    eta_BSS: float = 0.95  # unitless
    E_absorb_pu: float = 0.8  # pu of E_BSS_max
    K_BSS: Optional[float] = None  # unitless, in range 0 to 1

    Vw1_min: float = 5.  # m^3
    Vw1_max: float = 120.  # m^3
    Vw1_0: float = 50.  # m^3
    Vw2_min: float = 5.  # m^3
    Vw2_max: float = 50.  # m^3
    Vw2_0: float = 25.  # m^3

    Vuse_desired: Union[float, Sequence[float]] = 150.  # m^3
    D: Optional[Sequence[Sequence[int]]] = None
    Quse1_max: float = 50.  # m^3 / hr
    Quse2_max: float = 50.  # m^3 / hr

    Cgrid: Optional[np.array] = None  # Cost / W-h
    C_BSS: float = 0.01  # Cost / W-h
    C_BSS_switching: float = 1.  # Cost per switch
    C_pump_switching: float = 1.  # Cost per switch

    Cw_short: float = 100.  # Cost / m^3

    eta_w: Optional[np.array] = None  # unitless

    # eqn:pump-flow-relation
    # Constant efficiency
    def Qw2_default(self, Ppump2):
        return Ppump2 * 3600. * self.eta_pump / (self.head * self.rho * self.g)

    def Ppump2_default(self, Qw2):
        return Qw2 * self.head * self.rho * self.g / (3600. * self.eta_pump)

    # Linear relation based on interpolating between endpoints
    def Qw2_linear(self, Ppump2):
        if Ppump2 >= self.Ppump2_min:
            return self.Qpump2_min + (self.Qpump2_max - self.Qpump2_min) / (self.Ppump2_max - self.Ppump2_min) * (Ppump2 - self.Ppump2_min)
        else:
            return 0

    def Ppump2_linear(self, Qw2):
        if Qw2 >= self.Qpump2_min:
            return self.Ppump2_min + (self.Ppump2_max - self.Ppump2_min) / (self.Qpump2_max - self.Qpump2_min) * (Qw2 - self.Qpump2_min)
        else:
            return 0

    def __post_init__(self):
        # Set defaults for any optional parameters that require creation of new objects or calculations that depend on
        # other parameters

        if self.Pload is None:
            self.Pload = np.random.rand(self.lookahead) * 1e3
        if self.P_PVavail is None:
            self.P_PVavail = np.array([0, 0, 0, 0, 0, 0, 0.4, 2.8, 10.6, 15.7, 19.1, 21.2,
                                       21.7, 20.7, 18.7, 14.8, 9.2, 3.0, 0.3, 0, 0, 0, 0, 0]) * 6 / 22 * 1000.
        if self.Ppump2_min is None:
            self.Ppump2_min = self.Ppump2_min_pu * self.Ppump2_max

        if self.Qw2 is None and self.Ppump2 is None:
            if self.Qpump2_min is None or self.Qpump2_max is None:
                self.Qw2 = self.Qw2_default
                self.Ppump2 = self.Ppump2_default
            else:
                P_points = [0, self.Ppump2_min, self.Ppump2_min, self.Ppump2_max]
                Q_points = [0, 0,               self.Qpump2_min, self.Qpump2_max]
                self.Qw2 = PiecewiseFunction(P_points, Q_points)
                self.Ppump2 = PiecewiseFunction(Q_points, P_points)

        if self.E_BSS_lower is None:
            self.E_BSS_lower = self.E_BSS_lower_pu * self.E_BSS_max
        if self.E_BSS_upper is None:
            self.E_BSS_upper = self.E_BSS_upper_pu * self.E_BSS_max
        if self.E_BSS0 is None:
            self.E_BSS0 = self.E_BSS0_pu * self.E_BSS_max

        if self.P_BSS_ch_max is None:
            self.P_BSS_ch_max = self.E_BSS_max / 10
        if self.P_BSS_disch_max is None:
            self.P_BSS_disch_max = self.E_BSS_max / 3
        if self.K_BSS is None:
            self.K_BSS = self.P_BSS_ch_max * self.eta_BSS * self.delta_t / ((1 - self.E_absorb_pu) * self.E_BSS_max)
            self.K_BSS = min(self.K_BSS, 1.)

        if self.D is None:
            self.D = (tuple(range(self.lookahead)),)
        # If Vuse_desired is passed as a single value rather than a Sequence, convert it to a tuple
        if not isinstance(self.Vuse_desired, (Sequence, np.ndarray)):
            self.Vuse_desired = tuple(self.Vuse_desired for _ in self.D)

        if self.Cgrid is None:
            self.Cgrid = 0.15 * np.ones(self.lookahead)

        if self.eta_w is None:
            self.eta_w = 1.0 * np.ones(self.lookahead)


class TwoPumpModel:
    def __init__(self, params: TwoPumpModelParameters):
        self.params = params
        self._init = None
        self.model = None
        self.solver_output = None
        self.initialized_operation = None
        self.initialized_operation_scalars = None
        self.optimal_operation = None
        self.optimal_operation_scalars = None

    def initialize(self, init_pumping_method='cheapest', quiet=False):
        """
        Initializes a model with using a greedy pumping heuristic.
        :param init_pumping_method: 'first' for first possible pumping period for Pump 1,
            'cheapest' for running Pump 1 in the cheapest possible pumping period in each day that has inadequate water
        :return: None
        """
        logger.debug('Beginning model initialization.')
        logger.debug(f'Using pumping initialization method: {init_pumping_method}')
        p = self.params
        lookahead = p.lookahead
        i = SimpleNamespace()
        # Initialization Routine
        # On the first pass, assign all opportunistic pumping periods based on P_PVavail and Pload
        i.sinv = np.zeros(lookahead, dtype=np.bool)
        i.sBSS = np.ones(lookahead, dtype=np.bool)
        i.spump1 = np.zeros(lookahead, dtype=np.bool)
        i.E_BSS = np.ones(lookahead) * p.E_BSS0
        i.P_PV = np.array(p.P_PVavail)
        i.Pavail = np.array(p.P_PVavail)
        i.Ppump1 = np.zeros(lookahead)
        i.Ppump2 = np.zeros(lookahead)
        i.P_BSSch = np.zeros(lookahead)
        i.P_BSSdisch = np.zeros(lookahead)
        i.Quse1 = np.zeros(lookahead)
        i.Quse2 = np.zeros(lookahead)
        i.Vw1 = np.ones(lookahead) * p.Vw1_0
        i.Vw2 = np.ones(lookahead) * p.Vw2_0
        for t in range(lookahead):
            sinv_prev = i.sinv[t - 1] if t > 0 else p.sinv0
            E_BSS_prev = i.E_BSS[t - 1] if t > 0 else p.E_BSS0
            Vw2_prev = i.Vw2[t - 1] if t > 0 else p.Vw2_0

            # Inverter mode logic copied from model
            below_upper_thresh = (E_BSS_prev - p.E_BSS_upper <= 0)
            stay_on_utility = sinv_prev * below_upper_thresh
            switch_to_utility = 1. if E_BSS_prev <= p.E_BSS_lower else 0.
            i.sinv[t] = 1. if stay_on_utility or switch_to_utility else 0.

            if ~i.sinv[t] and p.P_PVavail[t] < p.Pload[t]:
                # Use battery capacity to supply load
                i.sBSS[t] = 0
                i.P_BSSdisch[t] = min(p.Pload[t] - p.P_PVavail[t], E_BSS_prev * p.eta_BSS / p.delta_t)

            i.Pavail[t] = p.P_PVavail[t] - ~i.sinv[t] * p.Pload[t]
            # Greedy water use
            # Check how much water is still needed for the day
            for dn, d in enumerate(p.D):
                if t in d:
                    Vuse = sum(i.Quse2[t2] + i.Quse1[t2] for t2 in d)
                    break
            else:
                raise ValueError(f'Time period {t=} not in any day ({p.D=})')
            Quse2_max = min(p.Quse2_max, p.Vuse_desired[dn] - Vuse)

            # Opportunistic pumping
            i.Ppump2[t] = min(i.Pavail[t],
                              p.Ppump2(p.Vw2_max + Quse2_max - Vw2_prev),
                              p.Ppump2_max)
            if i.Ppump2[t] < p.Ppump2_min:
                i.Ppump2[t] = 0
            i.Quse2[t] = min(Quse2_max, Vw2_prev + p.Qw2(i.Ppump2[t]) - p.Vw2_min)
            i.Vw2[t] = Vw2_prev + p.Qw2(i.Ppump2[t]) - i.Quse2[t]
            Vuse += i.Quse2[t]
            # Any remaining PV available stored in BSS
            if i.Pavail[t] - i.Ppump2[t] > 0:
                i.P_BSSch[t] = min(i.Pavail[t] - i.Ppump2[t],
                                   p.K_BSS * (p.E_BSS_max - E_BSS_prev) / (p.eta_BSS * p.delta_t),
                                   p.P_BSS_ch_max)
            i.P_PV[t] = ~i.sinv[t] * p.Pload[t] + i.Ppump2[t] + i.P_BSSch[t] - i.P_BSSdisch[t]
            #if i.P_BSSch[t] > 0:
            #    i.sBSS[t] = 1
            i.E_BSS[t] = E_BSS_prev \
                         + i.P_BSSch[t] * p.eta_BSS * p.delta_t \
                         - i.P_BSSdisch[t] / p.eta_BSS * p.delta_t

        # After any pumping based on available PV is done, check if water is sufficient
        # This can be found by formulating an LP problem to find the maximum effective water use possible
        # given the assigned pumping schedule.
        infeasible_pumping = np.zeros(lookahead, dtype=np.bool)
        t_pump = None
        while not np.all(i.spump1):
            water_use_model = self.build_water_use_model(i)

            opt = pyo.SolverFactory('glpk')
            #log_infeasible_constraints(water_use_model)
            #log_infeasible_bounds(water_use_model)
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = opt.solve(water_use_model, tee=True)

            if results.Solver[0]['Termination condition'] in ('infeasible', 'other'):
                # log_infeasible_constraints(water_use_model)
                # log_infeasible_bounds(water_use_model)
                # log_close_to_bounds(water_use_model)
                if t_pump is None:
                    logger.debug(f'Initialization infeasible even before any pumping is assigned.')
                    break
                else:
                    logger.debug(f'Initialization detected infeasible pumping assignment at t={t_pump}. Backing out.')
                    infeasible_pumping[t_pump] = True
                    i.spump1[t_pump] = 0
                    water_use_model = self.build_water_use_model(i)
            else:
                if t_pump is not None:
                    logger.debug(f'Initialization found feasible pumping assignment at t={t_pump}.')
                    infeasible_pumping = np.zeros(lookahead, dtype=np.bool)

            # water_use_model.obj.display()
            # water_use_model.display()
            # water_use_model.pprint()
            Vuse_init_max = water_use_model.obj()
            _, optimal_water_use = vars_to_df(water_use_model)
            i.Quse1 = optimal_water_use['Quse1'].values
            i.Vw1 = optimal_water_use['Vw1'].values
            i.Quse2 = optimal_water_use['Quse2'].values
            i.Vw2 = optimal_water_use['Vw2'].values
            if Vuse_init_max >= sum(p.Vuse_desired)*0.99:
                # Break when water usage is as much as desired. A small fudge factor is used since the
                # numerical results may be slightly less than the threshold.
                break

            # Add more pumping! Do pumping from grid in period with minimum cost in the first day in which there is a
            # deficit of water.
            can_pump = (i.spump1 == 0) & ~infeasible_pumping & ((i.Vw1 + p.Qw1) < p.Vw1_max)
            if init_pumping_method == 'cheapest':
                for dn, d in enumerate(p.D):
                    # Check if pumping is needed on this day. If not, skip ahead
                    if optimal_water_use['Vuse'][dn] >= 0.99*p.Vuse_desired[dn]:
                        continue

                    Cgrid_avail = ma.array(p.Cgrid, mask=~can_pump)
                    # Mask all days except the one being checked
                    for d_other in p.D:
                        if d_other is d:
                            continue
                        for t in d_other:
                            Cgrid_avail[t] = ma.masked
                    # If no available hours, skip to the next day
                    if np.all(Cgrid_avail.mask):
                        continue
                    # Assign pumping to the cheapest hour available
                    t_pump = np.argmin(Cgrid_avail)
                    i.spump1[t_pump] = 1
                    break
                else:
                    # If no time was found to assign pumping to, exit the loop. Initialization problem may be infeasible.
                    break
            elif init_pumping_method == 'first':
                Cgrid_avail = ma.array(p.Cgrid, mask=~can_pump)
                t_pump = np.nonzero(Cgrid_avail)[0][0]
                i.spump1[t_pump] = 1
            else:
                raise ValueError(f'Invalid init_pumping_method value "{init_pumping_method}"')

        # Assign at end of function to ensure that the data is complete before being assigned
        self._init = i

    def build_water_use_model(self, i):
        p = self.params
        water_use_model = pyo.ConcreteModel()
        water_use_model.t = pyo.RangeSet(0, p.lookahead - 1)
        water_use_model.d = pyo.RangeSet(0, len(p.D) - 1)
        water_use_model.Quse1 = Var(water_use_model.t, bounds=(0, p.Quse1_max), initialize=initializer(i.Quse1))
        water_use_model.Quse2 = Var(water_use_model.t, bounds=(0, p.Quse2_max), initialize=initializer(i.Quse2))
        water_use_model.Vw1 = Expression(water_use_model.t, rule=lambda m, t: p.Vw1_0 + quicksum(
            i.spump1[k] * p.Qw1 - m.Quse1[k] for k in range(t + 1)) * p.delta_t)
        water_use_model.Vw2 = Expression(water_use_model.t, rule=lambda m, t: p.Vw2_0 + quicksum(
            p.Qw2(i.Ppump2[k]) - m.Quse2[k] for k in range(t + 1)) * p.delta_t)

        water_use_model.Vuse = Expression(water_use_model.d, rule=lambda m, d:
            pyo.quicksum(p.eta_w[t] * water_use_model.Quse1[t] * p.delta_t +
                         p.eta_w[t] * water_use_model.Quse2[t] * p.delta_t
                         for t in p.D[d]))
        water_use_model.Vuse_total = pyo.sum_product(water_use_model.Vuse)

        water_use_model.obj = pyo.Objective(expr=water_use_model.Vuse_total, sense=pyo.maximize)
        water_use_model.Vw1_min = Constraint(water_use_model.t, rule=lambda m, t: p.Vw1_min <= m.Vw1[t])
        water_use_model.Vw1_max = Constraint(water_use_model.t, rule=lambda m, t: m.Vw1[t] <= p.Vw1_max)
        water_use_model.Vw2_min = Constraint(water_use_model.t, rule=lambda m, t: p.Vw2_min <= m.Vw2[t])
        water_use_model.Vw2_max = Constraint(water_use_model.t, rule=lambda m, t: m.Vw2[t] <= p.Vw2_max)

        water_use_model.Vuse_limit = Constraint(water_use_model.d, rule=lambda m, d: m.Vuse[d] <= p.Vuse_desired[d])

        return water_use_model


    def build_model(self):
        logger.debug('Building model')
        model = pyo.ConcreteModel()
        p = self.params
        i = self._init

        # Model index
        model.t = pyo.RangeSet(0, p.lookahead - 1)
        model.d = pyo.RangeSet(0, len(p.D) - 1)

        # Decision Variables
        model.spump1 = Var(model.t, domain=pyo.Binary, initialize=initializer(i.spump1))
        if p.Ppump2_min > 0:
            model.Ppump2 = semicontinuous_var(model.t, bounds=(p.Ppump2_min, p.Ppump2_max),
                                              initialize=initializer(i.Ppump2))
        else:
            # Must be == 0
            model.Ppump2 = Var(model.t, bounds=(0, p.Ppump2_max),
                               initialize=initializer(i.Ppump2))
        model.Quse1 = Var(model.t, bounds=(0, p.Quse1_max), initialize=initializer(i.Quse1))  # eqn:var-limit
        model.Quse2 = Var(model.t, bounds=(0, p.Quse2_max), initialize=initializer(i.Quse2))  # eqn:var-limit

        # Intermediate Variables
        model.E_BSS = Var(model.t, bounds=(0, p.E_BSS_max), initialize=initializer(i.E_BSS))

        # eqn:pump1-power
        model.Ppump1 = Expression(model.t, rule=lambda m, t: p.Ppump1_max * m.spump1[t])

        # eqn:pump1-flow
        model.Qpump1 = Expression(model.t, rule=lambda m, t: p.Qw1 * m.spump1[t])

        # eqn:pump2-flow
        model.Qpump2 = Var(model.t, bounds=(0, p.Qw2(p.Ppump2_max)), initialize=initializer(p.Qw2(i.Ppump2)))
        if hasattr(p.Qw2, 'xs') and hasattr(p.Qw2, 'ys'):
            model.Ppump2v = Var(model.t, bounds=(0, p.Ppump2_max), initialize=initializer(i.Ppump2))
            model.Ppump2v_eq = Constraint(model.t, rule=lambda m, t: m.Ppump2[t] == m.Ppump2v[t])
            model.Qpump2_eq = pyo.Piecewise(model.t, model.Qpump2, model.Ppump2v,
                                            pw_pts=p.Qw2.xs, f_rule=p.Qw2.ys,
                                            pw_constr_type='EQ', pw_repn='INC')  # SOS2 no supported for CBC solver
        else:
            model.Qpump2_eq = Constraint(model.t, rule=lambda m, t: m.Qpump2[t] == p.Qw2(m.Ppump2[t]))

        # eqn:pv-limit
        model.P_PV = Var(model.t, bounds=lambda m, t: (0, float(p.P_PVavail[t])), initialize=initializer(i.P_PV))

        # eqn:power-balance-PV
        model.P_PV_inverter = Expression(model.t, rule=lambda m, t: m.P_PV[t] - m.Ppump2[t])

        # eqn:BSS-mode-charging2
        model.P_BSSch = Var(model.t, bounds=(0, p.P_BSS_ch_max), initialize=initializer(i.P_BSSch))

        model.P_BSSdisch = Var(model.t, bounds=(0, p.P_BSS_disch_max), initialize=initializer(i.P_BSSdisch))

        # eqn:inverter-mode
        def sinv_rule(m, t):
            if t >= 1:
                sinv_prev = m.sinv[t - 1]
                E_BSS_prev = m.E_BSS[t - 1]
                below_upper_thresh = linear_le_zero(E_BSS_prev - p.E_BSS_upper)
                stay_on_utility = linear_and(sinv_prev, below_upper_thresh)
                switch_to_utility = linear_le_zero(E_BSS_prev - p.E_BSS_lower)
                sinv_curr = linear_or(stay_on_utility, switch_to_utility)
            else:
                # In first period, previous values are constants, not expressions
                sinv_prev = p.sinv0
                E_BSS_prev = p.E_BSS0
                below_upper_thresh = (E_BSS_prev - p.E_BSS_upper <= 0)
                stay_on_utility = sinv_prev * below_upper_thresh
                switch_to_utility = 1. if E_BSS_prev <= p.E_BSS_lower else 0.
                sinv_curr = 1. if stay_on_utility or switch_to_utility else 0.
            return sinv_curr

        model.sinv = Expression(model.t, rule=sinv_rule)

        model.P_avail_to_charge = Expression(model.t, rule=lambda m, t:
            float(p.P_PVavail[t]) - m.Ppump2[t] - (1 - m.sinv[t]) * float(p.Pload[t]))
        model.sBSS = Expression(model.t, rule=lambda m, t: linear_ge_zero(m.P_avail_to_charge[t]))

        # eqn:BSS-mode-charging2
        def P_BSS_ch_rule(m, t):
            E_BSS_prev = m.E_BSS[t - 1] if t >= 1 else p.E_BSS0
            absorption_mode_limit = p.K_BSS * (p.E_BSS_max - E_BSS_prev) / (p.eta_BSS * p.delta_t)
            return absorption_mode_limit

        model.BSS_mode_charging1 = Expression(model.t, rule=P_BSS_ch_rule)

        # eqn:BSS-mode-charging3
        model.BSS_mode_charging3 = Expression(model.t, rule=lambda m, t: linear_maxn(m.P_avail_to_charge[t], 0))

        # eqn:????
        model.BSS_mode_charging4 = Constraint(model.t,
                                              rule=lambda m, t: m.P_BSSch[t] == linear_minn(m.BSS_mode_charging1[t],
                                                                                            m.BSS_mode_charging3[t],
                                                                                            m.sBSS[t] * p.P_BSS_ch_max))

        # eqn:BSS-mode-discharging
        model.BSS_mode_discharging = Constraint(model.t, rule=lambda m, t:
            m.P_BSSdisch[t] <= (1 - m.sBSS[t]) * p.P_BSS_disch_max)

        # eqn:power-balance-inverter-grid
        model.Pgrid_inverter = Expression(model.t,
                                          rule=lambda m, t: m.sinv[t] * float(p.Pload[t]))

        # eqn:power-balance-grid
        model.Pgrid = Expression(model.t,
                                 rule=lambda m, t: m.Pgrid_inverter[t] + m.Ppump1[t])

        # eqn:BSS-balance
        model.E_BSS_eq = Constraint(model.t, rule=lambda m, t: m.E_BSS[t] == p.E_BSS0 + quicksum(
            m.P_BSSch[k] * p.eta_BSS - m.P_BSSdisch[k] / p.eta_BSS
            for k in range(t + 1)) * p.delta_t)

        # eqn:water-balance-1
        model.Vw1 = Expression(model.t, rule=lambda m, t: p.Vw1_0 + quicksum(
            m.Qpump1[k] - m.Quse1[k] for k in range(t + 1)) * p.delta_t)
        # eqn:water-balance-2
        model.Vw2 = Expression(model.t, rule=lambda m, t: p.Vw2_0 + quicksum(
            m.Qpump2[k] - m.Quse2[k] for k in range(t + 1)) * p.delta_t)

        # eqn:total-water
        model.Vuse = Expression(model.d, rule=lambda m, d:
            quicksum(p.eta_w[t]*(model.Quse1[t] + model.Quse2[t]) for t in p.D[d])*p.delta_t)

        # eqn:power-balance-inverter
        model.power_balance_pv = Constraint(model.t,
                                            rule=lambda m, t: m.P_PV_inverter[t] + m.P_BSSdisch[t] - m.P_BSSch[t]
                                                              - (1 - m.sinv[t]) * float(p.Pload[t]) == 0)
        # eqn:grid-inverter-positive
        model.pgrid_inverter_pos = Constraint(model.t, rule=lambda m, t: m.Pgrid_inverter[t] >= 0)

        # eqn:pv-inverter-positive
        model.p_PV_inverter_pos = Constraint(model.t, rule=lambda m, t: m.P_PV_inverter[t] >= 0)

        # eqn:var-limit
        model.Vw1_min = Constraint(model.t,
                                   rule=lambda m, t: p.Vw1_min <= m.Vw1[t])
        model.Vw1_max = Constraint(model.t,
                                   rule=lambda m, t: m.Vw1[t] <= p.Vw1_max)
        model.Vw2_min = Constraint(model.t,
                                   rule=lambda m, t: p.Vw2_min <= m.Vw2[t])
        model.Vw2_max = Constraint(model.t,
                                   rule=lambda m, t: m.Vw2[t] <= p.Vw2_max)

        # Objective Function
        # eqn:objective-function
        model.grid_energy_cost = Expression(expr=quicksum(p.Cgrid[t] * model.Pgrid[t] * p.delta_t for t in model.t))
        model.battery_use_cost = Expression(
            expr=quicksum(p.C_BSS * (model.P_BSSch[t] + model.P_BSSdisch[t]) * p.delta_t for t in model.t))
        model.inadequate_water = Expression(expr=quicksum(linear_maxn(p.Vuse_desired[d] - model.Vuse[d], 0) for d in model.d))
        model.inadequate_water_cost = Expression(expr=p.Cw_short * model.inadequate_water)
        model.BSS_mode_switching_cost = Expression(expr=p.C_BSS_switching * num_switches(model.sBSS))
        model.pump_switching_cost = Expression(expr=p.C_pump_switching * num_switches(model.spump1))
        model.TOTAL_COST = Expression(expr=model.grid_energy_cost
                                           + model.battery_use_cost
                                           + model.inadequate_water_cost
                                           + model.BSS_mode_switching_cost
                                           + model.pump_switching_cost)

        model.obj = pyo.Objective(expr=model.TOTAL_COST)

        # Assign at end of function to ensure that the data is complete before being assigned
        self.model = model
        self.initialized_operation_scalars, self.initialized_operation = self.get_values()
        self.optimal_operation_scalars, self.optimal_operation = None, None

    def optimize(self, time_limit=5 * 60, solver_select='scip'):
        model = self.model
        p = self.params
        self.solver_output = None

        pre_optimization_obj_value = evaluate_expression(model.obj)

        pre_optimization = [f'Starting solution costs:',
                            f'    Grid energy cost: {evaluate_expression(model.grid_energy_cost):.1f}',
                            f'    Battery use cost: {evaluate_expression(model.battery_use_cost):.1f}',
                            f'    Inadequate water cost: {evaluate_expression(model.inadequate_water_cost):.1f}'
                                f' ({evaluate_expression(model.inadequate_water) / sum(p.Vuse_desired) * 100:.1f}% of desired water not supplied)',
                            f'    Battery mode switching cost: {evaluate_expression(model.BSS_mode_switching_cost):.1f}',
                            f'    Pump switching cost: {evaluate_expression(model.pump_switching_cost):.1f}',
                            f'    TOTAL: {pre_optimization_obj_value:.1f}']

        for l in pre_optimization:
            logger.info(l)

        # Check if starting solution is feasible
        log_infeasible_constraints(model)

        # Solve
        logger.info(f'Solving with solver_select={solver_select}')

        if solver_select == 'bonmin':
            # conda install -c conda-forge cyipopt
            # conda install -c conda-forge glpk
            # conda install -c conda-forge coinbonmin

            solver = pyo.SolverFactory('bonmin')
            solver.options['bonmin.algorithm'] = 'B-BB'
            solver.options['bonmin.time_limit'] = time_limit
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver.solve(model, tee=True)

        elif solver_select == 'mindtpy':
            solver = pyo.SolverFactory('mindtpy')
            solver.options = dict()
            solver.options['mip_solver'] = 'glpk'
            solver.options['nlp_solver'] = 'ipopt'
            solver.options['strategy'] = 'OA'
            solver.options['init_strategy'] = 'rNLP'
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver.solve(model, tee=True, **solver.options)

        elif solver_select == 'neos-cplex':
            import os
            os.environ['NEOS_EMAIL'] = neos_email_address
            solver_manager = pyo.SolverManagerFactory('neos')
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver_manager.solve(model, opt='cplex', keepfiles=True, tee=True)

        elif solver_select == 'neos-cbc':
            import os
            os.environ['NEOS_EMAIL'] = neos_email_address
            solver_manager = pyo.SolverManagerFactory('neos')
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver_manager.solve(model, opt='cbc', keepfiles=True, tee=True)

        elif solver_select == 'neos-minto':
            import os
            os.environ['NEOS_EMAIL'] = neos_email_address
            solver_manager = pyo.SolverManagerFactory('neos')
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver_manager.solve(model, opt='minto', keepfiles=True, tee=True)

        elif solver_select == 'py':
            solver = pyo.SolverFactory('py')
            results = solver.solve(model)

        elif solver_select == 'cbc':
            # Print linear terms until a nonlinear one is found and raises and exception that we can look into
            # for t in pyomo.core.expr.current._decompose_linear_terms(model.grid_energy_cost):
            #    t.pprint()
            # model_linearized = pyo.TransformationFactory('contrib.induced_linearity').create_using(model)
            solver = pyo.SolverFactory('cbc')
            solver.options['Seconds'] = time_limit
            solver.options['RatioGap'] = 0.01
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver.solve(model, tee=True, keepfiles=True)

        elif solver_select == 'scip':
            # Print linear terms until a nonlinear one is found and raises and exception that we can look into
            # for t in pyomo.core.expr.current._decompose_linear_terms(model.grid_energy_cost):
            #    t.pprint()
            # model_linearized = pyo.TransformationFactory('contrib.induced_linearity').create_using(model)
            solver = pyo.SolverFactory('scip')
            solver.options['limits/time'] = time_limit
            solver.options['RatioGap'] = 0.01
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver.solve(model, tee=True, keepfiles=True)

        elif solver_select == 'glpk':
            # Print linear terms until a nonlinear one is found and raises and exception that we can look into
            # for t in pyomo.core.expr.current._decompose_linear_terms(model.grid_energy_cost):
            #    t.pprint()
            # model_linearized = pyo.TransformationFactory('contrib.induced_linearity').create_using(model)
            solver = pyo.SolverFactory('glpk')
            solver.options['tmlim'] = time_limit
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = solver.solve(model, tee=True, keepfiles=True)
        else:
            raise ValueError(f'Invalid solver selection: {solver_select}')
        # SolverFactory('gdpopt').solve(model, strategy='LOA')

        #if results.Solver[0]['Termination condition'] == 'infeasible':
        log_infeasible_constraints(model)
        log_infeasible_bounds(model)
        #     log_close_to_bounds(model)

        # model.obj.display()
        # model.display()
        # model.pprint()

        self.solver_output = results

        post_optimization_obj_value = evaluate_expression(model.obj)
        post_optimization = ['Optimal solution costs:',
                             f'    Grid energy cost: {evaluate_expression(model.grid_energy_cost):.1f}',
                             f'    Battery use cost: {evaluate_expression(model.battery_use_cost):.1f}',
                             f'    Inadequate water cost: {evaluate_expression(model.inadequate_water_cost):.1f}'
                                f' ({evaluate_expression(model.inadequate_water) / sum(p.Vuse_desired) * 100:.1f}% of desired water not supplied)',
                             f'    Battery mode switching cost: {evaluate_expression(model.BSS_mode_switching_cost):.1f}',
                             f'    Pump switching cost: {evaluate_expression(model.pump_switching_cost):.1f}',
                             f'    TOTAL: {post_optimization_obj_value:.1f}']

        for l in itertools.chain(post_optimization):
            logger.info(l)
        logger.info(f'Objective value change: {pre_optimization_obj_value:.1f} -> {post_optimization_obj_value:.1f}')
        logger.info(f'Ratio of optimal / initial: {post_optimization_obj_value/pre_optimization_obj_value:.2f}')


    def get_values(self):
        scalar_vars, array_vars = vars_to_df(self.model)
        array_vars = array_vars[:self.params.lookahead]
        # Convert binary columns to boolean values
        array_vars = array_vars.astype({'spump1': np.bool, 'sinv': np.bool, 'sBSS': np.bool})
        # Add some other columns of interest
        array_vars['Vuse_desired'] = pd.Series(self.params.Vuse_desired)
        array_vars['Pload'] = pd.Series(self.params.Pload)
        array_vars['P_PVavail'] = pd.Series(self.params.P_PVavail)
        array_vars.index = array_vars.index*self.params.delta_t

        return scalar_vars, array_vars

    def get_optimal(self):
        self.optimal_operation_scalars, self.optimal_operation = self.get_values()
        return self.optimal_operation

    def plot_optimal(self):
        o = self.optimal_operation
        o[['P_BSSch', 'P_BSSdisch', 'Ppump1', 'Ppump2', 'Pload', 'P_PV']].plot(drawstyle='steps')
        o[['E_BSS']].plot()
        o[['Qpump1', 'Quse1', 'Qpump2', 'Quse2']].plot(drawstyle='steps')
        o[['Vw1', 'Vw2']].plot()
        o[['sinv', 'sBSS', 'spump1']].plot(include_bool=True, drawstyle='steps', subplots=True)
        return

    def do_all(self, optimize=True, plots=False):
        self.initialize()
        self.build_model()
        if optimize:
            self.optimize(solver_select='cbc')
            # Sometimes CBC seems to decide the problem is infeasible when GLPK doesn't. No idea why.
            # If this happens, try solving with glpk instead, then try CBC a second time.
            if self.solver_output.Solver[0]['Termination condition'] == 'infeasible':
                self.optimize(solver_select='glpk')
                self.optimize(solver_select='cbc')
        self.get_optimal()
        if plots:
            self.plot_optimal()

    def log_infeasible_constraints(self):
        log_infeasible_constraints(self.model)
        log_infeasible_bounds(self.model)



