from pathlib import Path
from datetime import datetime
import logging.config


def postpend(filename, string):
    filepath = Path(filename)
    return filepath.parent / (filepath.stem + string + filepath.suffix)


program_runtime = datetime.now()
results_dir = Path('output') / 'operational_optimization' / f'{program_runtime:%Y%m%d_%H%M}'
results_dir.mkdir()

logfile = results_dir / 'optimization_demo.log'
logfile_debug = results_dir / 'optimization_demo_debug.log'


class LogOnlyLevel(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, log_record):
        return log_record.levelno == self.__level


logging_config = {
    'version': 1,
    'formatters': {
        'file': {'format': '%(asctime)s %(levelname)-8s %(name)s: %(message)s'},
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': logfile,
            'formatter': 'file',
            'level': 'INFO'},
        'file_debug': {
            'class': 'logging.FileHandler',
            'filename': logfile_debug,
            'formatter': 'file',
            'level': 'DEBUG'},
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'file',
            'level': 'DEBUG'}
    },
    'loggers': {
        '': {'handlers': ['file', 'console', 'file_debug'],
             'level': 'WARNING'},
        'ems': {'level': 'DEBUG'},
        'pyomo': {'level': 'INFO'},
        'optimization_demo': {'level': 'DEBUG'},
    },
    'filters': {
        'debug_only': {
            '()': LogOnlyLevel,
            'level': logging.DEBUG
        }
    }
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger('optimization_demo')

import pickle
import random
from typing import Optional
from types import SimpleNamespace
from itertools import groupby
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from ems.optimization_two_pumps import TwoPumpModel, TwoPumpModelParameters
import pandas as pd
#matplotlib.use('TkAgg')
#plt.ion() # Set matplotlib to interactive so plots don't block
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times'],
                     'legend.framealpha': 0.8,
                     'legend.frameon': True})
plt.rc('font', size=6)

# In order to reduce the code & data dependencies needed for the paper results to be reproducible,
# the data from the ABB inverter dataset is saved to a csv file for the window of time that was used.
# Code to export the data from the dataset is commented out below.
## How to define db connection and load ABB inverter dataset:
# import sqlalchemy
# from ems.datasets import ABBInverterDataSet
# engine = sqlalchemy.create_engine("sqlite+pysqlite:///tmp.sqlite", echo=False, future=False)
# ds_abb = ABBInverterDataSet(db_engine=engine)
# ds_abb.import_new_data()
# df = ds_abb.get_data_by_date(start='2/21/2021', end='2/28/2021')['P_out'] * 7.5e3
# df.to_csv('data/operational_optimization_pv_out.csv', header=True)

# Load PV dataset from csv file
pv_data = pd.read_csv('data/operational_optimization_pv_out.csv', parse_dates=['dt'], index_col='dt')


def time_of_use_rate(h):
    """
    # Electric rates by time of day.
    # From Güneşköy spreadsheet with bill summaries from 12/2019 to 10/2020
    """
    rate = {6: 0.39,
            17: 0.64,
            22: 0.95,
            24: 0.39}

    for hmax, r in rate.items():
        if h < hmax:
            return r
    return r


def time_of_day_water_efficiency(h : pd.DataFrame):
    h = np.squeeze(h.values)
    eta_w = np.array([[ 0, 1.0],
                      [ 6, 1.0],
                      [10, 0.8],
                      [14, 0.5],
                      [18, 0.7],
                      [22, 1.0],
                      [24, 1.0]]).T

    eta_wt = np.interp(h, eta_w[0], eta_w[1])
    return eta_wt


def multi_day_demo(data : SimpleNamespace, final_idx=23):
    local_hour = pd.DataFrame(data.available_pv.tz_localize('UTC').tz_convert('Europe/Istanbul').index.hour,
                              index=data.available_pv.index)
    electric_rates = local_hour.applymap(time_of_use_rate)
    electric_rates /= 7.0  # Convert from TL to USD based on approximate exchange rate at the time

    # Model metaparameters
    days = 3
    lookahead = days*24

    p = TwoPumpModelParameters(

        lookahead=lookahead,
        delta_t=1,
        D=tuple(tuple(g) for k, g in groupby(range(lookahead), lambda t: t // 24)),

        # Model data
        Pload=data.Pload.values,
        P_PVavail=data.available_pv.values,

        Ppump1_max=15000.,
        Qw1=50.,

        Ppump2_max=2200.,
        Ppump2_min_pu=0.3,
        eta_pump=0.4,
        head=41.,
        rho=1000.,
        g=9.81,

        E_BSS_max=9600.,
        E_BSS_lower_pu=0.1,
        E_BSS_upper_pu=0.95,
        E_BSS0_pu=data.E_BSS0_pu,
        sinv0=data.sinv0,
        P_BSS_ch_max_pu=1 / 10,
        P_BSS_disch_max_pu=1 / 3,
        eta_BSS=0.95,
        E_absorb_pu=0.8,

        Vw1_min=5.,
        Vw1_max=120.,
        Vw1_0=data.Vw1_0,
        Vw2_min=5.,
        Vw2_max=50.,
        Vw2_0=data.Vw2_0,

        Vuse_desired=data.Vuse_desired.values,
        Quse1_max=50.,
        Quse2_max=50.,

        Cgrid=electric_rates.values,
        C_BSS=0.01,
        C_BSS_switching=1.,
        C_pump_switching=1.,

        Cw_short=100.,

        eta_w=time_of_day_water_efficiency(local_hour)
    )

    figs = []

    m = TwoPumpModel(p)
    m.index = data.available_pv.index
    m.initialize()
    m.build_model()
    m.initialized_operation.index = m.index

    fname_base = results_dir / str(data.seed) / f'{m.index[0]:%Y%m%d_%H%M}'
    with open(postpend(fname_base, '_params.pkl'), 'wb') as f:
        pickle.dump(p, f)
    m.initialized_operation.to_pickle(postpend(fname_base, '_initialized.pkl'))
    m.initialized_operation_scalars.to_pickle(postpend(fname_base, '_initialized_scalars.pkl'))
    fig_initialized = plot_optimization_results(m, initialized_operation=True)
    figs.append(fig_initialized)

    # Sometimes CBC seems to decide the problem is infeasible when other solvers don't. No idea why.
    #m.optimize(solver_select='cbc', time_limit=30)
    m.optimize(solver_select='scip')
    m.get_optimal()

    data.E_BSS_final_pu = m.optimal_operation.at[final_idx, 'E_BSS'] / p.E_BSS_max
    data.sinv_final = m.optimal_operation.at[final_idx, 'sinv']
    data.Vw1_final = m.optimal_operation.at[final_idx, 'Vw1']
    data.Vw2_final = m.optimal_operation.at[final_idx, 'Vw2']

    m.optimal_operation.index = m.index

    m.optimal_operation.to_pickle(postpend(fname_base, '_optimized.pkl'))
    m.optimal_operation_scalars.to_pickle(postpend(fname_base, '_optimized_scalars.pkl'))

    fig_optimized = plot_optimization_results(m)
    figs.append(fig_optimized)

    return m, data, figs


def plot_optimization_results(m, initialized_operation=False):

    if not initialized_operation:
        o = m.optimal_operation
    else:
        o = m.initialized_operation

    fig = plt.figure(figsize=(10.64, 4.40))

    # Manually setting the axis positions so that I can tweak each plot size independently
    # without messing up the position of everything on the page.
    l_margin = 0.04
    r_margin = 0.015
    t_margin = 0.04
    b_margin = 0.08
    ax_width = 1/3 - l_margin - r_margin
    ax_height = 1/2 - t_margin - b_margin
    ax = fig.add_axes([l_margin, 0.5 + b_margin, ax_width, ax_height]) # Position: [left, bottom, width, height]
    ax.set_title('PV-side Power')
    plot_data = pd.DataFrame()
    plot_data['PV available'] = o['P_PVavail']
    plot_data['PV utilized'] = o['P_PV']
    plot_data['BSS (discharge)'] = o['P_BSSdisch']
    plot_data['BSS (charge)'] = -1 * o['P_BSSch']
    plot_data['Load'] = -1 * o['Pload'] * (1 - o['sinv'])
    plot_data['Pump 2'] = -1 * o['Ppump2']
    plot_data /= 1000
    ax = plot_data.plot(ax=ax, drawstyle='steps-post')
    ax.xaxis.grid(True, which='minor')
    #plt.ylim(top=0.9)
    ax.set_xlabel(None)
    ax.set_ylabel('Power (kW)')
    ax.legend(loc="upper right")

    ht_factor = 0.7
    ax = fig.add_axes([l_margin, b_margin + ax_height*(1 - 0.7), ax_width, ax_height*0.7])
    ax.set_title('Grid-side Power')
    plot_data = pd.DataFrame()
    plot_data['Grid'] = o['Pgrid']
    plot_data['Load'] = -1 * o['Pload'] * o['sinv']
    plot_data['Pump 1'] = -1 * o['Ppump1']
    plot_data /= 1000
    ax = plot_data.plot(ax=ax, drawstyle='steps-post')
    ax.set_xlabel(None)
    ax.set_ylabel('Power (kW)')
    ax.legend(loc="upper right")

    ht_factor = 0.7
    ax = fig.add_axes([1/3 + l_margin, 0.5 + b_margin + ax_height*(1 - ht_factor), ax_width, ax_height*ht_factor])
    ax.set_title('BSS Energy')
    plot_data = pd.DataFrame()
    plot_data['BSS Energy'] = o['E_BSS']
    start_time = plot_data.index[0]
    plot_data.index = plot_data.index + (plot_data.index[1] - plot_data.index[0])
    plot_data.loc[start_time, 'BSS Energy'] = m.params.E_BSS0
    plot_data = plot_data.sort_index()
    plot_data /= 1000

    ax = plot_data.plot(ax=ax)

    ax.set_xlabel(None)
    ax.set_ylabel('Energy (kW-h)')
    ax.legend(loc="upper right")

    ax = fig.add_axes([1/3 + l_margin, b_margin, ax_width, ax_height])
    ax.set_title('BSS Power')
    plot_data = pd.DataFrame()
    plot_data['Charging'] = o['P_BSSch']
    plot_data['Discharging'] = -1 * o['P_BSSdisch']
    plot_data /= 1000

    ax = plot_data.plot(ax=ax, drawstyle='steps-post')

    ax.set_xlabel(None)
    ax.set_ylabel('Power (kW)')
    ax.legend(loc="upper right")

    table_ht = 0.3
    ax_height = (1. - table_ht)/2 - b_margin - t_margin
    ax = fig.add_axes([2 / 3 + l_margin, table_ht + 2*b_margin + ax_height + t_margin, ax_width, ax_height])
    ax.set_title('Reservoir Water Volumes')
    plot_data = pd.DataFrame()
    plot_data['Res. 1'] = o['Vw1']
    plot_data['Res. 2'] = o['Vw2']

    start_time = plot_data.index[0]
    plot_data.index = plot_data.index + (plot_data.index[1] - plot_data.index[0])
    plot_data.loc[start_time, 'Res. 1'] = m.params.Vw1_0
    plot_data.loc[start_time, 'Res. 2'] = m.params.Vw2_0
    plot_data = plot_data.sort_index()

    ax = plot_data.plot(ax=ax)

    ax.set_xlabel(None)
    ax.set_ylabel('Water Volume (m^3)')

    ax.legend(loc="upper right")

    ax = fig.add_axes([2 / 3 + l_margin, table_ht + b_margin, ax_width, ax_height])
    ax.set_title('Water Usage')
    plot_data = pd.DataFrame()
    plot_data['Res. 1'] = o['Quse1']
    plot_data['Res. 2'] = o['Quse2']

    ax = plot_data.plot(ax=ax, drawstyle='steps-post')

    ax.set_xlabel(None)
    ax.set_ylabel('Water Use (m^3/h)')
    ax.legend(loc="upper right")

    ax = fig.add_axes([2 / 3 + l_margin, 0.03, ax_width, table_ht - 0.03])
    ax.set_axis_off()

    if initialized_operation:
        ax.text(0.75, 0.5, 'Plotting initialized values',
                ha='left', va='top', transform=ax.transAxes)
    else:
        table_data = pd.DataFrame({'Initialized': m.initialized_operation_scalars,
                                   'Optimized': m.optimal_operation_scalars}).round(1)
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                         rowLabels=table_data.index, colWidths=[0.25, 0.25], loc='center right')

    #plt.show(block=True)
    return fig


class ScenarioGenerator:
    """
    A callable class to return actuals for PV_avail, Pload, and Vuse_desired.
    Implemented as a class so that long sequence of randomly generated values can be kept between calls rather than
    regenerated every time. Values are cached for a single seed value only. If called with a different seed, new
    series of random values are generated.
    """
    def __init__(self, Pload_max=1e3, Vuse_mean=140., Vuse_std=60.):
        # Date range for generating random sequences. Must be log enough to include all start and end dates.
        # Changing the end date will not change the sequence except to make it longer.
        # Changing the start date WILL change the sequence, even for the same seed value.
        self.Pload_max = Pload_max
        self.Vuse_mean = Vuse_mean
        self.Vuse_std = Vuse_std
        self.start_date = pd.to_datetime('1/1/2010')
        self.end_date = pd.to_datetime('12/31/2025')
        self._seed = 143
        self.Pload : Optional[pd.DataFrame] = None
        self.Vuse_desired : Optional[pd.DataFrame] = None


    def seed(self, seed):
        self._seed = seed
        random.seed(seed)
        self.Pload_seed = random.randint(0, 10 ** 7)
        self.Vuse_seed = random.randint(0, 10 ** 7)
        self.other_values_seed = random.randint(0, 10 ** 7)
        self.generate_Pload()
        self.generate_Vuse_desired()

    def generate_Pload(self):
        # Generate Pload sequence using random numbers
        random.seed(self.Pload_seed, version=2)
        # Uniform distribution
        # Avoid use of numpy.random since it is not guaranteed to be consistent across versions. Use Python instead.
        num_points = int((self.end_date - self.start_date) / pd.to_timedelta('1h'))
        Pload_values = [random.uniform(0, self.Pload_max) for _ in range(num_points)]
        Pload = pd.Series(Pload_values,
                          index=pd.date_range(start=self.start_date, end=self.end_date, freq='1h', closed='left', tz=None))
        self.Pload = Pload

    def generate_Vuse_desired(self):
        # Generate Vuse_desired sequence using random numbers
        random.seed(self.Vuse_seed, version=2)
        # Normal distribution with given mean and standard deviation
        # Avoid use of numpy.random since it is not guaranteed to be consistent across versions. Use Python instead.
        num_points = int((self.end_date - self.start_date) / pd.to_timedelta('1d'))
        Vuse_values = [random.gauss(self.Vuse_mean, self.Vuse_std) for _ in range(num_points)]
        Vuse = pd.Series(Vuse_values,
                         index=pd.date_range(start=self.start_date, end=self.end_date, freq='1D', closed='left', tz=None))
        self.Vuse_desired = Vuse

    def __call__(self, start, end, seed=143) -> SimpleNamespace:
        """
        :param start: Pandas-compatible starting datetime or string
        :param end: Pandas-compatible ending datetime or string
        :param seed: All randomly generated data can be reproduced by calling with the same seed.
        :return: Namespace with fields for available_pv, Vuse_desired, and Pload.
        """
        if seed != self._seed:
            self.seed(seed)

        data = SimpleNamespace()
        data.seed = seed

        # Read PV_avail sequence from data file
        idx_start = pv_data.index.get_slice_bound(start, "left")
        idx_end = pv_data.index.get_slice_bound(end, "left")
        data.available_pv = pv_data['P_out'][idx_start:idx_end]
        # data.available_pv = ds_abb.get_data_by_date(start=start, end=end)['P_out']*7.5e3

        # Extract Pload and Vuse_desired from pre-generated random sequences
        data.Pload = self.Pload[start:end]
        data.Vuse_desired = self.Vuse_desired[start:end]

        # Set some other values
        random.seed(self.other_values_seed, version=2)
        data.E_BSS0_pu = random.uniform(0.1, 1.0)
        data.sinv0 = random.randint(0, 1)
        data.Vw1_0 = random.uniform(5., 120.)
        data.Vw2_0 = random.uniform(5., 50.)

        return data


scenario_generator = ScenarioGenerator(Pload_max=1e3, Vuse_mean=140., Vuse_std=60.)


def run_all_days(seed=42):
    start_date = pd.to_datetime('2/24/2021')
    end_date = pd.to_datetime('2/27/2021')
    lookahead = pd.to_timedelta('3d')

    (results_dir / str(seed)).mkdir()
    pdf_fname = results_dir / str(seed) / 'optimization_plots.pdf'

    with PdfPages(pdf_fname) as pdf:

        for optimization_start in pd.date_range(start=start_date, end=end_date - lookahead, freq='1D', tz=None):
            optimization_end = optimization_start + lookahead
            logger.info('='*60)
            logger.info(f'Beginning operational optimization')
            logger.info(f'   Period begins: {optimization_start}')
            logger.info(f'   Period ends:   {optimization_end}')

            data = scenario_generator(optimization_start, optimization_end, seed)

            logger.info(f'seed={data.seed}, E_BSS0_pu={data.E_BSS0_pu:.2f}, Vw1_0={data.Vw1_0:.1f}, '
                        f'Vw2_0={data.Vw2_0:.1f}, sinv0={data.sinv0}')

            m, data, figs = multi_day_demo(data, 23)
            logger.info(f'Saving plots to PDF at {pdf_fname}')
            for fig in figs:
                pdf.savefig(fig)
            plt.close('all')


def main():
    for seed in [42]:
        run_all_days(seed)


if __name__ == '__main__':
    main()
