
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class RatesCurvesConfig:
    file_path: str
    dates: tuple[str, ...] = ('1/31/2024', '1/31/2022')
    instrument_col: str = 'Instrument'
    maturity_col: str = 'Maturity (yrs)'
    fra_label: str = 'FRA'
    irs_label: str = 'IRS'


class RatesCurves:
    def __init__(self, config: RatesCurvesConfig):
        self.cfg = config


    def load_data(file_path):
    # Load and preprocess data
        data = pd.read_excel(file_path, header=2)
        data.columns = data.columns.str.strip()
        data = data.dropna(axis=1, how='all')
        data.columns.values[-2] = '1/31/2024'
        data.columns.values[-1] = '1/31/2022'
        return data

    def zero_rate_FRA(self, FRA_data, column):
        # Calculate zero rates from FRA data (unchanged)
        zero_rates = []
        maturities = FRA_data[self.cfg.maturity_col].values
        fra_rates = FRA_data[column].values

        for i, fra_rate in enumerate(fra_rates):
            if maturities[i] == 0.5:
                r_0_5 = (1 / (1 - fra_rate * 0.5)) - 1
                zero_rates.append(r_0_5)
            else:
                r_prev = zero_rates[-1]
                r_i = ((1 + r_prev) * maturities[i] / (1 - fra_rate * maturities[i])) * (1 / maturities[i]) - 1
                zero_rates.append(r_i)

        return zero_rates

    def zero_rate_IRS(self, IRS_data, column):
        # Bootstrap zero rates from IRS data (unchanged)
        zero_rates = []
        maturities = IRS_data[self.cfg.maturity_col].values
        swap_rates = IRS_data[column].values

        for i, swap_rate in enumerate(swap_rates):
            T = int(maturities[i])
            if i == 0:
                y_K = swap_rate
            else:
                P_T = 1 / (1 + swap_rate) ** T
                sum_P = sum([1 / (1 + zero_rates[j]) ** (j + 1) for j in range(min(T, len(zero_rates)))])
                y_K = (1 - P_T) / sum_P
            zero_rates.append(y_K)

        return zero_rates

    def forward_rates(self, zero_rates):
        # Calculate forward rates from zero rates (unchanged)
        fwd_rates = []
        for i in range(1, len(zero_rates)):
            fwd_rate = ((1 + zero_rates[i]) * (i + 1) / (1 + zero_rates[i - 1]) * i) - 1
            fwd_rates.append(fwd_rate)
        return fwd_rates

    def calculate_swap_value(self, zero_rates, swap_rate, maturity):
        # Calculate swap value (unchanged)
        fixed_leg_value = sum([swap_rate * (1 / (1 + zero_rates[i]) ** (i + 1)) for i in range(maturity)])
        floating_leg_value = 1 - (1 / (1 + zero_rates[maturity - 1]) ** maturity)
        return fixed_leg_value, floating_leg_value

    def generate_curves(self, file_path):
        # Generate zero and forward rate curves (unchanged structure)
        data = self.load_data(file_path)
        FRA_data = data[data[self.cfg.instrument_col] == self.cfg.fra_label]
        IRS_data = data[data[self.cfg.instrument_col] == self.cfg.irs_label]

        results = {}
        for date in self.cfg.dates:
            fra_zero_rates = self.zero_rate_FRA(FRA_data, date)
            irs_zero_rates = self.zero_rate_IRS(IRS_data, date)
            results[date] = {
                'FRA': {'Zero Rates': fra_zero_rates, 'Forward Rates': self.forward_rates(fra_zero_rates)},
                'IRS': {'Zero Rates': irs_zero_rates, 'Forward Rates': self.forward_rates(irs_zero_rates)},
            }

        return results, IRS_data, FRA_data

    def plot_curves(self, results, FRA_data, IRS_data):
        # Plot FRA curves (unchanged)
        plt.figure(figsize=(12, 5))
        maturities = FRA_data[self.cfg.maturity_col].values
        for i, (key, curve_type) in enumerate(zip(['Zero Rates', 'Forward Rates'], ['Zero Rate', 'Forward Rate']), 1):
            plt.subplot(1, 2, i)
            for date, data in results.items():
                y_values = data['FRA'][key]
                if key == 'Forward Rates':
                    plt.plot(maturities[:-1], y_values, label=f'FRA {curve_type} {date}')
                else:
                    plt.plot(maturities, y_values, label=f'FRA {curve_type} {date}')
            plt.title(f'FRA {curve_type} Curves')
            plt.xlabel('Maturity (years)')
            plt.ylabel(curve_type)
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot IRS curves (unchanged)
        plt.figure(figsize=(12, 5))
        maturities = IRS_data[self.cfg.maturity_col].values
        for i, (key, curve_type) in enumerate(zip(['Zero Rates', 'Forward Rates'], ['Zero Rate', 'Forward Rate']), 1):
            plt.subplot(1, 2, i)
            for date, data in results.items():
                y_values = data['IRS'][key]
                if key == 'Forward Rates':
                    plt.plot(maturities[:-1], y_values, label=f'IRS {curve_type} {date}')
                else:
                    plt.plot(maturities, y_values, label=f'IRS {curve_type} {date}')
            plt.title(f'IRS {curve_type} Curves')
            plt.xlabel('Maturity (years)')
            plt.ylabel(curve_type)
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    def calculate_swap_value_20y(self, zero_rates, swap_rate, maturity=20):
        # Calculate 20-year swap value (unchanged)
        fixed_leg_value = sum([1 / (1 + zero_rates[i]) ** (i + 1) for i in range(maturity)]) * swap_rate
        floating_leg_value = 1 - (1 / (1 + zero_rates[maturity - 1]) ** maturity)
        net_swap_value = fixed_leg_value - floating_leg_value
        return net_swap_value, fixed_leg_value, floating_leg_value
