"""Conditional capacitor energy model. Python 3.10+; standard library only.

All scenario parameters are assumptions, not measurements of a built device.
The source power is hypothetical sustained electrical power BEFORE input
conversion. Conversion and voltage compatibility are assumed; cold start is
outside this model. See README.md and the report for limitations.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import csv
import json
import math


@dataclass(frozen=True)
class Scenario:
    capacitance_f: float = 100e-6
    source_power_w: float = 4e-6
    input_efficiency: float = 0.50
    standby_power_w: float = 0.5e-6
    leakage_resistance_ohm: float = 10e6
    low_voltage_v: float = 2.0
    high_voltage_v: float = 3.3
    event_energy_j: float = 100e-6
    output_efficiency: float = 0.80

    def __post_init__(self):
        finite = [self.capacitance_f, self.source_power_w,
                  self.input_efficiency, self.standby_power_w,
                  self.low_voltage_v, self.high_voltage_v,
                  self.event_energy_j, self.output_efficiency]
        if not all(math.isfinite(x) for x in finite):
            raise ValueError('All parameters except leakage resistance must be finite.')
        if self.capacitance_f <= 0 or self.source_power_w < 0 or self.standby_power_w < 0:
            raise ValueError('Capacitance must be positive and powers nonnegative.')
        if not 0 < self.input_efficiency <= 1 or not 0 < self.output_efficiency <= 1:
            raise ValueError('Efficiencies must lie in (0, 1].')
        if not 0 <= self.low_voltage_v < self.high_voltage_v or self.event_energy_j <= 0:
            raise ValueError('Require 0 <= low < high and positive event energy.')
        if math.isnan(self.leakage_resistance_ohm) or self.leakage_resistance_ohm <= 0:
            raise ValueError('Leakage resistance must be positive or infinity.')

    @property
    def surplus_w(self):
        return self.input_efficiency * self.source_power_w - self.standby_power_w

    @property
    def leakage_at_high_w(self):
        return self.high_voltage_v ** 2 / self.leakage_resistance_ohm

    @property
    def critical_source_power_w(self):
        return (self.standby_power_w + self.leakage_at_high_w) / self.input_efficiency

    @property
    def usable_event_energy_j(self):
        return self.output_efficiency * 0.5 * self.capacitance_f * (
            self.high_voltage_v ** 2 - self.low_voltage_v ** 2)

    @property
    def minimum_capacitance_f(self):
        return 2 * self.event_energy_j / (self.output_efficiency * (
            self.high_voltage_v ** 2 - self.low_voltage_v ** 2))

    @property
    def equilibrium_voltage_v(self):
        if self.surplus_w <= 0:
            return 0.0
        return math.sqrt(self.surplus_w * self.leakage_resistance_ohm)

    def recharge_time_s(self, start_voltage_v=None):
        """Time to high threshold after already-started operation; infinity if unreachable."""
        start = self.low_voltage_v if start_voltage_v is None else start_voltage_v
        if not math.isfinite(start) or not 0 <= start <= self.high_voltage_v:
            raise ValueError('Starting voltage must be finite and between zero and high.')
        if start == self.high_voltage_v:
            return 0.0
        # Strict inequality: at the boundary the threshold is approached only asymptotically.
        if self.surplus_w <= self.leakage_at_high_w or math.isclose(
                self.surplus_w, self.leakage_at_high_w, rel_tol=1e-12, abs_tol=0.0):
            return math.inf
        delta_e = 0.5 * self.capacitance_f * (self.high_voltage_v ** 2 - start ** 2)
        if math.isinf(self.leakage_resistance_ohm):
            return delta_e / self.surplus_w
        veq2 = self.surplus_w * self.leakage_resistance_ohm
        # log1p avoids cancellation when leakage is very small.
        return self.capacitance_f * self.leakage_resistance_ohm / 2 * math.log1p(
            (self.high_voltage_v ** 2 - start ** 2) / (veq2 - self.high_voltage_v ** 2))

    def voltage_at_s(self, time_s, start_voltage_v=None):
        """Uncontrolled charging trajectory; no event switching or upper clamp."""
        start = self.low_voltage_v if start_voltage_v is None else start_voltage_v
        if not math.isfinite(time_s) or time_s < 0 or not math.isfinite(start) or start < 0:
            raise ValueError('Time and starting voltage must be finite and nonnegative.')
        e0 = 0.5 * self.capacitance_f * start ** 2
        if math.isinf(self.leakage_resistance_ohm):
            energy = e0 + self.surplus_w * time_s
        else:
            tau_e = self.leakage_resistance_ohm * self.capacitance_f / 2
            eq = self.surplus_w * tau_e
            energy = e0 + (eq - e0) * (-math.expm1(-time_s / tau_e))
        # Zero is a boundary of the ideal energy model; below the load's low
        # threshold the actual circuit would need a different shutdown model.
        return math.sqrt(2 * max(0.0, energy) / self.capacitance_f)

    def post_event_voltage_v(self):
        if self.event_energy_j > self.usable_event_energy_j + 1e-15:
            return None
        return math.sqrt(max(0.0, self.high_voltage_v ** 2 -
            2 * self.event_energy_j / (self.output_efficiency * self.capacitance_f)))

    def fixed_event_recharge_time_s(self):
        after = self.post_event_voltage_v()
        return None if after is None else self.recharge_time_s(after)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def export_results(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = Scenario()
    summary = {
        'status': 'Illustrative conditional model; no experimental validation',
        'baseline_parameters_SI': asdict(baseline),
        'critical_source_power_uW': baseline.critical_source_power_w * 1e6,
        'equilibrium_voltage_V': baseline.equilibrium_voltage_v,
        'full_window_recharge_s': baseline.recharge_time_s(),
        'fixed_100uJ_event_recharge_s': baseline.fixed_event_recharge_time_s(),
        'post_event_voltage_V': baseline.post_event_voltage_v(),
        'usable_event_energy_uJ': baseline.usable_event_energy_j * 1e6,
        'minimum_capacitance_uF': baseline.minimum_capacitance_f * 1e6,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    power_rows = []
    for resistance in [1e6, 10e6, 100e6, math.inf]:
        for power_uW in [0.5, 1, 2, 3, 3.178, 4, 8, 8.452, 16]:
            s = replace(baseline, source_power_w=power_uW*1e-6,
                        leakage_resistance_ohm=resistance)
            t = s.recharge_time_s()
            power_rows.append({'source_power_uW':power_uW,
                'leakage_resistance_Mohm':resistance/1e6,
                'critical_source_power_uW':s.critical_source_power_w*1e6,
                'energy_threshold_reachable':math.isfinite(t),
                'full_window_recharge_s':t,
                'fixed_100uJ_event_recharge_s':s.fixed_event_recharge_time_s(),
                'provenance':'assumed sustained power, not a measured DC output'})
    write_csv(out/'power_sweep.csv', power_rows)
    cap_rows=[]
    for c_uF in [10, 22, 47, 100, 220, 470, 1000]:
        s=replace(baseline, capacitance_f=c_uF*1e-6)
        cap_rows.append({'capacitance_uF':c_uF,
            'usable_event_energy_uJ':s.usable_event_energy_j*1e6,
            'can_supply_100uJ':s.post_event_voltage_v() is not None,
            'full_window_recharge_s':s.recharge_time_s(),
            'post_event_voltage_V':s.post_event_voltage_v(),
            'fixed_100uJ_event_recharge_s':s.fixed_event_recharge_time_s(),
            'leakage_resistance_fixed_Mohm':10})
    write_csv(out/'capacitance_sweep.csv', cap_rows)
    trajectory=[]
    for p in [2, 4, 8]:
        s=replace(baseline, source_power_w=p*1e-6)
        for t in range(0, 1201, 5):
            trajectory.append({'time_s':t,'source_power_uW':p,
                'voltage_V':s.voltage_at_s(t),'event_switching':'disabled for equilibrium illustration'})
    write_csv(out/'charging_trajectories.csv',trajectory)
    losses=[]
    for eta in [.25,.5,.75]:
        for q in [.1,.5,1.0]:
            s=replace(baseline,input_efficiency=eta,standby_power_w=q*1e-6)
            losses.append({'input_efficiency':eta,'standby_power_uW':q,
                'critical_source_power_uW':s.critical_source_power_w*1e6,
                'full_window_recharge_s_at_4uW':s.recharge_time_s(),
                'assumption':'independent illustrative sensitivity inputs'})
    write_csv(out/'loss_sweep.csv',losses)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).with_name('results')))
    args=parser.parse_args()
    print(json.dumps(export_results(args.output),indent=2))
