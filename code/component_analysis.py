"""Datasheet-based energy-budget scenarios; not a verified circuit design."""
from pathlib import Path
from dataclasses import replace
import csv,json,math
from model import Scenario

ROOT=Path(__file__).resolve().parent

def calculate():
    rail_v=3.3;eta_out=.8;housekeeping=.25e-6
    # TI TMP117 SNOSD82D, p.6: inactive serial bus; 25 C shutdown values.
    typical_energy=rail_v*135e-6*15.5e-3
    limit_energy=rail_v*220e-6*17.5e-3
    # KYOCERA AVX X5R Rev 4, p.2: min(10,000 MOhm,500 MOhm-uF/C_uF).
    c_nominal=100e-6
    r_bound=min(10000e6,500e6/(c_nominal*1e6))
    rows=[]
    for label,current,energy in [('Typical',.15e-6,typical_energy),('Specified-limit',.5e-6,limit_energy)]:
        for pin in [4e-6,8e-6]:
            s=replace(Scenario(),source_power_w=pin,leakage_resistance_ohm=r_bound,
                event_energy_j=energy,standby_power_w=housekeeping+rail_v*current/eta_out)
            t=s.fixed_event_recharge_time_s()
            rows.append(dict(case=label,input_uW=pin*1e6,task_load_uJ=energy*1e6,
                task_storage_uJ=energy/eta_out*1e6,standby_storage_uW=s.standby_power_w*1e6,
                leakage_at_high_uW=s.leakage_at_high_w*1e6,
                required_source_uW=s.critical_source_power_w*1e6,
                remaining_power_at_high_uW=(s.surplus_w-s.leakage_at_high_w)*1e6,
                refill_s=None if math.isinf(t) else t,
                energy_capacity_uJ=s.usable_event_energy_j*1e6))
    summary=dict(typical_task_uJ=typical_energy*1e6,specified_limit_task_uJ=limit_energy*1e6,
        resistance_stress_Mohm=r_bound/1e6,nominal_C_uF=100,
        capacitance_sensitivity_uF=[50,80,100,120],
        available_load_energy_uJ=[eta_out*.5*c*1e-6*(3.3**2-2**2)*1e6 for c in [50,80,100,120]],
        sensor_power_active_typ_uW=rail_v*135e-6*1e6,
        sensor_power_active_limit_uW=rail_v*220e-6*1e6,
        neglected_incoming_per_active_window_typ_uJ=.5*8e-6*15.5e-3*1e6,
        rows=rows)
    return summary

def export():
    out=ROOT/'extended_results';s=calculate()
    (out/'component_summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    with (out/'component_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=s['rows'][0]);w.writeheader();w.writerows(s['rows'])
    return s

if __name__=='__main__':print(json.dumps(export(),indent=2))
