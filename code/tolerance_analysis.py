"""Deterministic effective-capacitance check; not a measured tolerance distribution."""
from dataclasses import replace
from pathlib import Path
import csv,json,math
from model import Scenario
from restart_analysis import simulate

ROOT=Path(__file__).resolve().parent

def calculate():
    capacity=[];service=[]
    for c in [80,90,99,100,110,120]:
        for vh in [2.55,2.8]:
            s=replace(Scenario(),capacitance_f=c*1e-6,high_voltage_v=vh)
            capacity.append(dict(C_uF=c,VH_V=vh,usable_uJ=s.usable_event_energy_j*1e6,
                                 task_feasible=s.usable_event_energy_j>=s.event_energy_j))
    for c in [80,90,100,110,120]:
        s=replace(Scenario(),capacitance_f=c*1e-6,high_voltage_v=2.8)
        for period in [None,1200]:
            r=simulate(period,0,False,s)
            service.append(dict(C_uF=c,VH_V=2.8,period_s=period,tasks_h=r['tasks_per_hour'],
                max_gap_s=r['max_gap_s'],source_J=r['ledger']['source_J'],residual_J=r['residual_J']))
    return dict(capacity=capacity,service=service,
        assumptions='Effective capacitance varied alone; R=10 Mohm, efficiencies fixed, zero restart cost, continuous standby, 1 h warm-up and 6 h observation. The 80-120 uF interval is assumed, not a measured distribution.')

def check(d):
    lookup={(r['C_uF'],r['VH_V']):r for r in d['capacity']}
    assert math.isclose(lookup[80,2.8]['usable_uJ'],122.88,abs_tol=1e-10)
    assert not lookup[99,2.55]['task_feasible']
    assert all(r['task_feasible'] for r in d['capacity'] if r['VH_V']==2.8)
    for c in [80,90,100,110,120]:
        a,b=[r for r in d['service'] if r['C_uF']==c]
        assert b['tasks_h']>a['tasks_h'] and b['max_gap_s']>a['max_gap_s']
        for r in [a,b]:
            assert math.isclose(r['source_J'],.0864,abs_tol=1e-10)
            assert abs(r['residual_J'])<1e-11

if __name__=='__main__':
    data=calculate();check(data)
    out=ROOT/'extended_results'
    (out/'tolerance_summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    for name in ['capacity','service']:
        with (out/('tolerance_'+name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=data[name][0]);w.writeheader();w.writerows(data[name])
    print('Capacitance check: 12 capacity cases and 10 service simulations passed.')
