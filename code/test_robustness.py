import unittest,math,json
from dataclasses import replace
import numpy as np
from model import Scenario
from robustness_analysis import threshold_controls,source_window,source_run,acoustic_budget,fit_split_checks,leakage_ranges,ROOT

class RobustnessChecks(unittest.TestCase):
    def test_voltage_control_reproduces_reference(self):
        rows=threshold_controls()
        for v,rate,gap in [(3.3,15,237.58031268),(2.8,24,149.74183363),(2.55,28,128.94798583)]:
            r=next(r for r in rows if r['VH_V']==v and r['period_s'] is None and r['boot_uJ']==0)
            self.assertEqual(r['tasks_h'],rate);self.assertAlmostEqual(r['max_gap_s'],gap,places=6)
    def test_equal_voltage_burst_control(self):
        rows=threshold_controls()
        for v,rate in [(3.3,27),(2.8,33),(2.55,36)]:
            r=next(r for r in rows if r['VH_V']==v and r['period_s']==1200 and r['boot_uJ']==0)
            self.assertEqual(r['tasks_h'],rate)
    def test_low_voltage_capacity_margin(self):
        s=replace(Scenario(),high_voltage_v=2.55)
        self.assertAlmostEqual(s.usable_event_energy_j*1e6,100.1,places=8)
        self.assertLess(replace(s,capacitance_f=99e-6).usable_event_energy_j,s.event_energy_j)
    def test_controls_equal_energy(self):
        for r in threshold_controls():
            self.assertAlmostEqual(r['source_J'],.0864,places=10)
            self.assertLess(abs(r['residual_J']),1e-11)
    def test_voltage_source_exponential_limit(self):
        s=replace(Scenario(),capacitance_f=.001,low_voltage_v=.05,high_voltage_v=8.,
            standby_power_w=0.,input_efficiency=1.,leakage_resistance_ohm=math.inf,event_energy_j=1e-6)
        voc=5.;pref=4e-6;rs=2.8*(voc-2.8)/pref
        r=source_window(s,voc,pref,duration=1000,initial=(.1,True))
        expected=voc-(voc-.1)*math.exp(-1000/(rs*s.capacitance_f))
        self.assertAlmostEqual(r['state'][0],expected,places=9)
    def test_source_energy_ledger(self):
        r=source_run(Scenario(),5.,4e-6,1200,100e-6)
        self.assertLess(abs(r['residual_J']),1e-9)
        self.assertGreater(len(r['boots']),0)
    def test_independent_solver_and_step(self):
        a=source_window(Scenario(),5.,4e-6,1200,100e-6,duration=3600)
        b=source_window(Scenario(),5.,4e-6,1200,100e-6,duration=3600,method='RK45',max_step=5)
        self.assertEqual(len(a['events']),len(b['events']))
        self.assertEqual(len(a['boots']),len(b['boots']))
        self.assertLess(max(abs(x-y) for x,y in zip(a['events'],b['events'])),1e-3)
        self.assertLess(max(abs(x-y) for x,y in zip(a['boots'],b['boots'])),1e-3)
    def test_calibrated_energy_and_ledgers(self):
        d=json.loads((ROOT/'extended_results/robustness_summary.json').read_text())
        for r in d['voltage_source']:
            self.assertAlmostEqual(r['source_J'],.0864,places=7)
            self.assertLess(abs(r['residual_J']),1e-9)
    def test_resistive_leakage_is_not_a_third_identifiable_shape(self):
        a=2.05;tau=119.1;c=.001;rp=tau/c
        for rl in [5e5,1e6,10e6]:
            rs=rp*rl/(rl-rp);vs=a*rl/(rl-rp)
            self.assertAlmostEqual(vs*rl/(rs+rl),a,places=12)
            self.assertAlmostEqual(c*rs*rl/(rs+rl),tau,places=10)
    def test_fit_split_sensitivity(self):
        rows=fit_split_checks()
        for split in [200,300,450]:
            a,b=[r for r in rows if r['split_s']==split]
            self.assertEqual(a['nfit']+a['ncheck'],18)
            self.assertLess(b['check_rmse_V'],a['check_rmse_V'])
    def test_acoustic_units(self):
        a=acoustic_budget()
        self.assertAlmostEqual(a['rows'][-1]['intensity_W_m2'],.01)
        self.assertAlmostEqual(a['rows'][-1]['area_for_4uW_cm2'][0],4.)
        self.assertAlmostEqual(a['rows'][0]['area_for_4uW_cm2'][0],4000.)
    def test_loss_range_monotonicity(self):
        r=[r for r in leakage_ranges() if r['standby_uW']==.86875 and r['eta_in']==.5]
        self.assertTrue(all(a['required_source_uW']>b['required_source_uW'] for a,b in zip(r,r[1:])))

if __name__=='__main__':unittest.main()
