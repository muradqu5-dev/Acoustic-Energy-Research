import math
import unittest
from dataclasses import replace
from model import Scenario
from revision_analysis import fixed_decay, maximum_capacitance, rk4_endpoint

class ExtendedLeakageChecks(unittest.TestCase):
    def test_models_match_at_baseline(self):
        a=Scenario();b=fixed_decay(100e-6)
        self.assertAlmostEqual(a.recharge_time_s(),b.recharge_time_s(),places=9)
    def test_constant_decay_keeps_RC_fixed(self):
        for c in [10e-6,47e-6,100e-6,1000e-6]:
            s=fixed_decay(c)
            self.assertAlmostEqual(s.capacitance_f*s.leakage_resistance_ohm,1000)
    def test_leakage_proportional_to_capacitance(self):
        self.assertAlmostEqual(fixed_decay(200e-6).leakage_at_high_w,2*fixed_decay(100e-6).leakage_at_high_w,places=12)
    def test_upper_capacitance_limit(self):
        cm=maximum_capacitance()
        self.assertTrue(math.isfinite(fixed_decay(cm*(1-1e-6)).recharge_time_s()))
        self.assertTrue(math.isinf(fixed_decay(cm).recharge_time_s()))
        self.assertTrue(math.isinf(fixed_decay(cm*(1+1e-6)).recharge_time_s()))
    def test_feasible_interval(self):
        self.assertLess(Scenario().minimum_capacitance_f,maximum_capacitance())
        self.assertTrue(math.isfinite(fixed_decay(47e-6).fixed_event_recharge_time_s()))
        self.assertTrue(math.isinf(fixed_decay(220e-6).recharge_time_s()))
    def test_independent_numerical_endpoint(self):
        s=fixed_decay(47e-6)
        exact=.5*s.capacitance_f*s.high_voltage_v**2
        self.assertLess(abs(rk4_endpoint(s,1000)-exact)/exact,1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
