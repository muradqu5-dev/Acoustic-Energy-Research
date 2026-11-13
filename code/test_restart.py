"""Independent and conservation checks for the added scenarios."""
import unittest
from dataclasses import replace
import numpy as np
from model import Scenario
from intermittent_analysis import simulate as original
from restart_analysis import simulate,window,independent_rk4,PERIODS,COSTS
from component_analysis import calculate

class RestartChecks(unittest.TestCase):
    def test_zero_cost_recovers_original_all_schedules(self):
        for p in PERIODS:
            a=simulate(p);b=original(p)
            self.assertEqual(a['tasks'],b['tasks'])
            self.assertTrue(np.allclose(a['events'],b['events'],rtol=0,atol=1e-6))
    def test_energy_conservation_all_cases(self):
        for gated in [False,True]:
            for p in PERIODS:
                for b in COSTS:
                    r=simulate(p,b,gated)
                    self.assertLess(abs(r['residual_J']),1e-11)
                    self.assertAlmostEqual(r['ledger']['source_J'],.0864,places=11)
    def test_restart_accounting(self):
        r=simulate(1200,100)
        self.assertEqual(r['restarts'],18)
        self.assertAlmostEqual(r['ledger']['restart_J'],.0018,places=12)
    def test_steady_never_restarts(self):
        a=simulate(None,0);b=simulate(None,300)
        self.assertEqual(b['restarts'],0);self.assertEqual(a['events'],b['events'])
    def test_storage_bounds(self):
        for gated in [False,True]:
            for p in PERIODS:
                r=simulate(p,300,gated)
                self.assertGreaterEqual(r['min_voltage_V'],0)
                self.assertLessEqual(r['max_voltage_V'],3.3+1e-10)
    def test_oversized_restart_rejected(self):
        with self.assertRaises(ValueError):simulate(1200,350)
    def test_restart_sweep_decreases_count_for_reported_cases(self):
        for gated in [False,True]:
            for p in [1200,3600]:
                counts=[simulate(p,b,gated)['tasks'] for b in COSTS]
                self.assertEqual(counts,sorted(counts,reverse=True))
    def test_continuous_and_gated_identical_without_brownout(self):
        for p in [None,10,60,300]:
            a=simulate(p,100,False);b=simulate(p,100,True)
            self.assertTrue(np.allclose(a['events'],b['events']))
    def test_independent_rk4_includes_restart(self):
        for gated in [False,True]:
            for p in [None,300,1200,3600]:
                a=window(Scenario(),p,100e-6,gated,duration=7200)
                b=independent_rk4(Scenario(),p,100e-6,gated,duration=7200)
                self.assertEqual(len(a['events']),len(b['events']))
                self.assertEqual(len(a['boots']),len(b['boots']))
                self.assertLess(max(abs(x-y) for x,y in zip(a['events'],b['events'])),1.)
    def test_no_source_no_tasks_or_restarts(self):
        r=window(replace(Scenario(),source_power_w=0),1200,100e-6,True,duration=7200)
        self.assertEqual(r['tasks'],0);self.assertEqual(r['restarts'],0)
    def test_datasheet_units(self):
        s=calculate()
        self.assertAlmostEqual(s['typical_task_uJ'],6.90525)
        self.assertAlmostEqual(s['specified_limit_task_uJ'],12.705)
        self.assertAlmostEqual(s['resistance_stress_Mohm'],5.)
    def test_component_failure_is_power_not_capacity(self):
        for r in calculate()['rows']:
            self.assertGreater(r['energy_capacity_uJ'],r['task_load_uJ'])
            if r['refill_s'] is None:self.assertLess(r['remaining_power_at_high_uW'],0)
            else:self.assertGreater(r['remaining_power_at_high_uW'],0)

if __name__=='__main__':unittest.main()
