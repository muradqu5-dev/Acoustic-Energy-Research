"""Checks for published-curve fitting and event-driven energy accounting."""
import math
import unittest
from dataclasses import replace
import numpy as np
from model import Scenario
from empirical_analysis import energy_model,voltage_model,fit_one
from intermittent_analysis import advance,time_to,run_window,simulate,rk4_check

class ExtendedStudyChecks(unittest.TestCase):
    def test_no_leakage_advance(self):
        self.assertAlmostEqual(advance(.001,2e-6,0.,20),.00104)
    def test_exact_update_composes(self):
        a=advance(.0002,1.5e-6,.002,123)
        b=advance(advance(.0002,1.5e-6,.002,40),1.5e-6,.002,83)
        self.assertAlmostEqual(a,b,places=14)
    def test_threshold_time_matches_original(self):
        s=Scenario();e=.5*s.capacitance_f*s.low_voltage_v**2;h=.5*s.capacitance_f*s.high_voltage_v**2
        self.assertAlmostEqual(time_to(e,h,s.surplus_w,2/(s.capacitance_f*s.leakage_resistance_ohm)),s.recharge_time_s(),places=8)
    def test_steady_task_spacing(self):
        s=Scenario();r=run_window(s,duration_s=3600)
        self.assertTrue(np.allclose(np.diff(r['events']),s.fixed_event_recharge_time_s(),atol=1e-7))
    def test_equal_input_energy(self):
        for p in [None,10,60,300,1200,3600]:
            self.assertAlmostEqual(simulate(p)['ledger']['source_J'],.0864,places=11)
    def test_ledger_closes(self):
        for p in [None,10,60,300,1200,3600]:
            self.assertLess(abs(simulate(p)['ledger']['energy_balance_error_J']),1e-11)
    def test_empty_storage_never_negative(self):
        r=run_window(replace(Scenario(),source_power_w=0),duration_s=3600)
        self.assertEqual(r['tasks'],0);self.assertEqual(r['end_energy_J'],0.)
        self.assertGreater(r['ledger']['empty_s'],0.)
    def test_upper_voltage_never_exceeded(self):
        for p in [None,60,1200,3600]:self.assertLessEqual(simulate(p)['max_voltage_V'],3.3+1e-8)
    def test_event_withdrawal_accounting(self):
        r=simulate(1200)
        self.assertAlmostEqual(r['ledger']['task_storage_J'],r['tasks']*125e-6,places=12)
    def test_fast_switching_approaches_average(self):
        steady=simulate();burst=simulate(.2)
        self.assertLessEqual(abs(steady['tasks']-burst['tasks']),1)
        self.assertLess(abs(steady['ledger']['leakage_J']-burst['ledger']['leakage_J']),5e-5)
    def test_independent_integration_event_counts(self):
        for p in [None,60,300,1200,3600]:
            exact=run_window(Scenario(),p,duration_s=3600)
            count,events,e=rk4_check(Scenario(),p,dt=.1)
            self.assertEqual(count,exact['tasks'])
            self.assertLess(max(abs(a-b) for a,b in zip(events,exact['events'])),2.)
    def test_task_too_large_rejected(self):
        with self.assertRaises(ValueError):run_window(replace(Scenario(),capacitance_f=10e-6))
    def test_positive_fit_recovers_energy_curve(self):
        t=np.arange(50,301,50);p=fit_one(energy_model,t,energy_model(t,2.5,250))
        self.assertTrue(np.allclose(p,[2.5,250],rtol=1e-5))
    def test_positive_fit_recovers_voltage_curve(self):
        t=np.arange(50,301,50);p=fit_one(voltage_model,t,voltage_model(t,2.5,250))
        self.assertTrue(np.allclose(p,[2.5,250],rtol=1e-5))

if __name__=='__main__':unittest.main()
