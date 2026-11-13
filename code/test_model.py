"""Independent numerical and limiting-case checks; python -m unittest -v."""
import math
import unittest
from dataclasses import replace
from model import Scenario


class EnergyModelChecks(unittest.TestCase):
    def setUp(self):
        self.s=Scenario()

    def test_energy_window(self):
        self.assertAlmostEqual(self.s.usable_event_energy_j*1e6,275.6,places=8)

    def test_no_leakage_linear_energy(self):
        s=replace(self.s, leakage_resistance_ohm=math.inf)
        self.assertAlmostEqual(s.recharge_time_s(),344.5/1.5,places=8)

    def test_zero_standby_and_no_leakage(self):
        s=replace(self.s, leakage_resistance_ohm=math.inf,standby_power_w=0)
        self.assertAlmostEqual(s.recharge_time_s(),344.5/2,places=8)

    def test_reachability_below_and_at_boundary(self):
        critical=self.s.critical_source_power_w
        self.assertTrue(math.isinf(replace(self.s,source_power_w=critical*(1-1e-7)).recharge_time_s()))
        at=replace(self.s,source_power_w=critical)
        self.assertTrue(math.isinf(at.recharge_time_s()))
        self.assertTrue(math.isfinite(replace(self.s,source_power_w=critical*(1+1e-7)).recharge_time_s()))

    def test_deficit_cannot_recharge(self):
        self.assertTrue(math.isinf(replace(self.s,source_power_w=0).recharge_time_s()))

    def test_calculated_time_reaches_high_threshold(self):
        self.assertAlmostEqual(self.s.voltage_at_s(self.s.recharge_time_s()),3.3,places=10)

    def test_equilibrium_limit(self):
        self.assertAlmostEqual(self.s.voltage_at_s(50000),math.sqrt(15),places=10)

    def test_capacitance_scaling_at_fixed_resistance(self):
        s=replace(self.s,capacitance_f=2*self.s.capacitance_f)
        self.assertAlmostEqual(s.recharge_time_s(),2*self.s.recharge_time_s(),places=8)
        self.assertEqual(s.critical_source_power_w,self.s.critical_source_power_w)

    def test_finite_leakage_increases_charging_time(self):
        self.assertGreater(self.s.recharge_time_s(),replace(self.s,leakage_resistance_ohm=math.inf).recharge_time_s())

    def test_negligible_leakage_limit(self):
        s=replace(self.s,leakage_resistance_ohm=1e20)
        self.assertAlmostEqual(s.recharge_time_s(),replace(self.s,leakage_resistance_ohm=math.inf).recharge_time_s(),places=7)

    def test_event_energy_accounting(self):
        before=0.5*self.s.capacitance_f*self.s.high_voltage_v**2
        after=0.5*self.s.capacitance_f*self.s.post_event_voltage_v()**2
        self.assertAlmostEqual((before-after)*self.s.output_efficiency,self.s.event_energy_j,places=12)

    def test_insufficient_capacitance_detected(self):
        s=replace(self.s,capacitance_f=10e-6)
        self.assertIsNone(s.post_event_voltage_v())
        self.assertIsNone(s.fixed_event_recharge_time_s())

    def test_event_at_minimum_capacitance_reaches_low_threshold(self):
        s=replace(self.s,capacitance_f=self.s.minimum_capacitance_f)
        self.assertAlmostEqual(s.post_event_voltage_v(),s.low_voltage_v,places=10)

    def test_fixed_event_refill_and_full_window_are_distinct(self):
        self.assertLess(self.s.fixed_event_recharge_time_s(),self.s.recharge_time_s())

    def test_rk4_independent_integration(self):
        # Integrate dE/dt directly, without using the analytical trajectory.
        for s in [self.s,replace(self.s,source_power_w=8e-6),replace(self.s,leakage_resistance_ohm=math.inf)]:
            duration=s.recharge_time_s()
            energy=0.5*s.capacitance_f*s.low_voltage_v**2
            steps=10000
            dt=duration/steps
            def derivative(e):
                return s.input_efficiency*s.source_power_w-s.standby_power_w-2*e/(s.capacitance_f*s.leakage_resistance_ohm)
            for _ in range(steps):
                k1=derivative(energy)
                k2=derivative(energy+dt*k1/2)
                k3=derivative(energy+dt*k2/2)
                k4=derivative(energy+dt*k3)
                energy+=dt*(k1+2*k2+2*k3+k4)/6
            expected=0.5*s.capacitance_f*s.high_voltage_v**2
            self.assertLess(abs(energy-expected)/expected,1e-9)

    def test_invalid_inputs(self):
        for update in [{'capacitance_f':0},{'input_efficiency':1.1},{'event_energy_j':-1},{'leakage_resistance_ohm':-1},{'source_power_w':float('nan')}]:
            with self.assertRaises(ValueError): replace(self.s,**update)


if __name__=='__main__': unittest.main(verbosity=2)
