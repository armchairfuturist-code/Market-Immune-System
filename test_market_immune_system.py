
import unittest
import pandas as pd
import numpy as np
from market_immune_system import MarketImmuneSystem, SignalStatus

class TestMarketImmuneSystem(unittest.TestCase):
    
    def setUp(self):
        self.mis = MarketImmuneSystem(lookback_days=100, fetch_days=200, min_lookback=20)
        # Create dummy data
        dates = pd.date_range("2023-01-01", periods=200)
        self.returns = pd.DataFrame(np.random.normal(0, 0.01, (200, 5)), index=dates, columns=["A", "B", "C", "D", "SPY"])
        self.prices = pd.DataFrame(np.random.uniform(100, 200, (200, 5)), index=dates, columns=["A", "B", "C", "D", "SPY"])
        self.volumes = pd.DataFrame(np.random.uniform(1000, 5000, (200, 5)), index=dates, columns=["A", "B", "C", "D", "SPY"])
        
    def test_calculate_turbulence(self):
        score, contribs = self.mis.calculate_turbulence(self.returns)
        self.assertIsInstance(score, float)
        self.assertIsInstance(contribs, pd.Series)
        self.assertGreaterEqual(score, 0)
        
    def test_calculate_absorption_ratio(self):
        score = self.mis.calculate_absorption_ratio(self.returns)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1000)
        
    def test_rolling_metrics(self):
        # Rolling Turbulence
        roll_turb = self.mis.calculate_rolling_turbulence(self.returns)
        self.assertEqual(len(roll_turb), len(self.returns) - 20) # Min lookback offset
        
        # Rolling Liquidity
        roll_liq = self.mis.calculate_rolling_liquidity(self.returns, self.prices, self.volumes, "SPY")
        self.assertEqual(len(roll_liq), len(self.returns))
        
    def test_signal_generation(self):
        # Normal
        sig, msg = self.mis.generate_signal(0.0, 100.0, 500.0, 5)
        self.assertEqual(sig, SignalStatus.GREEN)
        
        # Crash
        sig, msg = self.mis.generate_signal(-2.0, 400.0, 500.0, 5) # > 370 critical
        self.assertEqual(sig, SignalStatus.BLACK)
        
        # Fragile (Absorption)
        sig, msg = self.mis.generate_signal(0.0, 100.0, 900.0, 5) # > 800
        self.assertEqual(sig, SignalStatus.ORANGE)

    def test_advanced_signal(self):
        # Fragility override
        sig = self.mis.get_advanced_signal(100, 0, 0.5, absorption=900)
        self.assertIn("Fragile", sig)
        
        # Strong Buy
        sig = self.mis.get_advanced_signal(100, 0, 0.4, absorption=500)
        self.assertIn("Buy", sig) # Low turb + good liq + low hurst = Strong Buy
        
    def test_market_metrics_structure(self):
        metrics = self.mis.get_current_metrics(self.returns, self.prices, self.volumes)
        self.assertTrue(hasattr(metrics, "turbulence_score"))
        self.assertTrue(hasattr(metrics, "absorption_ratio"))
        self.assertTrue(hasattr(metrics, "top_contributors"))
        
if __name__ == '__main__':
    unittest.main()
