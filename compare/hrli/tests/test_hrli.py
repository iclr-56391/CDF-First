import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from compare.hrli.hrli import HRLI
from model.base import ForwardInput


class HRLITests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        condition = torch.linspace(-2.0, 2.0, 48).unsqueeze(1)
        target = 0.7 * condition + 0.35 * torch.randn_like(condition)
        self.condition = condition
        self.target = target
        self.model = HRLI(1, 1, cv_samples=40, cv_iter=8)
        self.model.fit(
            DataLoader(TensorDataset(condition, target), batch_size=12),
            torch.device("cpu"),
        )

    def test_density_is_finite_and_normalized(self):
        grid = torch.linspace(-5.0, 5.0, 501).unsqueeze(1)
        condition = torch.zeros_like(grid)
        result = self.model(ForwardInput(condition, grid))
        density = result.log_prob.exp()
        integral = torch.trapezoid(density, grid[:, 0]).item()

        self.assertTrue(torch.isfinite(result.log_prob).all())
        self.assertAlmostEqual(integral, 1.0, delta=0.01)

    def test_cdf_is_bounded_and_monotone(self):
        grid = torch.linspace(-5.0, 5.0, 101).unsqueeze(1)
        condition = torch.zeros_like(grid)
        cdf = self.model.get_cdf(condition, grid)

        self.assertTrue(torch.isfinite(cdf).all())
        self.assertTrue(torch.all((cdf >= 0.0) & (cdf <= 1.0)))
        self.assertTrue(torch.all(cdf[1:] >= cdf[:-1]))

    def test_reports_selected_bandwidths(self):
        self.assertEqual(self.model._x_bandwidth.shape, (1,))
        self.assertEqual(self.model._y_bandwidth.shape, (1,))
        self.assertTrue(torch.isfinite(torch.tensor(self.model.cv_score)))


if __name__ == "__main__":
    unittest.main()
