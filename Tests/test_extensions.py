"""Tests for the integrated hard-negative and GP pipelines."""

import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from Topanda import (
    GPTopanda,
    GPHTopanda,
    HardNegativeTopanda,
    TrainingConfig,
    TripletLearner,
)
from Topanda.DeepMetricLearning.extensions.miners import candidate_pool, mine_negatives
from Topanda.DeepMetricLearning.extensions.residual_topanda import ResidualTopanda


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        features, self.labels = load_iris(return_X_y=True)
        self.features = pd.DataFrame(StandardScaler().fit_transform(features))
        self.base = TripletLearner(
            embedding_dim=8, epochs=1, device="cpu"
        ).fit(self.features, self.labels)

    def test_public_imports(self):
        from Topanda import GPTopanda as PublicGP
        from Topanda import GPHTopanda as PublicGPH
        from Topanda import HardNegativeTopanda as PublicHard

        self.assertIs(PublicGP, GPTopanda)
        self.assertIs(PublicGPH, GPHTopanda)
        self.assertIs(PublicHard, HardNegativeTopanda)

    def test_identity_initialization(self):
        values = torch.randn(15, 8)
        self.assertTrue(torch.equal(ResidualTopanda(8)(values), values))

    def test_gp_variants_freeze_base_and_preserve_dimension(self):
        before = {key: value.clone() for key, value in self.base.model.state_dict().items()}
        config = TrainingConfig(
            strategy="random",
            preservation_lambda=1.0,
            epochs=2,
            batch_size=16,
            candidate_pool_size=12,
        )
        for pipeline, strategy in ((GPTopanda, "random"), (GPHTopanda, "hard")):
            model = pipeline(self.base, config).fit(self.features, self.labels)
            result = model.transform(self.features.iloc[:9])
            self.assertEqual(result.shape, (9, 8))
            self.assertTrue(np.isfinite(result.values).all())
            self.assertEqual(model.result_.config["strategy"], strategy)
            self.assertTrue(all(not p.requires_grad for p in model.base_.model.parameters()))
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.base_.model.state_dict()[key]))
                self.assertTrue(torch.equal(value, self.base.model.state_dict()[key]))

    def test_zero_epoch_refinement_is_identity(self):
        config = TrainingConfig(strategy="hard", preservation_lambda=1.0, epochs=0)
        model = GPHTopanda(self.base, config).fit(self.features, self.labels)
        np.testing.assert_array_equal(
            model.transform(self.features), self.base.transform(self.features)
        )

    def test_hard_miner_selects_closest_wrong_class(self):
        mined = mine_negatives(
            "hard",
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.5, 0.0]]),
            torch.tensor([0]),
            torch.tensor([0, 1, 1]),
            torch.tensor([10, 20, 30]),
            1.0,
        )
        self.assertEqual(mined.negative_indices.item(), 30)

    def test_candidate_pool_contains_every_class(self):
        labels = np.repeat(np.arange(10), 2)
        for size in (10, 11, 16, 25):
            pool = candidate_pool(labels, size, np.random.default_rng(42))
            self.assertEqual(len(pool), size)
            self.assertEqual(len(np.unique(labels[pool])), 10)

    def test_hard_negative_training_is_reproducible(self):
        options = dict(
            embedding_dim=8,
            epochs=2,
            batch_size=16,
            candidate_pool_size=12,
            seed=42,
        )
        first = HardNegativeTopanda(**options).fit(self.features, self.labels)
        second = HardNegativeTopanda(**options).fit(self.features, self.labels)
        np.testing.assert_array_equal(
            first.transform(self.features), second.transform(self.features)
        )

    def test_invalid_labels_are_rejected(self):
        for pipeline in (HardNegativeTopanda(), GPTopanda(self.base)):
            with self.assertRaises(ValueError):
                pipeline.fit(self.features, np.zeros(len(self.features)))

    def test_unfitted_base_is_rejected(self):
        with self.assertRaises(ValueError):
            GPTopanda(TripletLearner()).fit(self.features, self.labels)


if __name__ == "__main__":
    unittest.main()
