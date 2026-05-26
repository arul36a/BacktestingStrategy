from __future__ import annotations


def test_import_src_modules():
    from src.data_loader import GoldDataLoader
    from src.feature_engineering import FeaturePipeline
    from src.ml_models import GoldMLEngine

    assert GoldDataLoader is not None
    assert FeaturePipeline is not None
    assert GoldMLEngine is not None
