# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Tests for the preset registry."""

import pytest

from exp.config import Config, ModelType
from exp.registries.presets import PresetRegistry, generate_config, preset_registry


class TestPresetRegistryContents:
    def test_all_14_base_datasets_registered(self):
        base = {
            "cora",
            "citeseer",
            "chameleon",
            "squirrel",
            "chameleon_filtered",
            "squirrel_filtered",
            "cornell",
            "texas",
            "film",
            "amazon_ratings",
            "minesweeper",
            "questions",
            "roman_empire",
            "tolokers",
        }
        registered = set(preset_registry.list_keys())
        assert base.issubset(registered)

    def test_total_preset_count(self):
        # 14 base NSD + 14*6 NSD variant presets
        assert len(preset_registry.list_keys()) > 14

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            preset_registry.get("definitely_not_a_preset")


class TestPresetRegistryGetOrDefault:
    def test_none_returns_default_config(self):
        result = preset_registry.get_or_default(None)
        assert isinstance(result, Config)
        assert result == Config()

    def test_known_name_returns_preset(self):
        result = preset_registry.get_or_default("cora")
        assert isinstance(result, Config)
        assert result.dataset.name == "cora"

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            preset_registry.get_or_default("nonexistent_preset")


class TestPresetValues:
    def test_cora_preset_values(self):
        cfg = preset_registry.get("cora")
        assert cfg.dataset.name == "cora"
        assert cfg.model.variant == "general"
        assert cfg.model.stalk_dim == 4
        assert cfg.model.type == ModelType.NSD

    def test_texas_uses_acc_stop_strategy(self):
        cfg = preset_registry.get("texas")
        assert cfg.optim.stop_strategy == "acc"

    def test_film_uses_diagonal_variant(self):
        cfg = preset_registry.get("film")
        assert cfg.model.variant == "diagonal"

    def test_chameleon_uses_orthogonal_variant(self):
        cfg = preset_registry.get("chameleon")
        assert cfg.model.variant == "orthogonal"

    def test_preset_returns_config_instance(self):
        for name in ["cora", "texas", "film", "amazon_ratings"]:
            assert isinstance(preset_registry.get(name), Config)


class TestPresetRegistryIsolation:
    """Each PresetRegistry instance is independent — useful in tests."""

    def test_fresh_registry_is_empty(self):
        r = PresetRegistry()
        assert r.list_keys() == []

    def test_fresh_registry_get_or_default_none(self):
        r = PresetRegistry()
        assert r.get_or_default(None) == Config()

    def test_registering_to_fresh_does_not_affect_global(self):
        r = PresetRegistry()
        r.register("test_preset", Config())
        assert "test_preset" not in preset_registry


class TestGenerateConfig:
    def test_generate_config_returns_config(self):
        cfg = generate_config(
            "cora",
            variant="general",
            stalk_dim=4,
            hidden_dim=32,
            num_layers=2,
            input_dropout=0.5,
            dropout=0.0,
            lr=0.01,
            weight_decay=5e-4,
        )
        assert isinstance(cfg, Config)
        assert cfg.dataset.name == "cora"
        assert cfg.model.variant == "general"
        assert cfg.model.stalk_dim == 4

    def test_generate_config_model_type_default_is_nsd(self):
        cfg = generate_config(
            "cora",
            variant="general",
            stalk_dim=4,
            hidden_dim=32,
            num_layers=2,
            input_dropout=0.0,
            dropout=0.0,
            lr=0.01,
            weight_decay=5e-4,
        )
        assert cfg.model.type == ModelType.NSD
