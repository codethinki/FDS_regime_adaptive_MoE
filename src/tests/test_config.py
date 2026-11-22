import pytest

from regime_moe.config import load_config


def test_defaults_load():
    cfg = load_config()
    assert cfg.name == "defaults"
    assert cfg.model.n_experts == cfg.router.hmm.n_states


def test_experiment_overrides_only_given_keys(tmp_path):
    exp = tmp_path / "exp.toml"
    exp.write_text('[train]\nepochs = 3\n[router.hmm]\nn_states = 2\n')
    cfg, base = load_config(exp), load_config()
    assert cfg.name == "exp"
    assert cfg.train.epochs == 3 and cfg.router.hmm.n_states == 2
    assert cfg.train.batch_size == base.train.batch_size
    assert cfg.router.hmm.n_iter == base.router.hmm.n_iter


@pytest.mark.parametrize("text", ['[train]\nepoch = 3\n', '[trian]\nepochs = 3\n', '[router.hmm]\nstates = 2\n'])
def test_unknown_keys_fail(tmp_path, text):
    exp = tmp_path / "exp.toml"
    exp.write_text(text)
    with pytest.raises(KeyError):
        load_config(exp)


def test_wrong_type_fails(tmp_path):
    exp = tmp_path / "exp.toml"
    exp.write_text('[train]\nepochs = "many"\n')
    with pytest.raises(TypeError):
        load_config(exp)
