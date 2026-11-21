import runpy
import sys

import numpy as np
import pandas as pd

from swingbot import config
from swingbot.report.build import CDN_URL, SHORT_NOTE, SUSPECT_LABEL
from swingbot.report.ml import build_ml_report
from tests.test_report import make_market_panel


def _panels():
    # stock: 1500 dates (2010-2015), validation from 2014 -> walk-forward folds + validation row
    stock = make_market_panel("stock", n_dates=1500, assets=("SPY", "QQQ", "AAA", "BBB"), seed=1)
    stock["is_etf"] = stock["asset"].isin(["SPY", "QQQ"])
    stock["split"] = np.where(stock["date"] >= "2014-01-01", "validation", "discovery")
    stock["split"] = stock["split"].astype("category")
    # crypto: 300 dates in two years -> every fold skipped
    return {"stock": stock, "crypto": make_market_panel("crypto", seed=2)}


def _fast(monkeypatch):
    monkeypatch.setitem(config.ML_PARAMS, "n_estimators", 20)
    monkeypatch.setitem(config.ML_PARAMS, "min_child_samples", 20)
    monkeypatch.setattr(config, "SHAP_ROWS", 500)
    monkeypatch.setattr(config, "SHAP_INTERACTION_ROWS", 60)


def test_ml_report_renders(tmp_path, monkeypatch):
    _fast(monkeypatch)
    out = build_ml_report(_panels(), tmp_path / "ml_report_x.html", horizon=10, as_of="x")
    html = out.read_text(encoding="utf-8")
    assert SUSPECT_LABEL in html and CDN_URL in html
    assert "SHAP explains the model" in html and SHORT_NOTE in html
    assert "only out-of-sample score outside discovery" in html
    assert "S&amp;P 500/400 constituents" in html
    assert "{stock_mean}" not in html
    for v in ("etf", "stock"):
        for n in ("oos", "shap", "interactions"):
            assert (tmp_path / f"ml_{n}_{v}_x.csv").exists()
    oos = pd.read_csv(tmp_path / "ml_oos_stock_x.csv")
    assert {"all", "validation"} <= set(oos["fold"].astype(str))
    inter = pd.read_csv(tmp_path / "ml_interactions_stock_x.csv")
    assert {"phase4_verdict", "phase4_lift"} <= set(inter.columns)
    assert "Best solo candidate" in html and "Skipped folds" in html
    # crypto: all folds skipped -> reasons rendered, oos csv still written, no crash
    assert "test dates" in html or "calendar year" in html
    assert (tmp_path / "ml_oos_crypto_x.csv").exists()
    assert not (tmp_path / "ml_shap_crypto_x.csv").exists()


def test_attribute_script(tmp_path, monkeypatch, capsys):
    _fast(monkeypatch)
    data = tmp_path / "data"
    (data / "panel").mkdir(parents=True)
    _panels()["crypto"].to_parquet(data / "panel" / "crypto.parquet", index=False)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(sys, "argv", ["attribute.py", "--horizon", "10"])
    runpy.run_path(str(config._REPO_ROOT / "scripts" / "attribute.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "SKIP stock" in out
    rep = list((tmp_path / "reports").glob("ml_report_*.html"))
    assert rep and rep[0].name in out
