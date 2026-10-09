"""CLI entry point: python -m decision_lab <command>"""

import argparse
import sys
from pathlib import Path

import numpy as np

from decision_lab import DATA_DIR, MODELS_DIR, RESULTS_DIR
from decision_lab.config import load_config


def cmd_generate(args):
    """Generate synthetic text-state datasets with typed-question labels."""
    from decision_lab.states.generator import generate_dataset

    cfg = load_config(args.config)
    q = cfg.questions
    print(f"Generating text states (train={cfg.generator.num_train}, "
          f"indist={cfg.generator.num_test_indist}, heldout={cfg.generator.num_test_heldout}; "
          f"questions: {len(q.choice)} choice / {len(q.score)} score / {len(q.noul)} noul)")
    generate_dataset(cfg, args.data_dir)


def cmd_extract(args):
    """Extract features from datasets using llama-server."""
    from decision_lab.backbone.features import extract_features
    from decision_lab.backbone.llama_server import LlamaServer

    cfg = load_config(args.config)
    gguf = Path(cfg.model.gguf_path).expanduser().resolve()

    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        for split in ["train", "test_indist", "test_heldout", "train_breakout", "test_breakout"]:
            data_path = args.data_dir / f"{split}.jsonl"
            if not data_path.exists():
                print(f"  skip {split}: not found")
                continue
            cache_path = args.data_dir / f"features_{split}.npz"
            print(f"Extracting features: {split} → {cache_path}")
            feats = extract_features(data_path, cache_path, server)
            print(f"  shape={feats.shape}")


def cmd_train(args):
    """Train typed decision head on extracted features."""
    from decision_lab.head.model import build_question_spec
    from decision_lab.head.train import encode_labels, train_head
    from decision_lab.states.dataset import load_dataset

    cfg = load_config(args.config)

    train_path = args.data_dir / "train.jsonl"
    if not train_path.exists():
        print(f"No training data at {train_path}")
        sys.exit(1)

    features = np.load(args.data_dir / "features_train.npz")["features"]
    states = load_dataset(train_path)
    question_spec = build_question_spec(cfg.questions)
    labels_by_qid = encode_labels(states, question_spec)

    assert len(features) == len(states), f"{len(features)} != {len(states)}"

    train_head(features, labels_by_qid, question_spec, cfg, args.models_dir / "head_trained.pt")


def cmd_train_dynamic(args):
    """Train dynamic decision head (attention-based) on extracted features."""
    from decision_lab.backbone.llama_server import LlamaServer
    from decision_lab.head.model import build_question_spec
    from decision_lab.head.dynamic_train import generate_dynamic_training_data, train_dynamic_head
    from decision_lab.states.dataset import load_dataset

    cfg = load_config(args.config)
    gguf = Path(cfg.model.gguf_path).expanduser().resolve()

    train_path = args.data_dir / "train.jsonl"
    if not train_path.exists():
        print(f"No training data at {train_path}")
        sys.exit(1)

    features = np.load(args.data_dir / "features_train.npz")["features"]
    states = load_dataset(train_path)
    question_spec = build_question_spec(cfg.questions)

    # Breakout states train the dynamic head only — never the typed head
    # (encode_labels would KeyError on document states for paddle_direction).
    brk_path = args.data_dir / "train_breakout.jsonl"
    if brk_path.exists():
        from decision_lab.states.breakout import breakout_question_spec
        brk_states = load_dataset(brk_path)
        brk_feats = np.load(args.data_dir / "features_train_breakout.npz")["features"]
        assert len(brk_feats) == len(brk_states), \
            f"{len(brk_feats)} != {len(brk_states)}"
        w = cfg.dynamic_head.breakout_weight
        states = states + brk_states * w
        features = np.concatenate([features] + [brk_feats] * w)
        question_spec = {**question_spec, **breakout_question_spec()}
        print(f"  mixed in {len(brk_states)} breakout states × weight {w}")

    assert len(features) == len(states), f"{len(features)} != {len(states)}"

    dcfg = cfg.dynamic_head
    print(f"Training dynamic head: {dcfg.hidden_dim} hidden, d_k={dcfg.d_k}, "
          f"anchor_epochs={dcfg.anchor_epochs}, variants={dcfg.variants_per_question}")

    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        samples = generate_dynamic_training_data(
            states, features, question_spec, server,
            num_variants_per_question=dcfg.variants_per_question,
        )
        train_dynamic_head(
            samples, cfg, args.models_dir / "head_dynamic.pt",
            anchor_fraction=dcfg.anchor_epochs / max(dcfg.max_epochs, 1),
        )


def cmd_eval(args):
    """Run benchmarks for all agents on test sets."""
    from decision_lab.backbone.llama_server import LlamaServer
    from decision_lab.eval.benchmark import run_benchmark

    cfg = load_config(args.config)
    gguf = Path(cfg.model.gguf_path).expanduser().resolve()

    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        all_metrics = run_benchmark(args.data_dir, cfg, server, args.models_dir)

    from decision_lab.eval.report import generate_report
    generate_report(all_metrics, args.results_dir)
    print(f"Report saved to {args.results_dir}")


def cmd_report(args):
    """Generate report from existing metrics JSON."""
    import json
    from decision_lab.eval.report import generate_report

    if not args.results_dir.joinpath("metrics.json").exists():
        print("No metrics.json found. Run eval first.")
        sys.exit(1)

    raw = json.loads(args.results_dir.joinpath("metrics.json").read_text())
    from decision_lab.eval.benchmark import AgentMetrics
    all_metrics = {
        test: {agent: AgentMetrics(**m) for agent, m in agents.items()}
        for test, agents in raw.items()
    }
    generate_report(all_metrics, args.results_dir)
    print(f"Report regenerated at {args.results_dir}")


def cmd_ui(args):
    """Start the web UI."""
    import uvicorn

    cfg = load_config(args.config)
    print(f"Starting web UI at http://{cfg.web.host}:{cfg.web.port} "
          f"(llama-server on port {cfg.model.server_port})")
    uvicorn.run(
        "decision_lab.webapp.app:app",
        host=cfg.web.host,
        port=cfg.web.port,
        log_level="info",
    )


def cmd_all(args):
    """Run full pipeline: generate → extract → train → eval → report."""
    args_data = argparse.Namespace(config=args.config, data_dir=args.data_dir)
    args_train = argparse.Namespace(config=args.config, data_dir=args.data_dir, models_dir=args.models_dir)
    args_eval = argparse.Namespace(
        config=args.config, data_dir=args.data_dir,
        models_dir=args.models_dir, results_dir=args.results_dir,
    )
    args_report = argparse.Namespace(config=args.config, results_dir=args.results_dir)

    print("=" * 60)
    print("STEP 1: Generate Data")
    print("=" * 60)
    cmd_generate(args_data)

    print()
    print("=" * 60)
    print("STEP 2: Extract Features")
    print("=" * 60)
    cmd_extract(args_data)

    print()
    print("=" * 60)
    print("STEP 3: Train Head")
    print("=" * 60)
    cmd_train(args_train)

    print()
    print("=" * 60)
    print("STEP 4: Evaluate")
    print("=" * 60)
    cmd_eval(args_eval)

    print()
    print("=" * 60)
    print("STEP 5: Report")
    print("=" * 60)
    cmd_report(args_report)

    print()
    print("Done. See results/report.md")


def main():
    parser = argparse.ArgumentParser(description="Decision Lab — Typed head vs LM benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd, fn in [
        ("generate", cmd_generate),
        ("extract", cmd_extract),
        ("train", cmd_train),
        ("train-dynamic", cmd_train_dynamic),
        ("eval", cmd_eval),
        ("report", cmd_report),
        ("all", cmd_all),
        ("ui", cmd_ui),
    ]:
        p = sub.add_parser(cmd, help=fn.__doc__ or "")
        p.set_defaults(func=fn)

    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", default=str(DATA_DIR), type=Path, help="Data directory")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), type=Path, help="Models directory")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), type=Path, help="Results directory")

    args = parser.parse_args()

    if not hasattr(args, "data_dir"):
        args.data_dir = Path("data")
    if not hasattr(args, "models_dir"):
        args.models_dir = Path("models")
    if not hasattr(args, "results_dir"):
        args.results_dir = Path("results")

    args.func(args)


if __name__ == "__main__":
    main()