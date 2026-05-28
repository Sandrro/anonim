from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import DEFAULT_IGNORED_TOKENS, evaluate_against_ground_truth
from .pipeline import anonymize_docx


def cmd_anonymize(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    report = anonymize_docx(
        input_docx=args.input,
        output_docx=out,
        report_json=report_path,
        llm_mode=args.llm_mode,
        model=args.model,
        env_path=args.env,
        fixture_mapping=args.fixture_mapping,
    )
    print(json.dumps({"output_docx": str(out), "report_json": str(report_path), "replacement_count": report["replacement_count"]}, ensure_ascii=False, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_docx = out_dir / "anonymized.docx"
    report_json = out_dir / "anonymization_report.json"
    metrics_json = out_dir / "eval_metrics.json"
    anonymize_docx(
        input_docx=args.input,
        output_docx=output_docx,
        report_json=report_json,
        llm_mode=args.llm_mode,
        model=args.model,
        env_path=args.env,
        fixture_mapping=args.fixture_mapping,
    )
    ignore = set(DEFAULT_IGNORED_TOKENS) if args.ignore_date_non_target else set()
    metrics = evaluate_against_ground_truth(
        output_docx=output_docx,
        ground_truth_docx=args.ground_truth,
        token_to_synthetic_mapping=args.token_to_synthetic_mapping,
        ignore_tokens=ignore,
    )
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_docx": str(output_docx), "report_json": str(report_json), "metrics_json": str(metrics_json), **metrics}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DOCX anonymization with LLM-first extraction and regex audit/fallback.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument("--env", default=None, help="Path to .env with OPENAI_API_KEY.")
    parser.add_argument("--llm-mode", choices=["openai", "off", "fixture"], default="openai", help="LLM provider. Use fixture only for bundled offline test.")
    parser.add_argument("--fixture-mapping", default="fixtures/synthetic_to_token_mapping.json", help="Value->token JSON for offline fixture mode.")

    sub = parser.add_subparsers(required=True)

    anonymize = sub.add_parser("anonymize", help="Anonymize one DOCX.")
    anonymize.add_argument("--input", required=True)
    anonymize.add_argument("--output", required=True)
    anonymize.add_argument("--report", default=None)
    anonymize.set_defaults(func=cmd_anonymize)

    eval_p = sub.add_parser("eval", help="Run anonymization and compare against tokenized ground truth DOCX.")
    eval_p.add_argument("--input", default="fixtures/synthetic_obezlichivanie_fixture.docx")
    eval_p.add_argument("--ground-truth", default="fixtures/tokenized_obezlichivanie_fixture.docx")
    eval_p.add_argument("--out-dir", default="outputs/eval")
    eval_p.add_argument("--token-to-synthetic-mapping", default="fixtures/token_to_synthetic_mapping.json")
    eval_p.add_argument("--ignore-date-non-target", action=argparse.BooleanOptionalAction, default=True)
    eval_p.set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
