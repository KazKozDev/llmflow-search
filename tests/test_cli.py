from deep_research.cli import build_parser


def test_run_print_controls_are_available() -> None:
    args = build_parser().parse_args(["run", "research question"])
    assert args.quiet is False
    assert args.no_report is False

    quiet_args = build_parser().parse_args(["run", "research question", "--quiet", "--no-report"])
    assert quiet_args.quiet is True
    assert quiet_args.no_report is True
