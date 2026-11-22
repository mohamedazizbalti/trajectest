from pathlib import Path

import pytest
from pydantic import ValidationError

from trajectest.test_case import load_test_case, load_test_cases

EXAMPLE_TESTS = Path(__file__).parent.parent / "examples" / "support_agent" / "tests"


def test_loads_every_example_test_case():
    test_cases = load_test_cases(EXAMPLE_TESTS)
    assert len(test_cases) == 4
    refund = next(case for case in test_cases if "approval" in case.name)
    assert refund.steps[0].expect_interrupt.contains == "approval"
    assert refund.steps[0].resume == {"approved": True}
    assert refund.steps[0].tool_order == "in_order"


def test_rejects_unknown_fields_so_typos_fail_loudly(tmp_path):
    typo = tmp_path / "typo.yaml"
    typo.write_text("name: x\nagent: a:b\ninput: hi\nforbiden_tools: [cancel_order]\n")
    with pytest.raises(ValidationError, match="forbiden_tools"):
        load_test_case(typo)
