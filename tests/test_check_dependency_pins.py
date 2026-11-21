"""6.26: dependency-pin checks reject wildcard and range pins."""

from __future__ import annotations

import pytest

from scripts import check_dependency_pins


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("pkg==1.2.3", True),
        ("pkg==1.2.3; python_version>'3.8'", True),
        ("pkg[extra]==2.0", True),
        ("pkg==1.*", False),
        ("pkg==1.0.*", False),
        ("pkg>=1.0", False),
        ("pkg~=1.0", False),
        ("pkg===1.0", False),
        ("pkg!=1.0", False),
        ("pkg", False),
        ("pkg==1.0,==2.0", False),
        ("pkg==1.0,!=1.0.1", False),
    ],
)
def test_requirement_exactness_rejects_wildcards_and_ranges(
    requirement: str, expected: bool
) -> None:
    assert check_dependency_pins._requirement_is_exact(requirement) is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.2.3", True),
        ("0.27.1", True),
        ("1.0.0-beta.1", True),
        ("1.x", False),
        ("1.0.0 - 2.0.0", False),
        ("^1.0.0", False),
        ("~1.0.0", False),
        (">=1.0.0", False),
        ("<2.0.0", False),
        ("latest", False),
        ("*", False),
        ("1.0", False),
        ("v1.2.3", False),
    ],
)
def test_javascript_version_must_be_a_plain_semver(version: str, expected: bool) -> None:
    assert check_dependency_pins._version_is_exact(version) is expected
