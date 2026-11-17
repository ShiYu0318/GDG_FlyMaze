from flymaze.contact import CONTACT_SEGMENT_WHITELIST


def test_approved_contact_whitelist_contains_only_head_and_physical_forefeet() -> None:
    assert CONTACT_SEGMENT_WHITELIST == {
        "head": "c_head",
        "left_forefoot": "lf_tarsus5",
        "right_forefoot": "rf_tarsus5",
    }
    assert not any("tibia" in segment for segment in CONTACT_SEGMENT_WHITELIST.values())
    assert not any("tarsus4" in segment for segment in CONTACT_SEGMENT_WHITELIST.values())
