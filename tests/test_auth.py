from dashboard.auth import check_credentials, parse_users


def test_parse_users_splits_pairs_and_keeps_colons_in_passwords():
    users = parse_users("tim1:abc123; tim2:pa:ss ;broken;:nouser;nopass:")
    assert users == {"tim1": "abc123", "tim2": "pa:ss "}


def test_parse_users_empty_means_no_users():
    assert parse_users("") == {}


def test_check_credentials():
    users = {"tim1": "abc123"}
    assert check_credentials(users, "tim1", "abc123")
    assert check_credentials(users, " tim1 ", "abc123")
    assert not check_credentials(users, "tim1", "wrong")
    assert not check_credentials(users, "ghost", "")
    assert not check_credentials({}, "", "")
