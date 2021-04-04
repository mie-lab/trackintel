import os
import pytest

import trackintel as ti


class TestUsers:
    """Tests for the UsersAccessor."""

    def test_accessor(self):
        """Test if the as_users accessor checks the required column for users."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))

        users = pfs.groupby("user_id", as_index=False).mean()
        users.rename(columns={"user_id": "id"}, inplace=True)
        assert users.as_users

        users = users.drop(["id"], axis=1)
        with pytest.raises(AttributeError):
            users.as_users
