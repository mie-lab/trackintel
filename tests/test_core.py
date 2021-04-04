import trackintel as ti


class TestPrint_version:
    """Tests for print_version() method."""

    def test_print_version(self, capsys):
        """Check if the correct message is printed."""
        ti.print_version()
        captured = capsys.readouterr()
        assert "This is trackintel v" in captured.out
