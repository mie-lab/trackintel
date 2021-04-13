import os
import pytest
import numpy as np

import trackintel as ti
from trackintel.visualization.util import a4_figsize


class TestA4_figsize:
    """Tests for a4_figsize() method."""

    def test_parameter(self, caplog):
        """Test different parameter configurations."""
        fig_width, fig_height = a4_figsize(columns=1)
        assert np.allclose([3.30708661, 2.04389193], [fig_width, fig_height])

        fig_width, fig_height = a4_figsize(columns=1.5)
        assert np.allclose([5.07874015, 3.13883403], [fig_width, fig_height])

        fig_width, fig_height = a4_figsize(columns=2)
        assert np.allclose([6.85039370, 4.23377614], [fig_width, fig_height])

        with pytest.raises(ValueError):
            a4_figsize(columns=3)

        a4_figsize(fig_height_mm=250)
        assert "fig_height too large" in caplog.text
