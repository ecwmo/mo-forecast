import pytest
import matplotlib.pyplot as plt

from helpers.anomaly_format import plot_footer


@pytest.mark.parametrize(
    "var_name, footer",
    [
        ("temp", "1971-2000 APHRODITE"),
        ("precip", "2001-2020 GSMaP"),
    ],
)
def test_plot_footer(var_name, footer):
    fig, ax = plt.subplots()

    plot_footer(ax, var_name)

    text_objects = [
        obj
        for obj in ax.get_children()
        if isinstance(obj, plt.Text) and (obj.get_text() == f"*baseline: {footer}")
    ]
    assert len(text_objects) == 1

    plt.close(fig)
