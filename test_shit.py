import numpy as np
import plotly.graph_objects as go
from plotly_resampler import register_plotly_resampler, FigureResampler


def test_shit():
    print("testing shit")
    register_plotly_resampler(mode='auto')

    x = np.random.uniform(low=3, high=6, size=(500,))
    y = np.random.uniform(low=3, high=6, size=(500,))

    # Build figure
    fig = go.Figure()

    # Add scatter trace with medium sized markers
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=x,
            y=y,
            marker=dict(
                color='LightSkyBlue',
                size=20,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
            showlegend=False
        )
    )

    fig.show_dash()


if __name__ == '__main__':
    test_shit()
