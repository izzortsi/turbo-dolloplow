import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(
        x=[1, 2, 3, 4],
        y=[2, 1, 3, 4],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[0.1, 0.2, 0.1, 0.1],
            arrayminus=[0.2, 0.4, 1, 0.2])
        ))
fig.show()