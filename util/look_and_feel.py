"""
A place to put all the plot styling to visualize settings.
"""
def plotly_style_tracks(fig):
    fig.update_layout(width=600, height=600)
    fig.update_xaxes(range=[0, 52000])
    fig.update_yaxes(range=[0, 52000])
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
    # fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
    # Show the plot
    fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig