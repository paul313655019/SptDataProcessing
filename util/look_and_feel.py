"""
A place to put all the plot styling to visualize settings.
"""
def plotly_style_tracks(fig):
    fig.update_layout(
        width=600, 
        height=600,
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgb(52, 52, 52)',
    )
    fig.update_xaxes(
        range=[0, 52000], 
        title=None, 
        showticklabels=False, 
        showgrid=False, 
        zeroline=False
        )
    fig.update_yaxes(
        range=[0, 52000],
        title=None, 
        showticklabels=False, 
        showgrid=False, 
        zeroline=False
        )
    # Original figure is 800 pixels with 65nm per pixel, so 52000 is the max value
    # fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
    # Show the plot
    
    return fig

def set_plotly_config(fig):
    """
    Set the configuration for the Plotly figure.
    Wrapper for fig.show(config=config)
    """
    config = {
        'toImageButtonOptions': {
            'scale': 1,
            'format': 'svg',
            'filename': 'figure.svg',
            'width': 800,
            'height': 600
        }
    }
    return fig.show(config=config)