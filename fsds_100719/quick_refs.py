"""A collection of functions containing convenient links to documentation and other resources."""

def youtube_playlists():
    pass
def matplotlib_links():
    """ Display links to matplotlib documentation and references.
    """
    from IPython.display import display, Markdown
    display(Markdown('''
    **Matplotlib References**
    - [Markers](https://matplotlib.org/3.1.1/api/markers_api.html)
    - [Colors](https://matplotlib.org/3.1.0/gallery/color/named_colors.html )
    - [Text](https://matplotlib.org/3.1.0/tutorials/text/text_intro.html )
    - [Text Properties](https://matplotlib.org/3.1.1/tutorials/text/text_props.html)
    '''))