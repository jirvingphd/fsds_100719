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
    
def mount_google_drive_shared_url(url,force=False):
    import sys
    if ("google.colab" in sys.modules) | (force==True):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print(f'\n[i] Lesson Folder URL:\n\t{url}')
            print('[i] Add this folder to your own Google Drive.')
            print('\t- Open Sidebar panel (> symmbol on left edge of screen) and click `Files` tab.\n\tFind your folder, somewhere inside of "/content/drive/My Drive/"')
            print('\t- Find the data file inside this folder and Right Click > "Copy Path"')
            print('[i] Enter that path into the next cell.')
        except:
            print('Something went wrong. Are you using Google Colab?')
    else:
        raise Exception('This function only works if using Google Colab.')        
        
        

# # def cohort_resources(topic='student_resource_folder'):
# #     """Displays quick reference url links and info.
# #     Args:
# #         topic (str): selects which reference info to show.
# #             - `student_resource_folder` : data science gdrive url
# #             - `fsds` :documentaion url"""
            
# #     if 'student_resource_folder' in topic:

        
# #     if 'fsds' in topic:
# #         print('fsds_100719 Package Documentation:')
        
# class Documentation():
#     """"Keyword-sorted package documentation links/resources"""
#     def __init__(self,name='package'):
#         self.__name__ = "package"
#         # print(f'Documentation loaded for {self.__name__}.')
#     homepage = "http://"
#     usage = "import fsds_100719 as fs"
#     topics = {}
    

    
        
# class QuickReference():
#     def __init__(self):
#         print('__init__() ran.')
#         print('fsds_100719 docs: https://fsds.readthedocs.io/en/latest/')
        
#     @property
#     def matplotlib(self):
#         docs = Documentation()
#         docs.topics['text'] ='https://matplotlib.org/3.1.1/tutorials/text/text_intro.html'
#         self.__matplotlib__ = Documentation()
        
        
    
#     cohort_resources = """- Data Science Student Resources:\n\t 
#     'https://flatiron.online/StudentResourcesGdrive"""
    
#     cohort_videos = """ """

#     @property
#     def matplotlib_docs(self):
#         self.docs['text'] = 'https://matplotlib.org/3.1.1/tutorials/text/text_intro.html'
    