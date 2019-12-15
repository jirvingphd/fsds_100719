"""A collection of functions containing convenient links to documentation and other resources."""
repo_links = {'python_package':('fsds_100719',
                                    'https://github.com/jirvingphd/fsds_100719'),
        'fulltime_notes':('fsds_100719_cohort_notes',
                                    'https://github.com/jirvingphd/fsds_100719_cohort_notes'),
            'parttime_notes':('fsds_pt_100719_cohort_notes',
                                    "https://github.com/jirvingphd/fsds_pt_100719_cohort_notes")}      
# def youtube_playlists():
#     pass
def html_colors():
    url = "https://www.w3schools.com/colors/colors_names.asp"
    print(url)

def matplotlib_links():
    """ Display links to matplotlib documentation and references.
    """
    from IPython.display import display, Markdown
    import sys
    
    txt = '''
    **Matplotlib References**
    - [Comprehensive Matplotlib OOP vs plt Blog Post](https://dev.to/skotaro/artist-in-matplotlib---something-i-wanted-to-know-before-spending-tremendous-hours-on-googling-how-tos--31oo)
    - [Markers](https://matplotlib.org/3.1.1/api/markers_api.html)
    - [Colors](https://matplotlib.org/3.1.0/gallery/color/named_colors.html )
    - [Text](https://matplotlib.org/3.1.0/tutorials/text/text_intro.html )
    - [Text Properties](https://matplotlib.org/3.1.1/tutorials/text/text_props.html)
    '''
    if 'google.colab' in sys.modules:
        print(txt)
    else:
        display(Markdown(txt))


def string_formatting():
    print('[i] See link for overview table of string formatting.')
    print('- https://mkaz.blog/code/python-string-format-cookbook/')
    print('- Example: "${:,.2f}" adds a $, uses commas as thousands separator, and displays 2 decimal points.')
    
    

class Link:
    def __init__(self,name=None,url=None):
        self.name = None
        self.url = None
        
    def click(self):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        
    def __md__(self):
        from IPython.display import Markdown,display,HTML
        ## Markdown clickable
        mkdown=[]
        [mkdown.append(f"- {k} :\n\t[{v[0]}]({v[1]})") for k,v in repo_links.items()]
        mkdown = '\n'.join(mkdown)
        display(Markdown(mkdown))
        msg= mkdown
        
    def __str__(self):
        msg = f"- {self.name} :\n\t:{self.url})"
        # msg = '\n'.join(msg)
        return msg
    def __repr__(self):
        # msg=[]
        # [msg.append(f"- {k} :\n\t- Name: {v[0]}\n\t- Link: {v[1]})") for k,v in repo_links.items()]
        # msg = '\n'.join(msg)
        msg = f"- {self.name} :\n\t:{self.url})"

        return msg
        
class LinkLibray:
    def __init__(self,topic):
        self._topic = topic
        self._id = f"LinkLibrary for {self._topic} Links"
        
    def __repr___(self):
        return self._id
    
    def __str__(self):
        return self._id
    # def __init__(self,*args):
    #     for arg in args:
    #         if isinstance(arg,Link):
    #             self.__setattr__()
  
class Library:
    
    def __init__(self):
        
        repo_links = {'python_package':('fsds_100719',
                                         'https://github.com/jirvingphd/fsds_100719'),
                'fulltime_notes':('fsds_100719_cohort_notes',
                                         'https://github.com/jirvingphd/fsds_100719_cohort_notes'),
                  'parttime_notes':('fsds_pt_100719_cohort_notes',
                                         "https://github.com/jirvingphd/fsds_pt_100719_cohort_notes")}
        
        cohort_repo_links = LinkLibray("Repo Links")
        # links={}
        for k,v in repo_links.items():
            name,url= v[0],v[1]
            link = Link(name,url)
            setattr(cohort_repo_links,k,link)
           
        self.cohort_repo_links  = cohort_repo_links
            
    
    #  matplotlib_links = {'artist_blogpost'
    # - [Comprehensive Matplotlib OOP vs plt Blog Post](https://dev.to/skotaro/artist-in-matplotlib---something-i-wanted-to-know-before-spending-tremendous-hours-on-googling-how-tos--31oo)
    # - [Markers](https://matplotlib.org/3.1.1/api/markers_api.html)
    # - [Colors](https://matplotlib.org/3.1.0/gallery/color/named_colors.html )
    # - [Text](https://matplotlib.org/3.1.0/tutorials/text/text_intro.html )
    # - [Text Properties](https://matplotlib.org/3.1.1/tutorials/text/text_props.html))
        
    

def cohort_links(md=False,ret=False):
    """Displays quick reference url links and info."""
    
    repo_links = {'Cohort Python Package':('fsds_100719',
                                         'https://github.com/jirvingphd/fsds_100719'),
                'full-time note repo':('fsds_100719_cohort_notes',
                                         'https://github.com/jirvingphd/fsds_100719_cohort_notes'),
                  'part-time note repo':('fsds_pt_100719_cohort_notes',
                                         "https://github.com/jirvingphd/fsds_pt_100719_cohort_notes")}    
    from IPython import display 
    import math
    print("[i] Links to cohort note repositories:")
    print("---"*int(math.floor((80/3)-2)))
     
    if md:
        ## Markdown clickable
        mkdown=[]
        [mkdown.append(f"- {k} :\n\t[{v[0]}]({v[1]})") for k,v in repo_links.items()]
        mkdown = '\n'.join(mkdown)
        display.Markdown(mkdown)  
        msg= mkdown
        
    else:
        ## optimized for print 
        msg=[]
        [msg.append(f"- {k} :\n\t- Name: {v[0]}\n\t- Link: {v[1]})") for k,v in repo_links.items()]
        msg = '\n'.join(msg)
        print(msg)
        
    if ret: 
        return msg

        
    # display.display(display.Markdown(msg))
    # return msg

    




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
    
    

def pandas_time_freq_codes():
    from IPython.display import Markdown, display
    
    import sys

    mkdwn_notes = """
    - **Pandas Frequency Abbreviations**<br><br>

    | Alias | 	Description |
    |----|-----|
    |B|	business day frequency|
    |C|	custom business day frequency|
    |D|	calendar day frequency|
    |W|	weekly frequency|
    |M|	month end frequency|
    |SM|	semi-month end frequency (15th and end of month)|
    |BM|	business month end frequency|
    |CBM|	custom business month end frequency|
    |MS|	month start frequency|
    |SMS|	semi-month start frequency (1st and 15th)|
    |BMS|	business month start frequency|
    |CBMS|	custom business month start frequency|
    |Q|	quarter end frequency|
    |BQ|	business quarter end frequency|
    |QS|	quarter start frequency|
    |BQS|	business quarter start frequency|
    |A|, Y	year end frequency|
    |BA|, BY	business year end frequency|
    |AS|, YS	year start frequency|
    |BAS|, BYS	business year start frequency|
    |BH|	business hour frequency|
    |H|	hourly frequency|
    |T|, min	minutely frequency|
    |S|	secondly frequency|
    |L|, ms	milliseconds|
    |U|, us	microseconds|
    |N|	nanoseconds|
    """

    # **Time/data properties of Timestamps**<br><br>

    # |Property|	Description|
    # |---|---|
    # |year|	The year of the datetime|
    # |month|	The month of the datetime|
    # |day|	The days of the datetime|
    # |hour|	The hour of the datetime|
    # |minute|	The minutes of the datetime|
    # |second|	The seconds of the datetime|
    # |microsecond|	The microseconds of the datetime|
    # |nanosecond|	The nanoseconds of the datetime|
    # |date|	Returns datetime.date (does not contain timezone information)|
    # |time|	Returns datetime.time (does not contain timezone information)|
    # |timetz|	Returns datetime.time as local time with timezone information|
    # |dayofyear|	The ordinal day of year|
    # |weekofyear|	The week ordinal of the year|
    # |week|	The week ordinal of the year|
    # |dayofweek|	The number of the day of the week with Monday=0, Sunday=6|
    # |weekday|	The number of the day of the week with Monday=0, Sunday=6|
    # |weekday_name|	The name of the day in a week (ex: Friday)|
    # |quarter|	Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.|
    # |days_in_month|	The number of days in the month of the datetime|
    # |is_month_start|	Logical indicating if first day of month (defined by frequency)|
    # |is_month_end|	Logical indicating if last day of month (defined by frequency)|
    # |is_quarter_start|	Logical indicating if first day of quarter (defined by frequency)|
    # |is_quarter_end|	Logical indicating if last day of quarter (defined by frequency)|
    # |is_year_start|	Logical indicating if first day of year (defined by frequency)|
    # |is_year_end|	Logical indicating if last day of year (defined by frequency)|
    # |is_leap_year|	Logical indicating if the date belongs to a leap year|
    # """
    if 'google.colab' in sys.modules:
        print(mkdwn_notes)    
    else:
        display(Markdown(mkdwn_notes))
    return

