"""A collection of functions containing convenient links to documentation and other resources."""
from IPython.display import Markdown, display, HTML

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
        return msg
        
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
    
def ts_date_str_formatting():
    from IPython.display import Markdown, display, HTML
    table="""
    Formatting follows the Python datetime <strong><a href='http://strftime.org/'>strftime</a></strong> codes.<br>
    The following examples are based on <tt>datetime.datetime(2001, 2, 3, 16, 5, 6)</tt>:
    <br><br>

    <table style="display: inline-block">  
    <tr><th>CODE</th><th>MEANING</th><th>EXAMPLE</th><tr>
    <tr><td>%Y</td><td>Year with century as a decimal number.</td><td>2001</td></tr>
    <tr><td>%y</td><td>Year without century as a zero-padded decimal number.</td><td>01</td></tr>
    <tr><td>%m</td><td>Month as a zero-padded decimal number.</td><td>02</td></tr>
    <tr><td>%B</td><td>Month as locale’s full name.</td><td>February</td></tr>
    <tr><td>%b</td><td>Month as locale’s abbreviated name.</td><td>Feb</td></tr>
    <tr><td>%d</td><td>Day of the month as a zero-padded decimal number.</td><td>03</td></tr>  
    <tr><td>%A</td><td>Weekday as locale’s full name.</td><td>Saturday</td></tr>
    <tr><td>%a</td><td>Weekday as locale’s abbreviated name.</td><td>Sat</td></tr>
    <tr><td>%H</td><td>Hour (24-hour clock) as a zero-padded decimal number.</td><td>16</td></tr>
    <tr><td>%I</td><td>Hour (12-hour clock) as a zero-padded decimal number.</td><td>04</td></tr>
    <tr><td>%p</td><td>Locale’s equivalent of either AM or PM.</td><td>PM</td></tr>
    <tr><td>%M</td><td>Minute as a zero-padded decimal number.</td><td>05</td></tr>
    <tr><td>%S</td><td>Second as a zero-padded decimal number.</td><td>06</td></tr>
    </table>
    <table style="display: inline-block">
    <tr><th>CODE</th><th>MEANING</th><th>EXAMPLE</th><tr>
    <tr><td>%#m</td><td>Month as a decimal number. (Windows)</td><td>2</td></tr>
    <tr><td>%-m</td><td>Month as a decimal number. (Mac/Linux)</td><td>2</td></tr>
    <tr><td>%#x</td><td>Long date</td><td>Saturday, February 03, 2001</td></tr>
    <tr><td>%#c</td><td>Long date and time</td><td>Saturday, February 03, 2001 16:05:06</td></tr>
    </table>  
    """ 
    display(HTML(table))


def ts_pandas_freq_aliases_anchored(ipython=True, return_str=False):
    from IPython.display import Markdown, display, HTML
    
    print("PANDAS ANCHORED TIME FREQUENCY ALIASES")
    print("[i] Documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets")
    anchored="""|Alias|Description|
    | --- | --- |
    |W-SUN|weekly frequency (Sundays). Same as ‘W’|
    |W-MON|weekly frequency (Mondays)|
    |W-TUE|weekly frequency (Tuesdays)|
    |W-WED|weekly frequency (Wednesdays)|
    |W-THU|weekly frequency (Thursdays)|
    |W-FRI|weekly frequency (Fridays)|
    |W-SAT|weekly frequency (Saturdays)|
    |(B)Q(S)-DEC|quarterly frequency, year ends in December. Same as ‘Q’|
    |(B)Q(S)-JAN|quarterly frequency, year ends in January|
    |(B)Q(S)-FEB|quarterly frequency, year ends in February|
    |(B)Q(S)-MAR|quarterly frequency, year ends in March|
    |(B)Q(S)-APR|quarterly frequency, year ends in April|
    |(B)Q(S)-MAY|quarterly frequency, year ends in May|
    |(B)Q(S)-JUN|quarterly frequency, year ends in June|
    |(B)Q(S)-JUL|quarterly frequency, year ends in July|
    |(B)Q(S)-AUG|quarterly frequency, year ends in August|
    |(B)Q(S)-SEP|quarterly frequency, year ends in September|
    |(B)Q(S)-OCT|quarterly frequency, year ends in October|
    |(B)Q(S)-NOV|quarterly frequency, year ends in November|
    |(B)A(S)-DEC|annual frequency, anchored end of December. Same as ‘A’|
    |(B)A(S)-JAN|annual frequency, anchored end of January|
    |(B)A(S)-FEB|annual frequency, anchored end of February|
    |(B)A(S)-MAR|annual frequency, anchored end of March|
    |(B)A(S)-APR|annual frequency, anchored end of April|
    |(B)A(S)-MAY|annual frequency, anchored end of May|
    |(B)A(S)-JUN|annual frequency, anchored end of June|
    |(B)A(S)-JUL|annual frequency, anchored end of July|
    |(B)A(S)-AUG|annual frequency, anchored end of August|
    |(B)A(S)-SEP|annual frequency, anchored end of September|
    |(B)A(S)-OCT|annual frequency, anchored end of October|
    |(B)A(S)-NOV|annual frequency, anchored end of November|"""
    return display(Markdown(anchored))

    

def ts_pandas_freq_aliases(ipython=True, return_str=False):
    from IPython.display import Markdown, display, HTML
    
    import sys
    print("PANDAS TIME FREQUENCY ALIASES")
    print("[i] Documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases")
    mkdwn_notes = """| Alias | Description |
    |----|-----|
    |B|business day frequency|
    |C|custom business day frequency|
    |D|calendar day frequency|
    |W|weekly frequency|
    |M|month end frequency|
    |SM|semi-month end frequency (15th and end of month)|
    |BM|business month end frequency|
    |CBM|custom business month end frequency|
    |MS|month start frequency|
    |SMS|semi-month start frequency (1st and 15th)|
    |BMS|business month start frequency|
    |CBMS|custom business month start frequency|
    |Q|quarter end frequency|
    |BQ|business quarter end frequency|
    |QS|quarter start frequency|
    |BQS|business quarter start frequency|
    |A, Y| year end frequency|
    |BA, BY |business year end frequency|
    |AS, YS |year start frequency|
    |BAS, BYS |business year start frequency|
    |BH|business hour frequency|
    |H|hourly frequency|
    |T, min |minutely frequency|
    |S|secondly frequency|
    |L, ms|milliseconds|
    |U, us |microseconds|
    |N|nanoseconds|
    """

    if ipython == False:
        print(mkdwn_notes)
        if return_str:
            return mkdwn_notes
    else:
        mkdown_notes_md = Markdown(mkdwn_notes)
        return display(mkdown_notes_md)




def ts_datetime_object_properties():
    from IPython.display import Markdown, display, HTML
    print('PYTHON DATETIME OBJECT ATTRIBUTES/METHODS')

    print('[i] Documentation: https://docs.python.org/2/library/datetime.html#datetime-objects')
    
    datetime_notes="""|Property|	Description|
    |---|---|
    |year|	The year of the datetime|
    |month|	The month of the datetime|
    |day|	The days of the datetime|
    |hour|	The hour of the datetime|
    |minute|	The minutes of the datetime|
    |second|	The seconds of the datetime|
    |microsecond|	The microseconds of the datetime|
    |nanosecond|	The nanoseconds of the datetime|
    |date|	Returns datetime.date (does not contain timezone information)|
    |time|	Returns datetime.time (does not contain timezone information)|
    |timetz|	Returns datetime.time as local time with timezone information|
    |dayofyear|	The ordinal day of year|
    |weekofyear|	The week ordinal of the year|
    |week|	The week ordinal of the year|
    |dayofweek|	The number of the day of the week with Monday=0, Sunday=6|
    |weekday|	The number of the day of the week with Monday=0, Sunday=6|
    |weekday_name|	The name of the day in a week (ex: Friday)|
    |quarter|	Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.|
    |days_in_month|	The number of days in the month of the datetime|
    |is_month_start|	Logical indicating if first day of month (defined by frequency)|
    |is_month_end|	Logical indicating if last day of month (defined by frequency)|
    |is_quarter_start|	Logical indicating if first day of quarter (defined by frequency)|
    |is_quarter_end|	Logical indicating if last day of quarter (defined by frequency)|
    |is_year_start|	Logical indicating if first day of year (defined by frequency)|
    |is_year_end|	Logical indicating if last day of year (defined by frequency)|
    |is_leap_year|	Logical indicating if the date belongs to a leap year|
    """
    
    return display(Markdown(datetime_notes))


# def prob_formulas():
#     """Overview of critical probability/permutation/factorial equations"""
#     pass

    
    ##
    

# def prob_combinations():
#     pass
    
# def prob_laws():
#     from IPython.display import Markdown, display, HTML

#     axioms="""#### Probability axioms

#     1.  Positivity : Prob is always 0 \<= P(E) \<=1

#     2.  Probability of a certain event: P(S)=1

#     3.  Additivity Union of 2 exclusive sets = sum prob of individual events
#         happening <br>
#     If $A\cap B = \emptyset $, then $P(A\cup B) = P(A) + P(B)$"""
#     display(Markdown(axioms))


def prob_permutations():
    from IPython.display import display,Markdown,HTML

    permutations=\
r"""### Permutations
- **Permutations have to do with arranging objects.**
- Order is important

### Permutations of $n$:
> ***How many ways to arrange $n$ objects?***

$$P(n) = n!$$
- where $n$ = total number of elements

### Permutations without replacement (the $k$-permutation of $n$)
> ***How many ways to select $k$ elements out of a pool of $n$ objects?***

$$ \large P_{k}^{n}= \dfrac{n!}{(n-k)!}$$ 
<center> this is known as a $k$-permutation of $n$. </center>

### Permutations with Replacement:

$$ \large P_{j}^{n} = n^j $$
- where $n$ = total number of elements
- $j$ = number of positions to fill

#### Sometimes called **Variations** with notation:

$$ \large \bar{V}_{j}^{n} = n^j$$


### Permutations with repeated elements
- e.g. selecting random letters from the word TENNESEE

$$\dfrac{n!}{n_1!n_2!\ldots n_j!}$$


"""
    import IPython.display as dp
    display(dp.Markdown(permutations))
# prob_permutations()


def prob_combinations():
    import IPython.display as dp

    combinations=\
r"""### Combinations

> ***How many ways can we create a subset $k$ out of $n$ objects?***

- Combinations are unordered.

- **The # of combinations equals number of variations, over the number of permutations:**
$$ \large C_{p}^{n}  = \frac{V_{p}^{n}}{P_p} = \frac{n!}{p!(n-p)!} $$


- Also represented as:
$$\displaystyle\binom{n}{k} = \dfrac{P_{k}^{n}}{k!}=\dfrac{ \dfrac{n!}{(n-k)!}}{k!} = \dfrac{n!}{(n-k)!k!}$$

"""
    dp.display(dp.Markdown(combinations))
prob_combinations()


def evaluation_metrics(markdown=True):
    from IPython.display import Markdown,display,HTML
    if markdown:
        formula = r'''$$ \text{Precision} = \frac{\text{Number of True Positives}}{\text{Number of Predicted Positives}} $$    
        $$ \text{Recall} = \frac{\text{Number of True Positives}}{\text{Number of Actual Total Positives}} $$  
$$ \text{Accuracy} = \frac{\text{Number of True Positives + True Negatives}}{\text{Total Observations}} $$

$$ \text{F1 score} = 2 * \frac{\text{Precision * Recall}}{\text{Precision + Recall}} $$'''        
        display(Markdown(formula))


def statistical_power(return_url = False):
    """Displays link to interactive statistical power calculator. 
    Returns url if return_url = True (Default is False)"""
    url = 'https://rpsychologist.com/d3/NHST/'
    print('[i] Interactive statistical power calculator: https://rpsychologist.com/d3/NHST/')
    if return_url:
        return url
    
    
def hypothesis_testing():
    import IPython.display as dp
    notes = """### Assumption Tests
 
|Assumption test| Function |
| --- | --- |
| **Normality**| `scipy.stats.normaltest`|
| **Equal Variance** | `scipy.stats.levene`|


### Hypothesis Tests

| Parametric tests (means) | Function | Nonparametric tests (medians) | Function |
| --- | --- | --- | --- |
| **1-sample t test** |`scipy.stats.ttest_1samp()`|  **1-sample Wilcoxon** |`scipy.stats.wilcoxon`|
| **2-sample t test** |`scipy.stats.ttest_ind()` | **Mann-Whitney U test** |`scipy.stats.mannwhitneyu()`|
| **One-Way ANOVA** | `scipy.stats.f_oneway()` | **Kruskal-Wallis** | `scipy.stats.kruskal` | 

 
 ### Post-Hoc Tests/Calculations
 
 | Post-Hoc Tests/Calculatons|Function|
 |--- | --- |
 |**Tukey's Pairwise Comparisons** | `statsmodels.stats.multicomp.pairwise_tukeyhsd`|
 |**Effect Size**| `Cohens_d`|
 |**Statistical Power** | `statsmodels.stats.power`:<br>  `TTestIndPower` , `TTestPower`
 """
    return dp.Markdown(notes)

