import requests
import time
from time import sleep
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import numpy as  np

def start_driver(url = 'https://instruction.learn.co/staff/students'):
    from selenium import webdriver
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(1)
    return driver


def load_login_data(login_data_file = "/Users/jamesirving/.secret/learn_login.json",
                   verbose=True):
    """Loads in json file from path"""
    with open(login_data_file,'r+') as f:
        import json
        fdata= f.read()
        login_data =  json.loads(fdata)
        
        if verbose:
            print("Loaded json data. Keys:")
            print(login_data.keys())
        return login_data
    
    
    
def github_login(driver,login_data=None):
    """Logs into GitHub Account (for instruction.learn)
    url = 'https://instruction.learn.co/staff/students'
    """
    if login_data is None:
        login_data= load_login_data()
        
    username = driver.find_element_by_xpath('//*[@id="login_field"]')
    username.send_keys(login_data['username'])

    password = driver.find_element_by_xpath('//*[@id="password"]')
    password.send_keys(login_data['password'])

    sign_in = driver.find_element_by_xpath('//*[@id="login"]/form/div[2]/input[8]')
    sign_in.click()
    
def instruct_menu_to_cohort_roster(driver,cohort="pt"):
    import time
    time.sleep(0.5)
    
    cohort_lead =driver.find_element_by_xpath('/html/body/div[1]/nav/div[1]/ul/li[1]/a')
    my_cohorts = driver.find_element_by_xpath('//*[@id="js-parentDropdownLink"]')
    if cohort=="pt":
        cohort_link = driver.find_element_by_xpath('//*[@id="js-childrenList-141"]/ul/li[1]/a')

    elif cohort=="ft":
        cohort_link = driver.find_element_by_xpath('//*[@id="js-sidenavChildrenList-140"]/li[2]/a')
#         return ft_cohort

    actions = ActionChains(driver)
    actions.move_to_element(cohort_lead)
    actions.pause(.5)
    actions.click(my_cohorts)
    actions.pause(.5)

    actions.click(cohort_link)
    return actions.perform()


def cohort_driver_to_csv(driver,output_file='cohort_output.csv',
                         debug=False,load=False,
                        load_kws=None):
    """Exports the table content inside of the driver.page_source to csv file.
    
    Args:
        driver (WebDriver): cohort instruct page's driver
        output_file (str): name of csv file to save.
    
    TO DO: Add link extraction"""
    my_html = driver.page_source
    soup = BeautifulSoup(my_html, 'html.parser')

    table = soup.find("table")
    rows = table.find_all('tr')

    output_rows = []
    for row in rows:
        row_text = row.get_text(separator='\t',strip=True)
        
        if "Links" in row_text:
            row_text=row_text.replace("\tLinks",' ')
        
        profile_links = [x['href'] for x in row.find_all('a')]#
        if debug:
            print(len(row_text.split('t')))
            if 'John' in row_text:
                print("Returning John row object")
                return row
            
        repl_dict={
            ':':' ',
#             ')':' ',
            '\n':' '
        }
        for k,v in repl_dict.items():
            row_text = row_text.replace(k,v)
#         row_text = row_text.replace(':',' ').replace(')',' ').replace('\n',' ')

        output_rows.append(row_text)#row.get_text(separator='\t',strip=True))

    with open(output_file, 'w+') as csvfile:
        csvfile.write('\n'.join(output_rows))
        
    print(f"[i] Successfully saved '{output_file}'")
    
    if load:
#         header = pd.read_csv(output_file,delimiter='\t',nrows=1)
        if load_kws is not None:
            df = pd.read_csv(output_file,delimiter='\t',**load_kws)
        else:
            df = pd.read_csv(output_file,delimiter='\t')
         
        ## Save column names to restore
#         cols = df.columns
        df.reset_index(inplace=True)
        cols = df.drop('index',axis=1).columns
        
        if df["Completed Lessons"].isna().any():
            shift_index = df.loc[(df['Completed Lessons'].isna())].index#.copy()
            
#             ## Preview bad row alignment
#             display(df.loc[shift_index])
            
            ## Replace the column data to match others
            cols_to_swap = {"Completed Lessons":"Last Checkin Note",
                           "Instructor":"Checkins (NoShows)",
                           "Checkins (NoShows)":"Last Checkin Note"}
            
            for bad_col,good_col in cols_to_swap.items():
#             df.loc[shift_index,'Completed Lessons']=df.loc[shift_index,'Last Checkin Note'].copy()
                df.loc[shift_index,bad_col]=df.loc[shift_index,good_col].copy()
            df.loc[shift_index,"Last Checkin Note"]=np.nan
        
        
#             ##Preview changes
#             display(df.loc[shift_index])
        
        
        
        # Drop one of the redundant columns
        drop_col = "Completed Lessons"#'Last Checkin Note'
        df.drop(columns=[drop_col],inplace=True)

        # Restore names to columns
        df.columns = cols
        
        return df
    
    
    
def help():
    print("[i] Workflow:")
    print("driver = start_driver()")
    print('login_data=load_login_data()')
    print("github_login(driver,login_data)")
    print("instruct_menu_to_cohort_roster(driver,cohort='pt')")
    print("df = cohort_driver_to_csv(driver,'pt_cohort_data.csv',load=True)")
help()

