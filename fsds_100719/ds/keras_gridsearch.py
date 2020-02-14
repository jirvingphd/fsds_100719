import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
def plot_confusion_matrix(conf_matrix, classes = None, normalize=True,
                          title='Confusion Matrix', cmap="Blues",
                          print_raw_matrix=False,
                          fig_size=(7,8)):
    """Check if Normalization Option is Set to True. 
    If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function.
    Note: Taken from bs_ds and modified
    - Can pass a tuple of (y_true,y_pred) instead of conf matrix.
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    
    ## make confusion matrix if given tuple of y_true,y_pred
    if isinstance(conf_matrix, tuple):
        y_true = conf_matrix[0].copy()
        y_pred = conf_matrix[1].copy()
        
        if y_true.ndim>1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim>1:
            y_pred = y_pred.argmax(axis=1)
        cm = metrics.confusion_matrix(y_true,y_pred)
    else:
        cm = conf_matrix
        
    ## Generate integer labels for classes
    if classes is None:
        classes = list(range(len(cm)))  
        
    ## Normalize data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt='.2f'
    else:
        fmt= 'd'
        
        
    fontDict = {
        'title':{
            'fontsize':16,
            'fontweight':'semibold',
            'ha':'center',
            },
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'xtick_labels':{
            'fontsize':10,
            'fontweight':'normal',
    #             'rotation':45,
            'ha':'right',
            },
        'ytick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':0,
            'ha':'right',
            },
        'data_labels':{
            'ha':'center',
            'fontweight':'semibold',

        }
    }

    # Create plot
    fig,ax = plt.subplots(figsize=fig_size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**fontDict['title'])
    plt.colorbar()

    tick_marks = classes#np.arange(len(classes))


    plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
    plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])

    # Determine threshold for b/w text
    thresh = cm.max() / 2.

    # fig,ax = plt.subplots()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 color='darkgray',**fontDict['data_labels']) #color="white" if cm[i, j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label',**fontDict['ylabel'])
    plt.xlabel('Predicted label',**fontDict['xlabel'])

    if print_raw_matrix:
        print_title = 'Raw Confusion Matrix Counts:'
        print('\n',print_title)
        print(conf_matrix)


    fig = plt.gcf()
    return fig


def plot_keras_history(history,figsize=(10,4),subplot_kws={}):
    if hasattr(history,'history'):
        history=history.history
    #     history = results.history
    fig,axes=plt.subplots(ncols=2,figsize=figsize,**subplot_kws)
    
    ax=axes[0]
    ax.plot(history['val_loss'],label='val_loss')
    ax.plot(history['loss'], label='loss')
    ax.legend()
    
    ax=axes[1]
    ax.plot(history['val_accuracy'],label='val_accuracy')
    ax.plot(history['accuracy'], label='accuracy')

    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    
def evaluate_model(y_true, y_pred,history=None):

    if y_true.ndim>1:
        y_true = y_true.argmax(axis=1)
    if y_pred.ndim>1:
        y_pred = y_pred.argmax(axis=1)   
        
    if history is not None:
        plot_keras_history(history)
    num_dashes=20
    print('\n')
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)

    print(metrics.classification_report(y_true,y_pred))
    
    fig = plot_confusion_matrix((y_true,y_pred))
    plt.show()
    


class Timer():
    
    def __init__(self, start=True,time_fmt='%m/%d/%y - %T'):
        import tzlocal
        import datetime as dt
        
        self.tz = tzlocal.get_localzone()
        self.fmt= time_fmt
        self._created = dt.datetime.now(tz=self.tz)
        
        if start:
            self.start()
            
    def get_time(self):
        import datetime as dt
        return dt.datetime.now(tz=self.tz)

        
    def start(self,verbose=True):
        self._laps_completed = 0
        self.start_ = self.get_time()
        if verbose: 
            print(f'\n[i] Timer started at {self.start_.strftime(self.fmt)}')
    
    def stop(self, verbose=True):
        self._laps_completed += 1
        self.end = self.get_time()
        self.elapsed = self.end -  self.start_
        if verbose: 
            print(f'[i] Timer stopped at {self.end.strftime(self.fmt)}')
            print(f'  - Total Time: {self.elapsed}')
            
    def __call__(self, verbose=True):
        self._laps_completed += 1
        self.end = self.get_time()
        self.elapsed = self.end -  self.start_
        if verbose: 
            print(f'[i] Clock Time: {self.end.strftime(self.fmt)}')
            print(f'  - Elapsed Time: {self.elapsed}')
    
    
from sklearn.metrics import make_scorer
def my_custom_scorer(y_true,y_pred,verbose=True):#,scoring='accuracy',verbose=True):
    """My custom score function to use with sklearn's GridSearchCV
    Maximizes the average accuracy per class using a normalized confusion matrix"""

    import sklearn.metrics as metrics
    from sklearn.metrics import confusion_matrix
    import numpy as np

    ## reduce dimensions of y_train and y_test
    if y_true.ndim>1:            
        y_true = y_true.argmax(axis=1)

    if y_pred.ndim>1:
        y_pred = y_pred.argmax(axis=1)
        
    evaluate_model(y_true,y_pred)
    print('\n\n')
    return metrics.accuracy_score(y_true,y_pred)



def get_secret_password(file):
    with open(file) as file:
        import json
        gmail = json.loads(file.read())
    # email_notification()
    print(gmail.keys())
    return gmail


def email_notification(password_obj=None,subject='GridSearch Finished',
                       msg='The GridSearch is now complete.'):
    """Sends email notification from gmail account using previously encrypyted password  object (an instance
    of EncrypytedPassword). 
    Args:
        password_obj (dict): Login info dict with keys: username,password.
        subject (str):Text for subject line.
        msg (str): Text for body of email. 

    Returns:
        Prints `Email sent!` if email successful. 
    """
    if password_obj is None:
        raise Exception('You must provide the password_obj.')
        # gmail = get_secret_password()
    else:
        assert ('username' in password_obj)&('password' in password_obj)
        gmail = password_obj
        
    if isinstance(msg,str)==False:
        msg=str(msg)
        
    
    # import required packages
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email import encoders
    

    ## WRITE EMAIL
    message = MIMEMultipart()
    message['Subject'] =subject
    message['To'] = gmail['username']
    message['From'] = gmail['username']
    message.attach(MIMEText(msg,'plain'))
    text_message = message.as_string()


    # Send email request
    try:
        with  smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
            
            server.login(gmail['username'],gmail['password'])
            server.sendmail(gmail['username'],gmail['username'], text_message)#text_message)
            server.close()
            print(f"Email sent to {gmail['username']}!")
        
    except Exception as e:
        print(e)
        print('Something went wrong')
       
       
       
def prepare_gridsearch_report(grid_search,X_test,y_test,
                              save_path = 'results/emails/'):
    """Creates a text report with grid search results 
    and saves it to disk. Text is returned and can be attached as 
    the `msg` param for email_notification'"""
    ## Make folders for saving email contents
    import os,sys
    import sklearn.metrics as metrics
    os.makedirs(save_path,exist_ok=True)
    
    ## Get time afor report
    import datetime as dt
    import tzlocal as tz
    now = dt.datetime.now(tz.get_localzone())
                  
    time = now.strftime("%m/%d/%Y - %I:%M %p")  
    
    ## filepaths for fig and report
    fig_fpath = save_path+'confusion_matrix.png'
    msg_text_path = save_path+'msg.txt'

    
    ## GET BEST PARAMS AND MODEL
    best_params = str(grid_search.best_params_)
    best_model = grid_search.best_estimator_#(grid.best_params_)
    
    # Get predictions
    y_hat_test = best_model.predict(X_test)
    
    ## Get Classification report
    report = metrics.classification_report(y_test.argmax(axis=1),y_hat_test)
    
    ## Get text confusion matrix
    cm = np.round(metrics.confusion_matrix(y_test.argmax(axis=1),y_hat_test,normalize='true'),2)
    cm_str = str(cm)

          
    ## Combine text for report
    msg_text = [f'Grid Search Results from {time}:\n']
    msg_text = ['The best params were:\n\t']
    msg_text.append(best_params)
    msg_text.append('\n\n')
    msg_text.append('Classification Report:\n')
    msg_text.append(report)
    msg_text.append('\n\n')

    msg_text.append('Confusion Matrix (normalized to true labels):\n')
    msg_text.append(cm_str)
                  

    
    ## Save the text to file
    with open(msg_text_path,'w+') as f:
        f.writelines(msg_text)
    print(f"Message saved as {msg_text_path}")
                  
    ## Load the (fixed) text from file
    with open(msg_text_path,'r') as f:
        txt = f.read()
        
    ## Plot and save confusion matrix
    fig = plot_confusion_matrix((y_test,y_hat_test))
    try:
        fig.savefig(fig_fpath, dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                    papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
        print(f"Figure saved as {fig_fpath}")           
    except Exception as e:
        print(f"[!] ERROR saving figure:\n\t{e}")
        
    return txt#,fig