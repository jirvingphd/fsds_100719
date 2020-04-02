"""A collection of functions not yet-ready for the jmi modules"""
def flat_list(L, result=None, print_results=True):
    """
    Function from Recursive Functions Section of Learn.co v2
    
    Args:
        L (list or scalar): The item/list to be tested and unpacked.
        result (list, optional): The list to add the contents of L to. Defaults to an empty list.
        print_results (bool, optional): Controls displaying of output. Defaults to True.
    
    Returns:
        result : flattened list L 
    """
    
    if result is None:
        result = []
    if print_results:
        print('Current L:', L) #Optional, to display process
    for i in L:
        if type(i) == list:
            flat_list(i, result)
        else:
            result.append(i)
    return result

see = 'example of what i was saying'
def example():
    print(see)
    
def flat_dict(D, result=None, print_results=True):
    """
    Function from Recursive Functions Section of Learn.co v2
    
    Args:
        D (dict or scalar): The item/list to be tested and unpacked.
        result (dict, optional): The list to add the contents of L to. Defaults to an empty list.
        print_results (bool, optional): Controls displaying of output. Defaults to True.
    
    Returns:
        result : flattened list L 
    """
    
    if result is None:
        result = []
    if print_results:
        print('Current D:', D) #Optional, to display process
    for i in D:
        if type(i) == list:
            flat_dict(i, result)
        else:
            result.append(i)
    return result

