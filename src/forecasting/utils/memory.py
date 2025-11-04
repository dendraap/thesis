from forecasting.utils.libraries_others import gc

def cleanup(*args) -> None:
    """
    Function to delete variable from memory and call garbage collector.
    
    Args:
        args : list of variable names (as objects) that want to delete.
    """
    
    for var in args:
        try:
            del var
        except:
            pass
    gc.collect()