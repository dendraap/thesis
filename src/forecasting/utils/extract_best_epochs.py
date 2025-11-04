from forecasting.utils.libraries_others import Optional, os, re

def extract_best_epoch_from_checkpoint(
    work_dir   : str, 
    model_name : str
) -> Optional[int]:
    """
    This function extract the best number of epochs from saved checkpoint.

    Args:
        work_dir (str)   : Path to the root where the model checkpoint is stored.
        model_name (str) : Subdirectory of model name that contains checkpoints.
    """

    # Merge path with model_name
    ckpt_dir   = os.path.join(work_dir, model_name)

    # Get all files in checkpoints folder
    ckpt_files = os.listdir(ckpt_dir)

    # Iterate through each files
    for f in ckpt_files:

        # Find file that containt 'best-epoch='
        match = re.search(r'best-epoch=(\d+)', f)

        # Get the number of epochs
        if match:
            return int(match.group(1))
        
    return None