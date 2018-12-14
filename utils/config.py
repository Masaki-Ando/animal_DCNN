import pandas as pd


def dump_config(dst_path, config):
    """
    dump Namaspace(argparse.Argment.parse_args()) to csv file.
    Args:
        dst_path: file path of generated csv file.
        config: argparse.Argment.parse_args()

    Returns: None

    """
    dict_ = vars(config)
    df = pd.DataFrame(list(dict_.items()), columns=['attr', 'status'])
    df.to_csv(dst_path, index=None)
