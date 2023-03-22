import argparse


def get_args_windows(args=None):
    parser = argparse.ArgumentParser(
        description="Simple training script for training a RetinaNet network."
    )

    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        help="number of workers for data loading - default 0",
        default=0,
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, help="batch size - default 2", default=2
    )
    parser.add_argument(
        "-g", "--num_gpus", type=int, help="number of used gpus - default 0", default=0
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        help="number of max training epochs - default 5",
        default=40,
    )
    parser.add_argument(
        "--csv_path",
        help="Path to file containing annotations (see readme)",
        type=str,
        # default= r'/home/ludger/8_RA_Finger/data/KI_csv.xlsx')
        default=r"F:\ra_data\KI_csv.xlsx",
    )
    parser.add_argument(
        "--data_path",
        help="Path to folder containing dcm_images (see readme)",
        type=str,
        # default=r'/home/ludger/8_RA_Finger/data')
        default=r"F:\ra_data",
    )
    parser.add_argument(
        "-pos",
        "--positive",
        help="if true -> test mode - default FALSE",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-neg",
        "--negative",
        help="if true -> test mode - default FALSE",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "-ae",
        "--adaptive_epochs",
        help="if true -> test mode - default FALSE",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-c",
        "--copy_log_path",
        help="",
        type=str,
        default="",
    )

    return parser.parse_args(args)


def get_args_server(args=None):
    parser = argparse.ArgumentParser(
        description="Simple training script for training a RetinaNet network."
    )

    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        help="number of workers for data loading - default 0",
        default=80,
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, help="batch size - default 2", default=2
    )
    parser.add_argument(
        "-g", "--num_gpus", type=int, help="number of used gpus - default 0", default=0
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        help="number of max training epochs - default 5",
        default=50,
    )
    parser.add_argument(
        "--csv_path",
        help="Path to file containing annotations (see readme)",
        type=str,
        default=r"/home/data/KI_csv.xlsx",
    )
    parser.add_argument(
        "--data_path",
        help="Path to folder containing dcm_images (see readme)",
        type=str,
        default=r"/home/data",
    )
    parser.add_argument(
        "-pos",
        "--positive",
        help="if true -> test mode - default FALSE",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-neg",
        "--negative",
        help="if true -> test mode - default FALSE",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "-ae",
        "--adaptive_epochs",
        help="if true -> test mode - default FALSE",
        type=int,
        default=40,
    )
    parser.add_argument(
        "-c",
        "--copy_log_path",
        help="",
        type=str,
        default="",
    )

    return parser.parse_args(args)
