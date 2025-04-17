# Built-in imports
import os

# My import
from src.dataset_build.tools import get_all_files, construct_subjects_data

def main():
    raw_data_folder = os.path.join(
        "Data",
    )

    complete_data, incomplete_data = get_all_files(raw_data_folder)
    construct_subjects_data(complete_data, "Complete")
    construct_subjects_data(incomplete_data, "Incomplete")


if __name__ == '__main__':
    main()
