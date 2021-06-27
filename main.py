import argparse

from model import Model

from utils import generate_report


def parse_args():
    """
    Parse the arguments from command-line.

    Returns:
        Parser object.
    """
    description = ("""
        ai-corona: Radiologist-assistant deep learning framework for COVID-19
            diagnosis in chest CT scans
        """)
    parser = argparse.ArgumentParser(description=description)
    help_text = 'Directory of CT scan case. Must only include DICOM files.'
    parser.add_argument('case_directory_path', type=str, help=help_text)
    return parser.parse_args()


def main():
    """
    Main function that runs ai-corona's inference on a CT scan case for
        COVID-19 diagnosis.
    """
    args = parse_args()
    ai_corona = Model()
    diagnosis = ai_corona.predict(case_directory_path=args.case_directory_path)
    generate_report(diagnosis=diagnosis)


if __name__ == '__main__':
    main()
