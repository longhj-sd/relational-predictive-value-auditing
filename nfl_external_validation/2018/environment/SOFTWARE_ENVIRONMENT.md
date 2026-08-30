# Software Environment

The 2018 replication was validated under Python 3.12 on Windows with the packages listed in `requirements-2018.txt`. The public package is designed to run from a clean repository checkout using only command-line supplied data paths.

Core package families:

- numpy
- pandas
- scipy
- scikit-learn
- pyarrow
- python-docx
- matplotlib
- PyYAML
- tabulate

The release verifier checks the presence of this document and the pinned protocol files. Reviewers may record their exact local versions with `pip freeze` after installing `requirements-2018.txt`.
