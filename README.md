# _ai_corona_

_ai_corona_ is a deep learning model for accurate diagnosis of COVID-19 in chest CT scans.

Find in-depth explanations in the paper:

[_ai-corona_: Radiologist-Assistant Deep Learning Framework for COVID-19 Diagnosis in Chest CT Scans](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250952) (PloS One 2021).

## Requirements

- Tensorflow
- Keras
- OpenCV
- PyDicom
- SciKit-Learn

## Usage

- Create a virtual environment:
  ```shell
  $ python -m venv .venv
  ```
- Activate the virtual environment:
  ```shell
  $ source .venv/bin/activate
  ```
- Install the required packages:
  ```shell
  $ pip install -r requirements.txt
  ```
- Run diagnosis:
  ```shell
  $ python main.py path_to_DICOMS_directory
  ```

## Citation

Please cite the paper if you use this code in your own work:

```
@article{yousefzadeh2021ai,
  title={ai-corona: Radiologist-assistant deep learning framework for COVID-19 diagnosis in chest CT scans},
  author={Yousefzadeh, Mehdi and Esfahanian, Parsa and Movahed, Seyed Mohammad Sadegh and Gorgin, Saeid and Rahmati, Dara and Abedini, Atefeh and Nadji, Seyed Alireza and Haseli, Sara and Bakhshayesh Karam, Mehrdad and Kiani, Arda and others},
  journal={PloS one},
  volume={16},
  number={5},
  pages={e0250952},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
