import os.path
import warnings

import sklearn
from classicdata import (
    USPS,
    ImageSegmentation,
    Ionosphere,
    LetterRecognition,
    MagicGammaTelescope,
    PenDigits,
    RobotNavigation,
)
from classicdata.dataset import GenericDataset
from classicexperiments import Estimator, Evaluation, Experiment
from loguru import logger
from pylmnn import LargeMarginNearestNeighbor
from sklearn.neighbors import KNeighborsClassifier
from sklearn_lvq import LgmlvqModel

from lann import LocallyAdaptiveNeighborsClassifier


def showwarning(message, *args, **kwargs):
    logger.warning(message)


warnings.showwarning = showwarning


def LargeMarginNearestNeighborClassifier(
    n_neighbors: int, random_state=None
) -> sklearn.pipeline.Pipeline:
    return sklearn.pipeline.make_pipeline(
        LargeMarginNearestNeighbor(n_neighbors=n_neighbors, random_state=random_state),
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors),
    )


datasets = [
    Ionosphere(),
    LetterRecognition(),
    MagicGammaTelescope(),
    PenDigits(),
    RobotNavigation(),
    ImageSegmentation(),
    USPS(),
]

OUTDOOR_PATH = "data/outdoor.txt"
if os.path.exists(OUTDOOR_PATH):
    datasets.append(
        GenericDataset(
            safe_name="outdoor",
            short_name="Outdoor Objects",
            long_name="Outdoor Objects",
            n_samples=None,
            n_features=None,
            n_classes=None,
            source=None,
            path=OUTDOOR_PATH,
        )
    )
ADRENAL_PATH = "data/adrenal.txt"
if os.path.exists(OUTDOOR_PATH):
    datasets.append(
        GenericDataset(
            safe_name="adrenal",
            short_name="Adrenal",
            long_name="Adrenal",
            n_samples=None,
            n_features=None,
            n_classes=None,
            source=None,
            path=ADRENAL_PATH,
        )
    )

N_NEIGHBORS = 5

estimators = [
    Estimator(
        "LANN",
        LocallyAdaptiveNeighborsClassifier,
        {"n_neighbors": N_NEIGHBORS, "n_epochs": 100, "random_state": 0},
    ),
    Estimator("knn", KNeighborsClassifier, {"n_neighbors": N_NEIGHBORS}),
    Estimator(
        "LMNN", LargeMarginNearestNeighborClassifier, {"n_neighbors": N_NEIGHBORS}
    ),
    Estimator("LGMLVQ", LgmlvqModel, {}),
]

experiments = [
    Experiment(
        dataset=dataset,
        estimator=estimator,
        estimation_function=sklearn.model_selection.cross_val_score,
        parameters={"n_jobs": -1},
        scaler=sklearn.preprocessing.StandardScaler(),
    )
    for estimator in estimators
    for dataset in datasets
]

evaluation = Evaluation(experiments=experiments, base_dir="evaluation")

evaluation.run()
evaluation.present()
