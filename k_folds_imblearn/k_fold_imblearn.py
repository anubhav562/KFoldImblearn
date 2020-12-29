import logging
import pandas as pd
from joblib import delayed, Parallel, wrap_non_picklable_objects

from sklearn.model_selection import KFold
from k_folds_imblearn.helper import (
    OVERSAMPLING_METHOD_LIST, SAMPLING_METHOD_NAME_TO_CLASS_MAPPING,  UNDER_SAMPLING_METHOD_LIST
)


class KFoldImblearn:
    def __init__(
            self, sampling_method: str, sampling_params: dict = None, k_folds: int = 5,
            k_fold_random_state: int = None, k_fold_shuffle: bool = False, logging_level: int = 50
    ):
        if sampling_params is None:
            sampling_params = {}

        self.__sampling_method = sampling_method
        self.__sampling_params = sampling_params

        self.__validate_sampling_method()
        self.__sampling_method_object = self.__prepare_sampler_object()

        self.__k_folds = k_folds
        self.__k_fold_random_state = k_fold_random_state
        self.__k_fold_shuffle = k_fold_shuffle

        self.__k_fold_object = self.__validate_and_instantiate_k_fold_object()

        self.k_fold_dataset_list = []

        logging.basicConfig(level=logging_level)
        self.__logger = logging.getLogger("KFoldImblearn")
        self.__logger.setLevel(level=logging_level)

    @property
    def k_fold_object(self):
        return self.__k_fold_object

    @property
    def sampling_method_object(self):
        return self.__sampling_method_object

    def __validate_sampling_method(self):
        if self.__sampling_method not in OVERSAMPLING_METHOD_LIST + UNDER_SAMPLING_METHOD_LIST:
            raise ValueError(
                f"The value of sampling_method should be one of the following: "
                f"{OVERSAMPLING_METHOD_LIST + UNDER_SAMPLING_METHOD_LIST}"
            )

    def __prepare_sampler_object(self):
        try:
            sampler_class_reference = SAMPLING_METHOD_NAME_TO_CLASS_MAPPING.get(self.__sampling_method)
            sampler_object = sampler_class_reference(**self.__sampling_params)
            return sampler_object
        except Exception as e:
            print(f"Exception occurred: {e}")

    def __validate_and_instantiate_k_fold_object(self):
        if type(self.__k_folds) != int:
            raise TypeError(f"k_folds should be of int type, the passed parameter is of type {type(self.__k_folds)}.")

        if self.__k_folds <= 1:
            raise ValueError("Value of k_folds should be greater than 1.")

        if type(self.__k_fold_random_state) != int and self.__k_fold_random_state is not None:
            raise TypeError(
                f"k_fold_random_state should be of int type, parameter of type {type(self.__k_fold_random_state)}"
                f" is passed."
            )

        if type(self.__k_fold_shuffle) != bool:
            raise TypeError(
                f"k_fold_shuffle should be of bool type, parameter of type {type(self.__k_fold_shuffle)} is passed."
            )

        return KFold(n_splits=self.__k_folds, random_state=self.__k_fold_random_state, shuffle=self.__k_fold_shuffle)

    def __repr__(self):
        string_representation = f"KFoldImblearn Instance \n" \
                                f"Sampling method: {self.__sampling_method}\n" \
                                f"Number of folds: {self.__k_folds}"
        return string_representation

    def k_fold_fit_resample(
            self, X: pd.DataFrame, y: pd.DataFrame, processing: str = "sequential", n_jobs: int = 1, verbose: int = 0
    ):

        k_fold_indices_tuple_list = (
            (training_indices, validation_indices) for training_indices, validation_indices in
            self.__k_fold_object.split(X, y)
        )

        if processing == "sequential":
            self.__logger.info(msg="Starting resampling in a sequential manner!")
            self.__fit_resample_sequentially(X, y, k_fold_indices_tuple_list)

        elif processing == "parallel":
            self.__logger.info(msg="Starting resampling in a parallel manner!")
            self.k_fold_dataset_list = Parallel(
                n_jobs=n_jobs, verbose=verbose
            )(self.__fit_resample_parallel(X, y, kth_index_tuple) for kth_index_tuple in k_fold_indices_tuple_list)

        return self.k_fold_dataset_list

    @delayed
    @wrap_non_picklable_objects
    def __fit_resample_parallel(self, X, y, kth_index_tuple):

        training_indices = kth_index_tuple[0]
        validation_indices = kth_index_tuple[1]

        X_train = X.iloc[training_indices]
        y_train = y.iloc[training_indices]

        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        X_train_resample, y_train_resample = self.__sampling_method_object.fit_resample(X_train, y_train)

        dataset_dict = {
            "resampled_train_set": (X_train_resample, y_train_resample),
            "validation_set": (X_validation, y_validation)
        }

        return dataset_dict

    def __fit_resample_sequentially(self, X, y, k_fold_indices_tuple_list):

        for index, kth_index_tuple in enumerate(k_fold_indices_tuple_list):
            self.__logger.info(msg=f"Starting resampling for the {index+1}th fold!")

            training_indices = kth_index_tuple[0]
            validation_indices = kth_index_tuple[1]

            X_train = X.iloc[training_indices]
            y_train = y.iloc[training_indices]

            X_validation = X.iloc[validation_indices]
            y_validation = y.iloc[validation_indices]

            X_train_resample, y_train_resample = self.__sampling_method_object.fit_resample(X_train, y_train)

            dataset_dict = {
                "resampled_train_set": (X_train_resample, y_train_resample),
                "validation_set": (X_validation, y_validation)
            }

            self.k_fold_dataset_list.append(dataset_dict)
