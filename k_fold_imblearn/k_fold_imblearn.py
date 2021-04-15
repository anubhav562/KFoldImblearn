import logging
import pandas as pd
import pickle

from joblib import delayed, Parallel, wrap_non_picklable_objects
from sklearn.model_selection import KFold

from k_fold_imblearn.helper import SAMPLING_METHOD_NAME_TO_CLASS_MAPPING


class KFoldImblearn:
    """
    KFoldImblearn

    KFoldImblearn handles the resampling of data in a k fold fashion, taking care of
    information leakage so that our results are not overly optimistic. It is built over
    the imblearn package and is compatible with all the oversampling as well as under
    sampling methods provided in the imblearn package

    While performing over-sampling, under-sampling and balanced-sampling we need to make
    sure that we are not touching/manipulating our validation or test set. Making changes
    to our validation set can lead us to have results that are overly optimistic.
    This over optimism of the results is called information leakage caused by the sampling
    techniques applied to the test set as well.

    In a typical holdout method (holdout simply means splitting data into test and train),
    over-optimism can be handled by simply resampling the training data, training the models
    on this resampled training data and finally testing it on the untouched test data.

    But if we want to apply sampling techniques over k folds
    (because we want to test our model over the k folds and want to have a
    general idea of how it is performing), then we would have to frame the logic
    and write the code ourselves. KFoldImblearn does the exact same process for us.

    Parameters
    ----------
    sampling_method : string
        The sampling method which is the user wants to apply to the data in a k-fold
        fashion. Can take the following values:

        "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "RandomOverSampler", "SMOTE",
        "SMOTENC", "SVMSMOTE", "CondensedNearestNeighbour", "EditedNearestNeighbours",
        "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss",
        "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks"

        The above sampling methods contain both over and under sampling techniques contained
        in the imblearn package.

    sampling_params : dict, default=None
        A parameter dictionary containing the arguments that will be fed to the sampling_method
        mentioned above. For eg. if we decide to choose "SMOTE", then sampling_params will be a
         dict of arguments that one will use while instantiating the SMOTE class

    k_folds : int, default=5
        Number of folds. Must be at least 2.

    k_fold_shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    k_fold_random_state : int,  default=None
        When `k_fold_shuffle` is True, `k_fold_random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.

    logging_level : int, default=50
        logging level for the custom logger setup for this class.
        values that can be assigned: 0, 10, 20, 30, 40 and 50

    """
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
        self.__logger = logging.getLogger(f"KFoldImblearn_{sampling_method}")
        self.__logger.setLevel(level=logging_level)

    @property
    def k_fold_object(self):
        return self.__k_fold_object

    @property
    def sampling_method_object(self):
        return self.__sampling_method_object

    def __validate_sampling_method(self):
        over_sampling_methods = (
            "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "RandomOverSampler", "SMOTE", "SMOTENC", "SVMSMOTE"
        )

        under_sampling_methods = (
            "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN",
            "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection",
            "RandomUnderSampler", "TomekLinks"
        )

        if self.__sampling_method not in over_sampling_methods + under_sampling_methods:
            raise ValueError(
                f"The value of sampling_method should be one of the following: "
                f"{over_sampling_methods + under_sampling_methods}"
            )

    def __prepare_sampler_object(self):
        try:
            sampler_class_reference = SAMPLING_METHOD_NAME_TO_CLASS_MAPPING.get(self.__sampling_method)
            sampler_object = sampler_class_reference(**self.__sampling_params)
            return sampler_object
        except Exception as e:
            self.__logger.critical(msg=f"Following exception occurred: {e}")

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

    def k_fold_fit_resample(self, X: pd.DataFrame, y: pd.DataFrame, n_jobs: int = 1, verbose: int = 0) -> list:
        """
        This method perform fit-resampling in a k fold fashion and returns the list containing 'k' datasets.

        :param X: X dataframe containing all the attributes/features.
        :param y: y dataframe containing all the labels.
        :param n_jobs: no. of CPU cores the user wants to employ in the resampling process. set n_jobs = -1 to use
                       all the CPU cores.
        :param verbose: an integer value which enables logging of completion of tasks.
        :return: list of dictionary containing 'k' datasets
        """
        try:
            # validation of the arguments passed in by the user
            self.__type_check_fit_resample_arguments(X, y, n_jobs, verbose)

            # k tuples in the form of (training_indices, validation_indices)
            k_fold_indices_tuple_list = [
                (training_indices, validation_indices) for training_indices, validation_indices in
                self.__k_fold_object.split(X, y)
            ]

            # joblib is used for spawning of multiple python processes
            # if n_jobs = 1 SequentialBackend is used while LokyBackend is used for n_jobs=-1 or n_jobs > 1
            self.k_fold_dataset_list = Parallel(
                n_jobs=n_jobs, verbose=verbose
            )(self.__fit_resample_parallel(X, y, kth_index_tuple) for kth_index_tuple in k_fold_indices_tuple_list)

            return self.k_fold_dataset_list

        except OSError as oe:
            self.__logger.critical(msg="----------- Terminated ------------")
            self.__logger.critical(msg=f"OSError occurred: {oe}")
            self.__logger.critical(
                msg="If your system faces memory shortage or resource shortage, try reducing the n_jobs argument."
            )

        except MemoryError as me:
            self.__logger.critical(msg="----------- Terminated ------------")
            self.__logger.critical(msg=f"MemoryError occurred: {me}")
            self.__logger.critical(
                msg="If your system faces memory shortage or resource shortage, try reducing the n_jobs argument."
            )

        except Exception as e:
            self.__logger.critical(msg="----------- Terminated ------------")
            self.__logger.critical(msg=f"The following exception occurred: {e}")

    @staticmethod
    def __type_check_fit_resample_arguments(X, y, n_jobs, verbose):
        """
        This method is used to for the validation of the arguments passed in by the user.
        :param X: should be a dataframe otherwise a TypeError is raised.
        :param y: should be a dataframe otherwise a TypeError is raised.
        :param n_jobs: should be an integer otherwise a TypeError is raised.
        :param verbose: should be an integer otherwise a TypeError is raised.
        :return: None
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should of type: {pd.DataFrame}. Argument X passed is of the type: {type(X)}")

        if type(y) != pd.DataFrame:
            raise TypeError(f"y should of type: {pd.DataFrame}. Argument y passed is of the type: {type(y)}")

        if type(n_jobs) != int:
            raise TypeError(f"n_jobs should of type: {int}. Argument n_jobs passed is of the type: {type(n_jobs)}")

        if type(verbose) != int:
            raise TypeError(f"verbose should of type: {int}. Argument verbose passed is of the type: {type(verbose)}")

    @delayed
    @wrap_non_picklable_objects
    def __fit_resample_parallel(self, X, y, kth_index_tuple):
        """
        This method has been decorated with the delayed and wrap_non_picklable_objects decorators
        because this is used with the joblib's Parallel class.
        This method simply re-samples the training data (and not the validation data).
        It internally calls the fit_resample method of the sampling strategy that has been
        chosen by the user.
        :param X: The X dataframe provided by the user.
        :param y: The y dataframe provided by the user.
        :param kth_index_tuple: tuple in the form of (training_indices, validation_indices).
        :return: a dictionary containing the resampled_training_set and validation_set.
        """

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

    def serialise_k_datasets_list(self, filepath, **kwargs):
        """
        This method is used for serialising the k_fold_dataset_list that has been resampled
        using this library.

        :param filepath: The file path where you want to store the dataset list.
        :param kwargs: The keyword arguments which are passed to pickle dump method (these are optional).
        :return: None
        """
        try:
            with open(filepath, mode='wb') as f:
                pickle.dump(obj=self.k_fold_dataset_list, file=f, **kwargs)

        except Exception as e:
            self.__logger.critical(msg=f"The following exception occurred: {e}")
