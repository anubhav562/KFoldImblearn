from sklearn.model_selection import KFold
from k_folds_imblearn.helper import (
    OVERSAMPLING_METHOD_LIST, SAMPLING_METHOD_NAME_TO_CLASS_MAPPING,  UNDER_SAMPLING_METHOD_LIST
)


class KFoldImblearn:
    def __init__(self, sampling_method, sampling_params=None, k_folds=5, k_fold_random_state=None, k_fold_shuffle=False):
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


if __name__ == "__main__":
    k_fold_imblearn_object = KFoldImblearn(sampling_method="SMOTE", k_folds=3, k_fold_shuffle=True)
    print(k_fold_imblearn_object)
