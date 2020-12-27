from sklearn.model_selection import KFold


class KFoldImblearn:
    def __init__(self, k_folds=5, k_fold_random_state=None, k_fold_shuffle=False):
        self.k_folds = k_folds
        self.k_fold_random_state = k_fold_random_state
        self.k_fold_shuffle = k_fold_shuffle

        self.k_fold_object = self.__validate_and_instantiate_k_fold_object()

    def __validate_and_instantiate_k_fold_object(self):
        if type(self.k_folds) != int:
            raise TypeError(f"k_folds should be of int type, the passed parameter is of type {type(self.k_folds)}.")
        if self.k_folds <= 1:
            raise ValueError("Value of k_folds should be greater than 1.")
        if type(self.k_fold_random_state) != int and self.k_fold_random_state is not None:
            raise TypeError(
                f"k_fold_random_state should be of int type, parameter of type {type(self.k_fold_random_state)}"
                f" is passed."
            )
        if type(self.k_fold_shuffle) != bool:
            raise TypeError(
                f"k_fold_shuffle should be of bool type, parameter of type {type(self.k_fold_shuffle)} is passed."
            )

        return KFold(n_splits=self.k_folds, random_state=self.k_fold_random_state, shuffle=self.k_fold_shuffle)


if __name__ == "__main__":
    k_fold_imblearn_object = KFoldImblearn(k_folds=3, k_fold_shuffle=True)
    i = 0
