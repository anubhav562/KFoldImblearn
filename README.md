# KFoldImblearn Introduction

    KFoldImblearn handles the resampling of data in a k fold fashion, taking care of
    information leakage so that our results are not overly optimistic. It is built over
    the imblearn package and is compatible with all the oversampling as well as under
    sampling methods provided in the imblearn package.

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
    
-----------------------------------------------

**The wrong approach of performing sampling in KFold Cross Validation:**

Most of the people would perform up-sampling/down-sampling on the whole dataset, and then would apply
K-Fold Cross Validation on the complete dataset. This is a wrong way as this approach is over-optimistic
and lead to information leakage. The validation set should always be kept untouched, or in other words no 
sampling should be applied to the validation set.

The correct approach would be first splitting the data into multiple folds and then applying sampling
just to the training data and let the validation data be as is.

**The image below states the correct approach of how the dataset should be resampled in a K-fold fashion.**

![alt text](https://github.com/anubhav562/KFoldImblearn/blob/main/docs/K_Fold_Imblearn_Banner.png?raw=True)

The correct way of performing Cross validation in a K-fold fashion is described above, and this is exactly what 
KFoldImblearn offers.

------------------------------------------------

## Installation
    
    pip install -i https://test.pypi.org/simple/ Test-KFoldImblearn==1.0.6
    
If you get any third-party module errors while installing the package such as 
**"could not find a version that satisfies the requirement <package_name>==X.X.X"**
then simply pip install the package mentioned by using the command below:

    pip install <package_name>==X.X.X
    
And then again try installing install KFoldImblearn


## Example
    
```python
from k_fold_imblearn import KFoldImblearn
from sklearn.datasets import make_classification
import pandas as pd
from datetime import datetime

# you can use your own X and y here, we have just made dummy data for the sake of example.
X, y = make_classification(n_samples=10000, weights=(0.1, ))

# instantiate KFoldImblearn by simply providing sampling_method and k_folds
k_fold_imblearn_object = KFoldImblearn(
        sampling_method="RandomOverSampler",
        k_folds=10
)

start_time = datetime.today()

# call the k_fold_fit_resample method by passing dataframe of X, y, verbose and n_jobs
k_fold_imblearn_object.k_fold_fit_resample(pd.DataFrame(X), pd.DataFrame(y), verbose=10, n_jobs=8)

end_time = datetime.today()

print(f"Total time taken: {end_time-start_time}")
print(k_fold_imblearn_object)
```

**Output**
```
[Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done   3 out of  10 | elapsed:    6.6s remaining:   15.6s
[Parallel(n_jobs=8)]: Done   5 out of  10 | elapsed:    6.6s remaining:    6.6s
[Parallel(n_jobs=8)]: Done   7 out of  10 | elapsed:    6.7s remaining:    2.8s
[Parallel(n_jobs=8)]: Done  10 out of  10 | elapsed:    6.7s finished
Total time taken: 0:00:07.035128
KFoldImblearn Instance 
Sampling method: RandomOverSampler
Number of folds: 10

Process finished with exit code 
```
