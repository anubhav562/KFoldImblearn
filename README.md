# KFoldImblearn

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

