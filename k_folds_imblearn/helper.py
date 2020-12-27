from imblearn.over_sampling import (
    ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE
)
from imblearn.under_sampling import (
    CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN,
    InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler,
    TomekLinks
)

OVERSAMPLING_METHOD_LIST = (
    "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "RandomOverSampler", "SMOTE", "SMOTENC", "SVMSMOTE"
)

UNDER_SAMPLING_METHOD_LIST = (
    "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN",
    "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler",
    "TomekLinks"
)

SAMPLING_METHOD_NAME_TO_CLASS_MAPPING = {
    "ADASYN": ADASYN,
    "BorderlineSMOTE": BorderlineSMOTE,
    "KMeansSMOTE": KMeansSMOTE,
    "RandomOverSampler": RandomOverSampler,
    "SMOTE": SMOTE,
    "SMOTENC": SMOTENC,
    "SVMSMOTE": SVMSMOTE,
    "CondensedNearestNeighbour": CondensedNearestNeighbour,
    "EditedNearestNeighbours": EditedNearestNeighbours,
    "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
    "AllKNN": AllKNN,
    "InstanceHardnessThreshold": InstanceHardnessThreshold,
    "NearMiss": NearMiss,
    "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule,
    "OneSidedSelection": OneSidedSelection,
    "RandomUnderSampler": RandomUnderSampler,
    "TomekLinks": TomekLinks
}
