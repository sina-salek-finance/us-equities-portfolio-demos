import numpy as np


def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print(
        "      Feature{space: <{padding}}      Importance".format(
            padding=max_feature_name_length - 8, space=" "
        )
    )

    for x_train_i in range(len(importances)):
        print(
            "{number:>2}. {feature: <{padding}} ({importance})".format(
                number=x_train_i + 1,
                padding=max_feature_name_length,
                feature=feature_names[indices[x_train_i]],
                importance=importances[indices[x_train_i]],
            )
        )
