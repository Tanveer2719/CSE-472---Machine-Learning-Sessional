
import numpy as np
import pandas as pd
import logisticRegression as lgr
import importlib

importlib.reload(lgr)


# Resampling
def custom_resample(x, y, proportion=1.0, seed = 42):
    """
    Custom function to perform bootstrapping
    """
    np.random.seed(seed)  # Ensure reproducibility
    # sample with replacement

    indices = np.random.choice(np.arange(len(x)), size=int(proportion*len(x)), replace=True)

    # iloc - index the dataframe and reset the index
    x_sampled = x.iloc[indices].reset_index(drop=True)
    y_sampled = y[indices]

    return x_sampled, y_sampled

# create and return bagging models
def create_bagging_models(x_train, y_train, num_models=9, class_weight=None):
    """
        x_train: Pandas Dataframe
        y_train: Pandas DataFrame 
    """
    models = []
    for i in range (num_models):
        x_resampled, y_resampled = custom_resample(x_train, y_train, seed=44+i)
        
        if class_weight:
            model = lgr.CustomLogisticRegression(class_weight=class_weight)
        else:
            model = lgr.CustomLogisticRegression()
            
        model.custom_fit(x_resampled, y_resampled)
        # model = LogisticRegression()
        # model.fit(x_resampled, y_resampled)
        models.append(model)

    return models

# return the predictions from the bagging models
def get_predictions_from_bagged_models(models, x):
    """
        get predictions from a list of models against the input data X
        
        X: pandas dataframe
    """
    return np.array([model.custom_predict(x) for model in models]).T

# augment the predictions to the validation set
def augment_predictions_to_validation_set(models, x_validation):
    """
        Augment the results got against the validation set from the models 

        Returns: 
            Augmented Pandas DataFrame 
    """
    model_predictions = get_predictions_from_bagged_models(models, x_validation)

    # create columns
    cols = [f'model_{i+1}' for i in range(model_predictions.shape[1])]

    x_validation_df = pd.DataFrame(x_validation)
    model_predictions_df = pd.DataFrame(model_predictions, columns=cols)


    augmented_validation_df = pd.concat([x_validation_df, model_predictions_df], axis=1)

    return augmented_validation_df

# train the meta classifier
def train_meta_classifier(augmented_validation_df, y_validation):
    """
        return the meta classifier that is trained using the augmented validation set 
    """
    meta_model = lgr.CustomLogisticRegression()
    meta_model.custom_fit(augmented_validation_df, y_validation)
    return meta_model

    