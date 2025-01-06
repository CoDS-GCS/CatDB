from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from ImbalancedLearningRegression import adasyn as reg_adasyn


def augmentation(data, target_attribute, task_type):
    if task_type == "regression":
        print(f"ImbalancedLearningRegression")
        data_resampled = reg_adasyn(data=data, y=target_attribute)
        data_resampled.dropna(subset=[target_attribute], inplace=True)
        return data_resampled
    else:
        gdf = data.groupby([target_attribute]).size().reset_index(name='counts')
        df_col_maps = dict()
        n_samples = min(gdf["counts"])
        if n_samples == 1:
            oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
            print(f"RandomOverSampler")
        else:
            oversample = ADASYN(sampling_strategy='minority', random_state=42)
            print(f"ADASYN")

        for col in data.columns:
            try:
                data[col].astype('int')
            except:
                uvalues = data[col].unique()
                uvalues_map = dict()
                uvalues_map_replace = dict()
                for i, uv in enumerate(uvalues):
                    uvalues_map[uv] = i
                    uvalues_map_replace[i] = uv
                df_col_maps[col] = uvalues_map_replace
                data[col] = data[col].map(uvalues_map)

        X = data.drop(target_attribute, axis=1)
        y = data[target_attribute]
        X_resampled, y_resampled = oversample.fit_resample(X, y)
        X_resampled[target_attribute] = y_resampled

        for col in df_col_maps.keys():
            X_resampled[col] = X_resampled[col].map(df_col_maps[col])
        X_resampled.dropna(subset=[target_attribute], inplace=True)
        return X_resampled
