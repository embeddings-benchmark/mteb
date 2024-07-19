def _fit_predict(model_i, task, task_df, classifer):
    clf = classifer()
    X_train = task_df.drop([task], axis=1).drop(model_i)
    y_train = task_df[[task]].drop(model_i)
    clf.fit(X_train.values, y_train.values)
    X_test = task_df.drop(columns=[task]).iloc[model_i]
    y_pred = clf.predict(X_test.values.reshape(1, -1))
    return float(y_pred)
