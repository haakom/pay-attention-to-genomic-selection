import optuna


def prepare_study(ttargs):
    if ttargs.historical_weather == 2:
        h_search_study_name = 'raw_weather_'
        study_name = 'raw_weather_'
    else:
        h_search_study_name = ''
        study_name = ''

    if ttargs.dropout_input == 1:
        h_search_study_name += 'augmented_'
        study_name += 'augmented_'
    elif ttargs.network == 1:
        h_search_study_name += "vanilla_cnn_hyperparameter_search_large"
        study_name += "vanilla_cnn_"
    elif ttargs.network == 2:
        h_search_study_name += "resnet_cnn_hyperparameter_search_large"
        study_name += "resnet_cnn_"
    elif ttargs.network == 3:
        if ttargs.nystrom_attention:
            h_search_study_name += "nystrom_"
            study_name += "nystrom_"
        else:
            h_search_study_name += "performer_"
            study_name += "performer_"
        h_search_study_name += "transformer_hyperparameter_search_large"
        study_name += "transformer_"
    elif ttargs.network == 4:
        if ttargs.nystrom_attention:
            h_search_study_name += "nystrom_"
            study_name += "nystrom_"
        else:
            h_search_study_name += "performer_"
            study_name += "performer_"
        h_search_study_name += "historical_transformer_hyperparameter_search_large"
        study_name += "historical_transformer_"
    elif ttargs.network == 5:
        h_search_study_name += "historical_pureTransformer_hyperparameter_search_large"
        if ttargs.separate_embedding == 1:
            study_name += "separate_embedding_"
        study_name += "historical_pureTransformer_"

    study_name += "crossvalidation"

    if ttargs.one_cycle == 1:
        study_name += "_OneCycle"

    if ttargs.hyper_search:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///hyperparameter_search.db",
            engine_kwargs={"connect_args": {
                "timeout": 600
            }},
        )
        h_search_frozen_trial = None
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    pruner=optuna.pruners.NopPruner(),
                                    load_if_exists=True)
    else:
        h_search_study = optuna.load_study(study_name=h_search_study_name,
                                           storage='sqlite:///' +
                                           ttargs.database_name)
        storage = optuna.storages.RDBStorage(
            url="sqlite:///crossvalidation.db",
            engine_kwargs={"connect_args": {
                "timeout": 600
            }},
        )
        h_search_frozen_trial = h_search_study.best_trial
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    pruner=optuna.pruners.NopPruner(),
                                    load_if_exists=True)

    return study, h_search_frozen_trial
