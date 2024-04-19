def test_oversampling_performance():
    for clf in classifiers.models_:
        DataTools.reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        samplers = DataSamplers(sampling_strategy='auto', random_state=seed)

        order = 0
        print("")
        for s in samplers.over_samplers_:
            order += 1
            print("Testing", clf.name_, "with", s.name_)

            pipe_line = make_pipeline(s.sampler_, StandardScaler(), clf.model_)

            cvr = dataset_1.cv_pipeline(pipe_line, num_folds=5, num_threads=num_threads, results_list=results_list,
                                        classifier_str=clf.name_, sampler_str=s.name_, order=order)

            cv_results.append(cvr)

            DataTools.reset_random_states(np_random_state, torch_random_state, cuda_random_state)

    print(results_list)

    presenter = ResultHandler(results_list, cv_results)
    with pd.option_context('display.float_format', '{:0.6f}'.format):
        print(presenter.to_df(columns=['Classifier', 'Sampler', 'Accuracy_Mean',
                                       'Balanced_Accuracy_Mean', 'F1_Mean', 'Precision_Mean', 'Recall_Mean']))
    presenter.record_results(key)