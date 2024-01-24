- Trying to reproduce the [figure](https://rcdata.nau.edu/genomic-ml/necromass/data-danny-same-other-cv-figure.png) 
results of when used on necromass data set with Featureless and LassoCV in python.
- Code was written in R.
- The data set needed: 
1. Data old, data new (22nd Dec).
2. All of the necromass(fungi + bacteria data sets), (Habitat = soil), (Melanization = High), (Melanization = Low)
3. Log transformed and Yeo-Johnson transformed data sets.

- Input : Data old, data new (22nd Dec), All of the necromass(fungi + bacteria data sets), Habitat = soil, Melanization = High, Melanization = Low, Log transformed and Yeo-Johnson transformed data sets.

- Output : Figure results of when used on necromass data set with Featureless and LassoCV in python.

```python
"Mean Squared Error": mse,
"FoldID": fold_id,
"# of Total Samples": n_sub_samples,
"Dataset": data_set_name,
"Index of Predicted Column": index_of_pred_col,
"Algorithm": learner_name,
```