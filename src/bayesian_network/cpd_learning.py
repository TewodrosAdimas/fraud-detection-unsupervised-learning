from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

def learn_cpds(model_structure, df_disc):
    model = BayesianModel(model_structure.edges())
    model.fit(df_disc, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    return model
