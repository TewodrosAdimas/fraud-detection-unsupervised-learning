from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

def learn_cpds(df, structure_model):
    """
    Learn CPDs (Conditional Probability Distributions)
    for a Bayesian Network using Bayesian Estimator.

    Parameters:
    - df: preprocessed + discretized dataframe
    - structure_model: output from HillClimbSearch (has .edges())

    Returns:
    - fitted BayesianNetwork model with CPDs
    """

    # Convert structure to BayesianNetwork
    model = BayesianNetwork(structure_model.edges())

    # Fit CPDs
    model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type="BDeu"  # good default for discrete BN
    )

    return model