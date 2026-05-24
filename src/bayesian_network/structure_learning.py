from pgmpy.estimators import HillClimbSearch, BicScore

def learn_structure(df_disc):
    """
    Learn Bayesian Network structure using Hill Climbing + BIC score.
    """
    scorer = BicScore(df_disc)
    hc = HillClimbSearch(df_disc)
    best_model = hc.estimate(scoring_method=scorer)
    return best_model
