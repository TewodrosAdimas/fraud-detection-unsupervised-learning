from pgmpy.estimators import HillClimbSearch, BicScore

def learn_structure(df):
    hc = HillClimbSearch(df)

    best_model = hc.estimate(
        scoring_method=BicScore(df),
         max_indegree=5
    )

    return best_model