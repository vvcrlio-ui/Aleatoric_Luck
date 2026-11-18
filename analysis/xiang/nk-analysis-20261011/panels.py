"""Model colours and panel order shared by the analysis figures."""

MODELS = [("ridge", "Ridge", "#2f5596"), ("lasso", "Lasso", "#6f92c8"), ("random_forest", "Random forest", "#4f7d4e"),
          ("extra_trees", "Extra trees", "#9cbf92"), ("xgboost", "XGBoost", "#b9767d"), ("lightgbm", "LightGBM", "#7d3c45"),
          ("shallow_neural_network", "Shallow neural network", "#c9a227"), ("super_learner", "Super learner", "#222222")]
OUTCOMES = [("materialHardship", "Material hardship"), ("gpa", "GPA"), ("grit", "Grit"), ("eviction", "Eviction"),
            ("jobTraining", "Job training"), ("layoff", "Layoff")]
ROWS = [[(f"ffc_tree_ordinal_{o}", f"FFC {label} · tree-ordinal"), (f"ffc_median_mode_{o}", f"FFC {label} · median/mode")]
        for o, label in OUTCOMES] + [[("smr_hourlywage", "SMR log hourly wage"), ("smr_totalincome", "SMR log total income")]]
TITLE, MUTED = "#203b4c", "#5c6971"
