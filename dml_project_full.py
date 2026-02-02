"""DML implementation for heterogeneous treatment effect estimation.

Implements the double machine learning framework (Chernozhukov et al., 2017)
for estimating average and conditional treatment effects in high-dimensional
confounding settings. The key innovation: use flexible ML (Lasso, RF) to estimate
confounding functions, but apply cross-fitting to avoid overfitting bias that
would invalidate inference.

Components:
- Data generation: synthetic outcomes with known heterogeneous effects
- DoubleML class: cross-fitted nuisance estimation and ATE/CATE computation
- Orthogonalized regression: removes confounding via residual partialing out
- Sensitivity analysis: robustness to regularization parameter choice

Depends on: numpy, pandas, scikit-learn, statsmodels, matplotlib.
Run the script directly to reproduce the full analysis pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt

RANDOM_STATE = 2026


def simulate_data(n=6000, n_cont=17, n_cat=3, cat_levels=(3, 4, 5), seed=RANDOM_STATE):
    """Generate synthetic data with confounded treatment and heterogeneous effects.
    
    Constructs a realistic scenario where assignment and outcomes both depend on 
    X through nonlinear, complex functions. Treatment is confounded, making naive
    regression biased. True effects are known for validation.
    """
    rng = np.random.RandomState(seed)

    # Mixed continuous + categorical features
    X_cont = rng.normal(size=(n, n_cont))
    X_cat = np.column_stack([rng.randint(0, L, size=n) for L in cat_levels])

    cont_cols = [f"c{i}" for i in range(n_cont)]
    cat_cols = [f"cat{i}" for i in range(len(cat_levels))]

    df = pd.DataFrame(X_cont, columns=cont_cols)
    for i, col in enumerate(cat_cols):
        df[col] = X_cat[:, i].astype(int)

    # Confounding: treatment depends on X via tanh + quadratic + categorical terms
    beta_t = rng.normal(scale=0.5, size=n_cont)
    mu_t = (np.tanh(X_cont.dot(beta_t))
            + 0.5 * (X_cont[:, :3] ** 2).sum(axis=1)
            + 0.3 * (X_cat[:, 0] == 1).astype(float))
    T = mu_t + rng.normal(scale=1.0, size=n)

    # Treatment effect is heterogeneous and depends on features
    tau = 0.5 + 0.4 * np.tanh(X_cont[:, 0]) - 0.3 * (X_cont[:, 1] > 0).astype(float)
    tau += 0.2 * (X_cat[:, -1] == (cat_levels[-1] - 1)).astype(float)

    # Outcome baseline probability
    beta_y = rng.normal(scale=0.7, size=n_cont)
    g = (0.8 * np.sin(X_cont.dot(beta_y) / 2)
         + 0.5 * np.exp(-((X_cont[:, :2]) ** 2).sum(axis=1) / 4)
         + 0.2 * (X_cat[:, 1] == 2).astype(float))

    # Logistic model: Y = Bernoulli(sigmoid(g + tau*T))
    linear_part = g + tau * T
    prob = 1 / (1 + np.exp(-linear_part))
    Y = rng.binomial(1, prob)

    df['T'] = T
    df['Y'] = Y
    df['tau_true'] = tau
    df['mu_t_true'] = mu_t
    df['p_true'] = prob

    return df, cont_cols, cat_cols


class DoubleML:
    """Double ML estimator with cross-fitted nuisance functions."""
    
    def __init__(self, n_splits=4, seed=RANDOM_STATE):
        self.n_splits = n_splits
        self.seed = seed

    def fit_nuisances(self, X, Y, T, learner='lasso', alpha=0.01):
        """Fit outcome m(X) and treatment p(X) models via K-fold cross-fitting.
        
        The cross-fitting ensures out-of-sample residuals, which is key to avoiding
        overfitting bias in the DML estimator. Predictions on test folds are stored.
        """
        n = X.shape[0]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        m_hat = np.zeros(n)
        p_hat = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, T_tr = Y[train_idx], T[train_idx]

            # Instantiate and fit learners on training fold
            if learner == 'lasso':
                m = Lasso(alpha=alpha, random_state=self.seed, max_iter=5000)
                p = Lasso(alpha=alpha, random_state=self.seed, max_iter=5000)
            else:
                m = RandomForestRegressor(n_estimators=200, random_state=self.seed)
                p = RandomForestRegressor(n_estimators=200, random_state=self.seed)

            m.fit(X_tr, Y_tr)
            p.fit(X_tr, T_tr)

            # Predict on held-out test fold
            m_hat[test_idx] = m.predict(X_te)
            p_hat[test_idx] = p.predict(X_te)

        self.m_hat = m_hat
        self.p_hat = p_hat
        return m_hat, p_hat

    def estimate_ate(self, Y, T):
        """Estimate ATE via orthogonalized regression on residuals.
        
        Partialing out: Y_res = Y - E[Y|X], T_res = T - E[T|X]. Then the 
        coefficient β in Y_res ~ T_res recovers the ATE. Uses HC1 robust SE.
        """
        Y_res = Y - self.m_hat
        T_res = T - self.p_hat
        X_reg = sm.add_constant(T_res)
        model = sm.OLS(Y_res, X_reg).fit(cov_type='HC1')
        ate = model.params[1]
        se = model.bse[1]
        return ate, se, model

    def estimate_cate(self, X, Y, T, learner=None):
        """Estimate conditional treatment effects via flexible regression learner.
        
        Fit a supervised learner on (X, T_res) to predict Y_res, then compute
        marginal effect as Δ for a unit increase in T. Uses Gradient Boosting.
        """
        Y_res = Y - self.m_hat
        T_res = T - self.p_hat
        data = np.hstack([X, T_res.reshape(-1, 1)])
        
        if learner is None:
            learner = GradientBoostingRegressor(n_estimators=200, random_state=self.seed)
        learner.fit(data, Y_res)

        # Marginal effect: predicted difference when T increases by 1
        T_res_plus = (T_res + 1).reshape(-1, 1)
        data_plus = np.hstack([X, T_res_plus])
        data_base = np.hstack([X, T_res.reshape(-1, 1)])

        y_plus = learner.predict(data_plus)
        y_base = learner.predict(data_base)
        cate_hat = y_plus - y_base
        return cate_hat, learner


def subgroup_indices(df):
    """Define three subgroups for conditional analysis."""
    s = {}
    s['A'] = df['c0'] > df['c0'].median()  # Upper half of main covariate
    s['B'] = df['cat0'] == 1               # Specific categorical level
    s['C'] = s['A'] & s['B']              # Intersection of A and B
    return s


def sensitivity_analysis(df, cont_cols, cat_cols, alphas):
    """Test ATE stability across different Lasso regularization parameters.
    
    Runs the full DML pipeline for each alpha value and collects ATE/SE.
    Large variation suggests sensitivity; small variation indicates robustness.
    """
    X_raw = df[cont_cols + cat_cols]
    numeric_ix = list(range(len(cont_cols)))
    categorical_ix = list(range(len(cont_cols), len(cont_cols) + len(cat_cols)))

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric_ix),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_ix)
    ])

    X = preprocess.fit_transform(X_raw)
    Y = df['Y'].values
    T = df['T'].values

    rows = []
    for a in alphas:
        dml = DoubleML(n_splits=4, seed=RANDOM_STATE)
        dml.fit_nuisances(X, Y, T, learner='lasso', alpha=a)
        ate, se, _ = dml.estimate_ate(Y, T)
        rows.append({'alpha': a, 'ATE': ate, 'SE': se})

    return pd.DataFrame(rows), preprocess


def main():
    print('Simulating data...')
    df, cont_cols, cat_cols = simulate_data(n=4000, n_cont=17, n_cat=3)

    print('Running sensitivity analysis over Lasso alphas...')
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
    res_df, prep = sensitivity_analysis(df, cont_cols, cat_cols, alphas)
    print(res_df.to_string(index=False))

    # Final DML with chosen alpha
    chosen = 0.01
    X = prep.transform(df[cont_cols + cat_cols])
    Y = df['Y'].values
    T = df['T'].values

    dml = DoubleML(n_splits=4, seed=RANDOM_STATE)
    dml.fit_nuisances(X, Y, T, learner='lasso', alpha=chosen)
    ate, se, _ = dml.estimate_ate(Y, T)
    print(f"\nFinal ATE (alpha={chosen}): {ate:.4f} (SE={se:.4f})")

    cate_hat, _ = dml.estimate_cate(X, Y, T)
    print(f"Mean estimated CATE: {cate_hat.mean():.4f} (true mean: {df.tau_true.mean():.4f})")

    subs = subgroup_indices(df)
    print('\nSubgroup results:')
    for k, mask in subs.items():
        if mask.sum() < 10:
            continue
        print(f"{k}: n={mask.sum()} mean CATE={cate_hat[mask].mean():.4f} true mean={df.loc[mask,'tau_true'].mean():.4f}")

    # Compare with naive OLS regression (no X controls)
    print('\n' + '='*70)
    print('COMPARISON: NAIVE OLS vs. DML')
    print('='*70)
    X_ols = sm.add_constant(T)
    ols_fit = sm.OLS(Y, X_ols).fit(cov_type='HC1')
    ate_ols = ols_fit.params[1]
    se_ols = ols_fit.bse[1]
    bias = ate_ols - ate
    bias_pct = (bias / ate_ols * 100) if ate_ols != 0 else 0
    print(f"Naive OLS (Y ~ T):        ATE = {ate_ols:.6f} (SE = {se_ols:.6f})")
    print(f"DML (orthogonalized):     ATE = {ate:.6f} (SE = {se:.6f})")
    print(f"Bias (OLS - DML):              = {bias:.6f}")
    print(f"Bias reduction:                = {bias_pct:.1f}%")
    print('='*70)

    # Plot sensitivity
    plt.figure(figsize=(8, 5))
    plt.errorbar(res_df['alpha'], res_df['ATE'], yerr=res_df['SE'], marker='o', markersize=8, capsize=5)
    plt.xscale('log')
    plt.xlabel('Lasso alpha (log scale)', fontsize=12)
    plt.ylabel('ATE estimate', fontsize=12)
    plt.title('Sensitivity of ATE to Lasso Regularization', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
