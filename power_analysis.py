from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()

sample_size = 30 
alpha = 0.05
power = 0.8

effect_size = analysis.solve_power(effect_size=None, nobs1=sample_size, alpha=alpha, power=power)
print(f"Minimum detectable effect size with n={sample_size}: {effect_size:.3f}")
