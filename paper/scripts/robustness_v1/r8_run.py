"""R8 (prereg sec. 8): primary model without supercolonial species and social parasites (Bell-Roberts 2024 Table 21)."""
import json, sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *
import stages.brain_analysis as B
from analysis import robustness_extra as RX

df, tree = primary_data(), mcc()
t0, _ = B.fit_primary(df, tree, SESOI)
assert abs(t0["estimate"] - 0.0351) < 5e-4 and t0["n"] == 270, t0
flags = RX.bell_roberts_flags(BR_XLSX)
d, dropped = RX.r8_subset(df, flags)
t, res = B.fit_primary(d, tree, SESOI)
pt, pool = B.primary_over_trees(d, posterior(), SESOI)
out = {"table21_rows": len(flags), "flagged_supercolonial": int(flags.supercolonial.sum()),
       "flagged_social_parasite": int(flags.social_parasite.sum()), "dropped_n": len(dropped),
       "dropped": dropped.to_dict("records"),
       "mcc": {k: t[k] for k in ("estimate", "se", "ci_low", "ci_high", "p", "n", "df", "lambda", "outcome", "ci90")},
       "posterior": pool, "robust_trees": pool["outcome"] == t["outcome"]}
(WORK / "r8.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("dropped_n", "mcc")}, default=float, indent=1))
print({k: pool[k] for k in ("estimate_median", "pooled_ci_low", "pooled_ci_high", "share_p_lt_0.05", "outcome")})
