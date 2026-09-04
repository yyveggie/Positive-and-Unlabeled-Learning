# Superseded: SAR runs under fixed-size labeling

These three files were produced before 2026-08-12, when the SAR mechanisms in
`make_pu` drew a fixed number of labeled positives without replacement with
unequal probabilities. That is not the SAR model: it correlates the draws and
compresses the propensity contrast by roughly a third, so the labeled set does
not follow f_l(x) = e(x) f_+(x) / c, which is what SAR-EM and the SCAR test
both assume. The mechanisms now label each positive independently with
probability e(x).

SCAR-mechanism results are unaffected (uniform fixed-size sampling is
exchangeable and satisfies f_l = f_+ exactly), and every SCAR result on disk
still reproduces bit-for-bit. Kept only for provenance; do not cite.
