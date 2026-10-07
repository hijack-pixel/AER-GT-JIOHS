"""Model-based interpretability analyses (Sec. 2.8 / Fig. 7).

This package implements, exactly as described in the manuscript:

1. **Readout weights** - node-wise AER readout weights are extracted from
   held-out windows, averaged over recovery-labelled windows within each
   fold, centered and normalized across ROIs within the fold, then averaged
   across folds; the eight ROIs with the highest cross-fold mean weights
   form the Top-8 set.
2. **ROI masking sensitivity** - each ROI is masked (its node-feature row is
   zeroed) one at a time in the held-out graph samples, the fold-specific
   trained model is reapplied without retraining, and the decrease in
   Recovery-F1 relative to the unmasked model is reported in percentage
   points.
3. **Readout contrasts** - state-specific mean readout contributions
   (baseline / seizure / recovery windows separately), fold-wise
   normalized, then recovery-minus-baseline and recovery-minus-seizure
   contrasts per Top-8 ROI, summarized across folds.
4. **ROI interaction map** - ROI-ROI attention matrices from the global
   attention encoder (averaged across heads, symmetrized) per held-out
   sample; state-wise means over recovery and baseline windows;
   recovery-minus-baseline contrast per ROI pair; readout-weighted
   interaction score (mean recovery-class readout contribution of the two
   endpoint ROIs); top-10% edges ranked by absolute score.

All analyses run on the corresponding held-out fold without additional
training or fold-specific tuning.
"""
