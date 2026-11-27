function root = aergt_cohort_root(cfg, cohort)
%AERGT_COHORT_ROOT Raw-data root folder of one cohort.
%
% cohort is 'labeled' (ictal-transition recordings), 'pretrain' (normative
% baseline, NB1 and NB2 saline recordings), or 'control' (pharmacological
% controls).

switch lower(cohort)
    case 'labeled'
        root = cfg.raw.epilepsy_root;
    case 'pretrain'
        root = cfg.raw.pretrain_root;
    case 'control'
        root = cfg.raw.control_root;
    otherwise
        error('aergt_cohort_root:UnknownCohort', 'Unknown cohort: %s', cohort);
end
end
