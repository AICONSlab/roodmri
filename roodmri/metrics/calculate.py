# Copyright (C) 2022  AICONS Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd


def calculate_metrics(
    df,
    transform_col,
    severity_col,
    metric_cols,
    clean_label,
    grouping_cols = None,
    alpha_oa = 2/3,
    alpha_deg = 2/3
):
    """
    Calculate benchmarking/robustness metrics.

    This function processes a dataframe containing segmentation metrics (e.g.,
    DSC, HD95) for each transform-severity-sample instance in a benchmarking
    dataset. That is, each row corresponds to the segmentation results for a
    model evaluated on an individual sample, with a particular transform
    applied at a particular severity level (e.g., Sample_001 transformed using
    random motion at severity level 3). The dataframe should also include
    segmentation results for the same model evaluated on the clean dataset with
    severity level 0 corresponding to clean data.

    Args:
        df: pandas DataFrame listing segmentation results for each
            transform-severity-sample combination in a benchmarking dataset.
        transform_col: String describing the column label for the transform
            column in df.
        severity_col: String describing the column label for the severity
            column in df.
        metric_cols: Dictionary where the keys are strings describing column
            labels for segmentation metric columns in df (e.g., 'DSC') and the
            values are booleans describing whether higher values for that metric
            are preferred. For example, {'DSC': True, 'HD95', False}.
            Benchmarking metrics will only be calculated for metrics specified
            in the metric_cols dictionary.
        clean_label: String describing the transform column value corresponding
            to clean data (e.g., 'Clean').
        grouping_cols (optional): Sequence of strings corresponding to column
            labels that should be used to segregate data without being
            aggregated over for the purpose of metric calculations. For example,
            results from multiple models and tasks can be included in the same
            DataFrame; this should be specified in the DataFrame by including
            columns describing the model and task. Default is None.
        alpha_oa (optional): Severity-level weighting parameter for overall
            metric calculations (e.g., wmDSC). See Boone et al., 2022 for
            details. Default is 2/3.
        alpha_deg (optional): Severity-level weighting parameter for degradation
            metric calculations (e.g., wmDSC). See Boone et al., 2022 for
            details. Default is 2/3.
    """
    # TODO assertions to check that column labels exist in df, etc.
    if grouping_cols is None:
        grouping_cols = []
    agg = (df.groupby(grouping_cols + [transform_col, severity_col])
           [list(metric_cols.keys())].agg(['mean', 'std'])) # agg over subjects
    clean_rows = (agg.xs(clean_label, axis=0, level=transform_col)
                  .droplevel(severity_col, axis=0))   # get clean rows
    agg.drop(clean_label, axis=0, level=transform_col, inplace=True) # drop rows
    svs = pd.Series(agg.index.get_level_values(level=severity_col),
                    index=agg.index) # get sv lvls as series; same index as agg
    oa_weights, deg_weights = alpha_oa ** svs, alpha_deg ** svs # calc. weights
    weights_sum = [weights.groupby(grouping_cols + [transform_col]).agg('sum')
                   for weights in [oa_weights, deg_weights]] # sum weights
    # calculate overall (weighted) metrics
    oa_metrics = (agg.mul(oa_weights, axis=0).groupby(grouping_cols
                  + [transform_col]).agg('sum') + clean_rows)
    oa_metrics = oa_metrics.div(weights_sum[0] + 1, axis=0)
    oa_metrics = pd.concat({'Overall': oa_metrics}, axis=1) # add column level
    # calculate degradation metrics (also weighted)
    deg_metrics = agg - clean_rows
    for metric_col, higher_is_better in metric_cols.items():
        if higher_is_better:
            new_col = -1 * deg_metrics.loc[:, (metric_col, 'mean')]
            deg_metrics.loc[:, (metric_col, 'mean')] = new_col
    deg_metrics = (deg_metrics.mul(deg_weights, axis=0)
                   .groupby(grouping_cols + [transform_col]).agg('sum')
                   .div(weights_sum[1], axis=0))
    deg_metrics = pd.concat({'Degradation': deg_metrics}, axis=1)
    # combine overall and degradation metrics
    combined = oa_metrics.join(deg_metrics)
    combined_agg = combined.groupby(grouping_cols).agg('mean')
    return combined, combined_agg
