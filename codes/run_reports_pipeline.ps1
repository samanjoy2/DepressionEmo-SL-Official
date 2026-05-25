param(
    [string]$ReportDir = "run_reports_advanced",
    [string]$EnsembleSubdir = "ensambled_results"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

$python = "python"
$codeDir = Join-Path $repoRoot "codes"

$reportDirPath = Join-Path $repoRoot $ReportDir
if (-not (Test-Path $reportDirPath)) {
    throw "Report dir not found: $reportDirPath"
}

$labelSummary = Join-Path $reportDirPath "label_scores_summary.csv"
$labelPlotsDir = Join-Path $reportDirPath "label_context_plots"
$combinedLabelPlotsDir = Join-Path $reportDirPath "combined_label_context_plots"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$PyArgs
    )
    Write-Host "==> $Name"
    & $python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

Invoke-Step -Name "Ensemble contexts (C0-C3)" -PyArgs @(
    (Join-Path $codeDir "ensemble_context_logits_advanced.py"),
    "--report_dir", $ReportDir,
    "--ensemble_subdir", $EnsembleSubdir
)
Invoke-Step -Name "Collect test scores" -PyArgs @(
    (Join-Path $codeDir "collect_test_scores_advanced.py"),
    "--report_dir", $ReportDir,
    "--ensemble_subdir", $EnsembleSubdir
)
Invoke-Step -Name "Collect per-label scores" -PyArgs @(
    (Join-Path $codeDir "collect_label_scores_advanced.py"),
    "--report_dir", $ReportDir,
    "--ensemble_subdir", $EnsembleSubdir
)
Invoke-Step -Name "Plot label context effects (F1)" -PyArgs @(
    (Join-Path $codeDir "plot_label_context_effects.py"),
    "--summary_csv", $labelSummary,
    "--out_dir", $labelPlotsDir,
    "--metric", "f1"
)
Invoke-Step -Name "Plot combined label deltas (F1)" -PyArgs @(
    (Join-Path $codeDir "plot_combined_label_context_deltas.py"),
    "--summary_csv", $labelSummary,
    "--out_dir", $combinedLabelPlotsDir,
    "--metric", "f1"
)
Invoke-Step -Name "Plot all training curves (val_macro_f1)" -PyArgs @(
    (Join-Path $codeDir "plot_all_training_curves.py"),
    "--report_dir", $ReportDir,
    "--metric", "val_macro_f1"
)
Invoke-Step -Name "Plot loss curves from metrics" -PyArgs @(
    (Join-Path $codeDir "plot_loss_curves_c0_c3_from_metrics.py"),
    "--report_dir", $ReportDir
)
Invoke-Step -Name "Combine C0 vs C3 loss curve images" -PyArgs @(
    (Join-Path $codeDir "combine_loss_curves_c0_c3.py"),
    "--report_dir", $ReportDir
)
Invoke-Step -Name "Context combination ensembles summary" -PyArgs @(
    (Join-Path $codeDir "ensemble_context_combinations_advanced.py"),
    "--report_dir", $ReportDir
)

Write-Host "Done."
