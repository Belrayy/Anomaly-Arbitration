import json
import sys
import os
import math
import statistics
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats


INPUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "predictions.json"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

json_name = os.path.splitext(os.path.basename(INPUT_JSON))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "predictions", "records", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    return data


def escape_latex(text):
    text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


records = load_data(INPUT_JSON)
n_total = len(records)

student_ids = []
predictions = []
scores = []

for r in records:
    student_ids.append(r.get("row_id"))
    predictions.append(r.get("prediction"))
    scores.append(float(r.get("anomaly_score")))

scores_arr = np.array(scores, dtype=float)
predictions_arr = np.array(predictions)

n = len(scores_arr)
mean_val = float(np.mean(scores_arr))
median_val = float(np.median(scores_arr))
var_val = float(np.var(scores_arr, ddof=1)) if n > 1 else 0.0
std_val = float(np.std(scores_arr, ddof=1)) if n > 1 else 0.0
min_val = float(np.min(scores_arr))
max_val = float(np.max(scores_arr))
q1 = float(np.percentile(scores_arr, 25))
q2 = float(np.percentile(scores_arr, 50))
q3 = float(np.percentile(scores_arr, 75))
iqr_val = q3 - q1
skew_val = float(scipy_stats.skew(scores_arr)) if n > 2 else 0.0
kurt_val = float(scipy_stats.kurtosis(scores_arr)) if n > 2 else 0.0
range_val = max_val - min_val

confidence = 0.95
if n > 1:
    sem = std_val / math.sqrt(n)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_margin = t_crit * sem
    ci_low = mean_val - ci_margin
    ci_high = mean_val + ci_margin
else:
    sem = 0.0
    ci_margin = 0.0
    ci_low = mean_val
    ci_high = mean_val

unique_preds, pred_counts = np.unique(predictions_arr, return_counts=True)
pred_distribution = dict(zip([str(u) for u in unique_preds], [int(c) for c in pred_counts]))

at_risk_count = int(pred_distribution.get("1", 0)) if "1" in pred_distribution else int(np.sum(predictions_arr == 1))
not_at_risk_count = n - at_risk_count
at_risk_pct = (at_risk_count / n * 100) if n > 0 else 0.0
not_at_risk_pct = (not_at_risk_count / n * 100) if n > 0 else 0.0

lower_fence = q1 - 1.5 * iqr_val
upper_fence = q3 + 1.5 * iqr_val
outliers = [
    (sid, pred, sc)
    for sid, pred, sc in zip(student_ids, predictions, scores)
    if sc < lower_fence or sc > upper_fence
]

top_at_risk = sorted(
    zip(student_ids, predictions, scores), key=lambda x: x[2]
)[:20]


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
ACCENT = "#2E5C8A"
ACCENT2 = "#C0392B"

hist_path = os.path.join(FIG_DIR, "histogram.png")
plt.figure(figsize=(8, 5))
plt.hist(scores_arr, bins=30, color=ACCENT, edgecolor="white", alpha=0.9)
plt.axvline(mean_val, color=ACCENT2, linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.4f}")
plt.axvline(median_val, color="#27AE60", linestyle="-.", linewidth=1.5, label=f"Median = {median_val:.4f}")
plt.title("Distribution of Student Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(hist_path, dpi=200)
plt.close()

pie_path = os.path.join(FIG_DIR, "pie.png")
plt.figure(figsize=(6, 6))
labels = [
    f"Not At-Risk (0)\n{not_at_risk_count} ({not_at_risk_pct:.1f}%)",
    f"At-Risk (1)\n{at_risk_count} ({at_risk_pct:.1f}%)",
]
sizes = [not_at_risk_count, at_risk_count]
colors = [ACCENT, ACCENT2]
plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f"{p:.1f}%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
plt.title("At-Risk Classification Distribution")
plt.tight_layout()
plt.savefig(pie_path, dpi=200)
plt.close()

box_path = os.path.join(FIG_DIR, "boxplot.png")
plt.figure(figsize=(7, 5))
bp = plt.boxplot(scores_arr, vert=False, patch_artist=True, widths=0.5,
                  flierprops={"marker": "o", "markerfacecolor": ACCENT2, "markersize": 5, "alpha": 0.6})
for patch in bp["boxes"]:
    patch.set_facecolor(ACCENT)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color(ACCENT2)
    median.set_linewidth(2)
plt.title("Boxplot of Student Risk Scores")
plt.xlabel("Risk Score")
plt.yticks([])
plt.tight_layout()
plt.savefig(box_path, dpi=200)
plt.close()


def fmt(x):
    return f"{x:.6f}"


def risk_row_color(score):
    if score < lower_fence:
        return "rowhighrisk"
    return "rowmild"


latex_lines = []
latex_lines.append(r"\documentclass[11pt,a4paper]{article}")
latex_lines.append(r"\usepackage[a4paper,margin=2.2cm]{geometry}")
latex_lines.append(r"\usepackage{fontspec}" if False else r"\usepackage[utf8]{inputenc}")
latex_lines.append(r"\usepackage[T1]{fontenc}")
latex_lines.append(r"\usepackage{graphicx}")
latex_lines.append(r"\usepackage{xcolor}")
latex_lines.append(r"\usepackage{booktabs}")
latex_lines.append(r"\usepackage{longtable}")
latex_lines.append(r"\usepackage{colortbl}")
latex_lines.append(r"\usepackage{array}")
latex_lines.append(r"\usepackage{hyperref}")
latex_lines.append(r"\usepackage{fancyhdr}")
latex_lines.append(r"\usepackage{titlesec}")
latex_lines.append(r"\usepackage{amsmath}")
latex_lines.append(r"\usepackage{float}")
latex_lines.append(r"\usepackage{caption}")
latex_lines.append(r"\usepackage{tocloft}")
latex_lines.append("")
latex_lines.append(r"\definecolor{primaryblue}{HTML}{2E5C8A}")
latex_lines.append(r"\definecolor{accentred}{HTML}{C0392B}")
latex_lines.append(r"\definecolor{lightgray}{HTML}{F2F2F2}")
latex_lines.append(r"\definecolor{rowhighrisk}{HTML}{F5B7B1}")
latex_lines.append(r"\definecolor{rowmild}{HTML}{FDEBD0}")
latex_lines.append(r"\definecolor{rowalt}{HTML}{F2F2F2}")
latex_lines.append("")
latex_lines.append(r"\hypersetup{colorlinks=true,linkcolor=primaryblue,urlcolor=primaryblue,citecolor=primaryblue,pdftitle={Anomaly Detection Report},pdfauthor={Anomaly Arbitration}}")
latex_lines.append("")
latex_lines.append(r"\titleformat{\section}{\color{primaryblue}\normalfont\Large\bfseries}{\thesection}{1em}{}")
latex_lines.append(r"\titleformat{\subsection}{\color{primaryblue}\normalfont\large\bfseries}{\thesubsection}{1em}{}")
latex_lines.append("")
latex_lines.append(r"\pagestyle{fancy}")
latex_lines.append(r"\fancyhf{}")
latex_lines.append(r"\fancyhead[L]{\small\textcolor{primaryblue}{Anomaly Detection-School Report}}")
latex_lines.append(r"\fancyhead[R]{\small\thepage}")
latex_lines.append(r"\renewcommand{\headrulewidth}{0.4pt}")
latex_lines.append("")
latex_lines.append(r"\begin{document}")
latex_lines.append("")

latex_lines.append(r"\begin{titlepage}")
latex_lines.append(r"\centering")
latex_lines.append(r"\vspace*{3cm}")
latex_lines.append(r"{\Huge\bfseries\color{primaryblue} OULAD Student\\[0.3em] At-Risk Detection Report by Anomaly Arbitration\par}")
latex_lines.append(r"\vspace{1.5cm}")
latex_lines.append(r"{\large Statistical Analysis of Student Risk Predictions on the Open University Learning Analytics Dataset\par}")
latex_lines.append(r"\vspace{2cm}")
latex_lines.append(r"\rule{0.6\textwidth}{0.4pt}")
latex_lines.append(r"\vspace{1cm}")
latex_lines.append(r"{\normalsize Generated on: " + datetime.now().strftime("%B %d, %Y at %H:%M") + r"\par}")
latex_lines.append(r"{\normalsize Total Students Analyzed: " + f"{n_total}" + r"\par}")
latex_lines.append(r"\vfill")
latex_lines.append(r"{\small OULAD Learning Analytics - Anomaly Arbitration\par}")
latex_lines.append(r"\end{titlepage}")
latex_lines.append("")

latex_lines.append(r"\tableofcontents")
latex_lines.append(r"\newpage")
latex_lines.append("")

latex_lines.append(r"\section{Executive Summary}")
latex_lines.append(
    f"This report presents a comprehensive statistical analysis of {n} student risk predictions "
    f"generated from the Open University Learning Analytics Dataset (OULAD). "
    f"The cohort contains {not_at_risk_count} students classified as not at-risk ({not_at_risk_pct:.2f}\\%) and "
    f"{at_risk_count} students classified as at-risk of withdrawal or failure ({at_risk_pct:.2f}\\%). "
    f"The mean risk score is {fmt(mean_val)} with a standard deviation of {fmt(std_val)}. "
    f"A 95\\% confidence interval for the mean score was computed as "
    f"[{fmt(ci_low)}, {fmt(ci_high)}]. "
    f"A total of {len(outliers)} statistical outliers were detected using the IQR method."
)
latex_lines.append("")

latex_lines.append(r"\section{Dataset Statistics}")
latex_lines.append(
    "The statistics below summarize the distribution of model-generated risk scores across all "
    "students in the OULAD prediction set."
)
latex_lines.append(r"\begin{table}[H]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Descriptive Statistics of Student Risk Scores}")
latex_lines.append(r"\begin{tabular}{lr}")
latex_lines.append(r"\toprule")
latex_lines.append(r"\rowcolor{primaryblue!15}\textbf{Statistic} & \textbf{Value} \\")
latex_lines.append(r"\midrule")
stat_rows = [
    ("Count", f"{n}"),
    ("Mean", fmt(mean_val)),
    ("Median", fmt(median_val)),
    ("Variance", fmt(var_val)),
    ("Standard Deviation", fmt(std_val)),
    ("Minimum", fmt(min_val)),
    ("Maximum", fmt(max_val)),
    ("Range", fmt(range_val)),
    ("Q1 (25th percentile)", fmt(q1)),
    ("Q2 / Median (50th percentile)", fmt(q2)),
    ("Q3 (75th percentile)", fmt(q3)),
    ("Interquartile Range (IQR)", fmt(iqr_val)),
    ("Skewness", fmt(skew_val)),
    ("Kurtosis", fmt(kurt_val)),
]
for i, (label, val) in enumerate(stat_rows):
    rowcolor = r"\rowcolor{rowalt}" if i % 2 == 0 else ""
    latex_lines.append(f"{rowcolor}{escape_latex(label)} & {val} \\\\")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")
latex_lines.append("")

latex_lines.append(r"\subsection{At-Risk Classification Distribution}")
latex_lines.append(r"\begin{table}[H]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Class Distribution}")
latex_lines.append(r"\begin{tabular}{lrr}")
latex_lines.append(r"\toprule")
latex_lines.append(r"\rowcolor{primaryblue!15}\textbf{Class} & \textbf{Count} & \textbf{Percentage} \\")
latex_lines.append(r"\midrule")
latex_lines.append(f"Not At-Risk (0) & {not_at_risk_count} & {not_at_risk_pct:.2f}\\% \\\\")
latex_lines.append(r"\rowcolor{rowalt}" + f"At-Risk (1) & {at_risk_count} & {at_risk_pct:.2f}\\% \\\\")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")
latex_lines.append("")

latex_lines.append(r"\section{Confidence Interval Analysis}")
latex_lines.append(
    f"Using Student's t-distribution with {n-1} degrees of freedom, the standard error of the mean "
    f"is {fmt(sem)}. The 95\\% confidence interval for the population mean risk score is:"
)
latex_lines.append(r"\[")
latex_lines.append(f"CI_{{95\\%}} = [{fmt(ci_low)},\\ {fmt(ci_high)}]")
latex_lines.append(r"\]")
latex_lines.append(
    f"with a margin of error of {fmt(ci_margin)}. This indicates that we are 95\\% confident the true "
    f"mean risk score of the underlying OULAD student population lies within this interval."
)
latex_lines.append("")

latex_lines.append(r"\section{Visualizations}")
latex_lines.append(r"\subsection{Risk Score Distribution}")
latex_lines.append(r"\begin{figure}[H]")
latex_lines.append(r"\centering")
latex_lines.append(r"\includegraphics[width=0.85\textwidth]{figures/histogram.png}")
latex_lines.append(r"\caption{Histogram of student risk scores with mean and median indicators}")
latex_lines.append(r"\end{figure}")
latex_lines.append("")
latex_lines.append(r"\subsection{Class Proportion}")
latex_lines.append(r"\begin{figure}[H]")
latex_lines.append(r"\centering")
latex_lines.append(r"\includegraphics[width=0.6\textwidth]{figures/pie.png}")
latex_lines.append(r"\caption{Proportion of at-risk vs. not at-risk students}")
latex_lines.append(r"\end{figure}")
latex_lines.append("")
latex_lines.append(r"\subsection{Score Spread}")
latex_lines.append(r"\begin{figure}[H]")
latex_lines.append(r"\centering")
latex_lines.append(r"\includegraphics[width=0.85\textwidth]{figures/boxplot.png}")
latex_lines.append(r"\caption{Boxplot showing quartiles and outliers of student risk scores}")
latex_lines.append(r"\end{figure}")
latex_lines.append("")

latex_lines.append(r"\section{Top At-Risk Students}")
latex_lines.append(
    f"The table below lists the top {len(top_at_risk)} students with the lowest risk scores "
    f"(most at-risk), color-coded by severity relative to the IQR fences "
    f"(lower fence = {fmt(lower_fence)}, upper fence = {fmt(upper_fence)})."
)
latex_lines.append(r"\begin{longtable}{rrr}")
latex_lines.append(r"\caption{Top At-Risk Students by Risk Score} \label{tab:topatrisk} \\")
latex_lines.append(r"\toprule")
latex_lines.append(r"\rowcolor{primaryblue!15}\textbf{Student ID} & \textbf{Prediction} & \textbf{Risk Score} \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\endfirsthead")
latex_lines.append(r"\toprule")
latex_lines.append(r"\rowcolor{primaryblue!15}\textbf{Student ID} & \textbf{Prediction} & \textbf{Risk Score} \\")
latex_lines.append(r"\midrule")
latex_lines.append(r"\endhead")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\endfoot")
for sid, pred, sc in top_at_risk:
    color = risk_row_color(sc)
    latex_lines.append(f"\\rowcolor{{{color}}}{sid} & {pred} & {fmt(sc)} \\\\")
latex_lines.append(r"\end{longtable}")


latex_lines.append(r"\end{document}")

tex_filename = f"report_{json_name}_{timestamp}.tex"
tex_path = os.path.join(OUTPUT_DIR, tex_filename)
with open(tex_path, "w") as f:
    f.write("\n".join(latex_lines))

print(f"Report generated at: {tex_path}")
print(f"Figures saved in: {FIG_DIR}")
print(f"Students processed: {n}")
print(f"Mean: {mean_val:.6f}, Median: {median_val:.6f}, StdDev: {std_val:.6f}")
print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
print(f"Outliers detected: {len(outliers)}")
# print(f"To compile: cd {OUTPUT_DIR} && pdflatex report.tex && pdflatex report.tex")