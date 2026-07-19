import pandas as pd

folder = "../data/school"

student_info = pd.read_csv(f"{folder}/studentInfo.csv")
student_registration = pd.read_csv(f"{folder}/studentRegistration.csv")
student_assessment = pd.read_csv(f"{folder}/studentAssessment.csv")
assessments = pd.read_csv(f"{folder}/assessments.csv")
student_vle = pd.read_csv(f"{folder}/studentVle.csv")

keys = ["code_module", "code_presentation", "id_student"]

assessment = student_assessment.merge(
    assessments[
        [
            "id_assessment",
            "code_module",
            "code_presentation",
            "assessment_type",
            "weight",
        ]
    ],
    on="id_assessment",
    how="left",
)

assessment_features = (
    assessment.groupby(keys)
    .agg(
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        min_score=("score", "min"),
        assessments_completed=("id_assessment", "count"),
        avg_weight=("weight", "mean"),
        banked_assessments=("is_banked", "sum"),
    )
    .reset_index()
)

vle_features = (
    student_vle.groupby(keys)
    .agg(
        total_clicks=("sum_click", "sum"),
        avg_clicks=("sum_click", "mean"),
        max_clicks=("sum_click", "max"),
        active_days=("date", "nunique"),
        vle_records=("sum_click", "count"),
    )
    .reset_index()
)

registration_features = student_registration.copy()
registration_features["registration_length"] = (
    registration_features["date_unregistration"]
    - registration_features["date_registration"]
)

dataset = (
    student_info.merge(
        registration_features[
            keys
            + [
                "date_registration",
                "date_unregistration",
                "registration_length",
            ]
        ],
        on=keys,
        how="left",
    )
    .merge(assessment_features, on=keys, how="left")
    .merge(vle_features, on=keys, how="left")
)

dataset.drop_duplicates(inplace=True)
dataset.dropna(inplace=True)

dataset.to_csv("school_dataset.csv", index=False)

print(f"Rows: {len(dataset)}")
print(f"Columns: {len(dataset.columns)}")
print(dataset.head())