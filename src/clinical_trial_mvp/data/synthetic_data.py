"""Generate reproducible synthetic clinical trial records."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

from clinical_trial_mvp.data.schema import TABLE_ORDER, TABLE_SCHEMAS


DEFAULT_SEED = 20260911
DATA_CUTOFF_DATE = date(2026, 9, 1)

PHASES = ["Phase I", "Phase II", "Phase III", "Phase IV"]
INDICATIONS = [
    "Breast Cancer",
    "Type 2 Diabetes",
    "Hypertension",
    "Rheumatoid Arthritis",
    "Asthma",
    "Migraine",
]
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
TREATMENT_ARMS = [
    "Placebo",
    "Standard of Care",
    "Investigational Drug Low Dose",
    "Investigational Drug High Dose",
]
SCREENING_STATUSES = ["Enrolled", "Randomized", "Screen Failed", "Withdrawn"]
VISIT_SCHEDULE = [
    ("Screening", -7),
    ("Baseline", 0),
    ("Week 4", 28),
    ("Week 8", 56),
    ("Week 12", 84),
    ("Follow-up", 112),
]
ADVERSE_EVENT_TERMS = [
    "Headache",
    "Nausea",
    "Fatigue",
    "Injection Site Reaction",
    "Dizziness",
    "Rash",
    "Upper Respiratory Tract Infection",
    "Insomnia",
]
SEVERITIES = ["Mild", "Moderate", "Severe"]
OUTCOMES = ["Recovered", "Recovering", "Ongoing", "Unknown"]


@dataclass(frozen=True)
class LabProfile:
    name: str
    unit: str
    reference_low: float
    reference_high: float
    decimals: int


LAB_PROFILES = {
    "ALT": LabProfile("Alanine aminotransferase", "U/L", 7.0, 56.0, 1),
    "AST": LabProfile("Aspartate aminotransferase", "U/L", 10.0, 40.0, 1),
    "HGB": LabProfile("Hemoglobin", "g/dL", 12.0, 17.5, 1),
    "WBC": LabProfile("White blood cell count", "10^9/L", 4.0, 11.0, 1),
    "CREAT": LabProfile("Creatinine", "mg/dL", 0.6, 1.3, 2),
}


SCREEN_FAILURE_CRITERIA = [
    ("Inclusion", "INC-01", "Age was outside the protocol-defined range."),
    ("Inclusion", "INC-02", "Disease duration did not meet the minimum requirement."),
    ("Inclusion", "INC-03", "Disease criteria were not met at screening."),
    ("Exclusion", "EXC-01", "A prohibited concomitant medication was identified."),
    ("Exclusion", "EXC-02", "A disallowed comorbidity was present."),
    ("Exclusion", "EXC-03", "Recent participation in another investigational study was identified."),
    ("Lab", "LAB-01", "Renal function was outside protocol limits."),
    ("Lab", "LAB-02", "Hepatic laboratory values exceeded protocol limits."),
    ("Lab", "LAB-03", "Hematology results were outside protocol limits."),
    ("Consent", "CON-01", "Informed consent was not completed before screening procedures."),
    ("Consent", "CON-02", "Consent was withdrawn before enrollment."),
    ("Withdrawal", "WDR-01", "The candidate withdrew before eligibility confirmation."),
]

DEVIATION_DESCRIPTIONS = {
    "Visit Window": [
        "A protocol visit occurred outside the permitted visit window.",
        "The scheduled assessment was completed after the protocol window closed.",
    ],
    "Consent": [
        "A consent re-signature was completed after a revised form became effective.",
        "A study procedure occurred before the current consent version was documented.",
    ],
    "Dosing": [
        "A scheduled dose was administered outside the protocol timing allowance.",
        "The dose record did not match the protocol-defined administration schedule.",
    ],
    "Eligibility": [
        "Eligibility documentation was incomplete when enrollment was confirmed.",
        "A protocol eligibility criterion was not verified before randomization.",
    ],
    "Procedure": [
        "A required protocol procedure was omitted at the scheduled visit.",
        "A protocol assessment used a nonstandard collection sequence.",
    ],
}

CORRECTIVE_ACTIONS = {
    "Visit Window": [
        "Reschedule the assessment and reinforce visit-window tracking.",
        "Update visit reminders and review the scheduling workflow.",
    ],
    "Consent": [
        "Retrain site staff on consent version control.",
        "Review the consent checklist with the study team.",
    ],
    "Dosing": [
        "Retrain dosing staff and reconcile the administration record.",
        "Add a second-person check to the dosing workflow.",
    ],
    "Eligibility": [
        "Retrain investigators on eligibility verification.",
        "Add a documented eligibility review before randomization.",
    ],
    "Procedure": [
        "Retrain staff on the procedure checklist.",
        "Update the site process and document the missed assessment.",
    ],
}

QUERY_FIELDS = {
    "Missing": ["visit_date", "lab_value", "ae_outcome", "procedure_status"],
    "Inconsistent": ["visit_date", "treatment_arm", "ae_outcome", "dosing_record"],
    "Outlier": ["lab_value", "vital_sign_value"],
    "Clarification": ["visit_date", "lab_value", "ae_outcome", "procedure_status"],
}

QUERY_TEXT = {
    "visit_date": "Please confirm the visit date; it falls outside the protocol window.",
    "lab_value": "Please verify the laboratory value and unit against the source record.",
    "ae_outcome": "Please clarify the adverse event outcome and update the record if resolved.",
    "procedure_status": "Please confirm whether the required procedure was completed.",
    "treatment_arm": "Please reconcile the recorded treatment arm with the randomization record.",
    "dosing_record": "Please clarify the dosing record and confirm the administration time.",
    "vital_sign_value": "Please verify the outlying vital-sign value against the source record.",
}


Row = dict[str, Any]


def iso(value: date | None) -> str:
    return "" if value is None else value.isoformat()


def fake_date_between(fake: Faker, start_date: date, end_date: date) -> date:
    if end_date < start_date:
        return start_date
    return fake.date_between_dates(date_start=start_date, date_end=end_date)


def write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_trials_and_sites(fake: Faker, rng: random.Random) -> tuple[list[Row], list[Row]]:
    trials: list[Row] = []
    sites: list[Row] = []
    site_counter = 1

    for trial_number in range(1, 5):
        trial_id = f"TRIAL-{trial_number:03d}"
        start_date = fake_date_between(fake, date(2024, 1, 1), date(2025, 9, 1))
        trial_sites: list[Row] = []

        for _ in range(rng.randint(3, 5)):
            enrollment_target = rng.randint(15, 35)
            activation_date = start_date + timedelta(days=rng.randint(30, 150))
            trial_sites.append(
                {
                    "site_id": f"SITE-{site_counter:03d}",
                    "trial_id": trial_id,
                    "region": rng.choice(REGIONS),
                    "activation_date": iso(activation_date),
                    "enrollment_target": enrollment_target,
                }
            )
            site_counter += 1

        trials.append(
            {
                "trial_id": trial_id,
                "phase": rng.choice(PHASES),
                "indication": rng.choice(INDICATIONS),
                "start_date": iso(start_date),
                "target_enrollment": sum(site["enrollment_target"] for site in trial_sites),
            }
        )
        sites.extend(trial_sites)

    return trials, sites


def generate_subjects(
    fake: Faker, rng: random.Random, sites: list[Row]
) -> tuple[list[Row], dict[str, dict[str, date | None]]]:
    subjects: list[Row] = []
    subject_dates: dict[str, dict[str, date | None]] = {}
    subject_counter = 1

    for site in sites:
        activation_date = date.fromisoformat(site["activation_date"])

        for _ in range(rng.randint(8, 14)):
            subject_id = f"SUBJ-{subject_counter:04d}"
            screening_status = rng.choices(
                SCREENING_STATUSES,
                weights=[45, 35, 12, 8],
                k=1,
            )[0]
            enrolled = screening_status in {"Enrolled", "Randomized", "Withdrawn"}
            if enrolled:
                enrollment_date = fake_date_between(
                    fake,
                    activation_date + timedelta(days=12),
                    activation_date + timedelta(days=224),
                )
                screening_date = enrollment_date - timedelta(days=7)
            else:
                enrollment_date = None
                screening_date = fake_date_between(
                    fake,
                    activation_date + timedelta(days=5),
                    activation_date + timedelta(days=210),
                )

            subjects.append(
                {
                    "subject_id": subject_id,
                    "site_id": site["site_id"],
                    "treatment_arm": rng.choice(TREATMENT_ARMS) if enrolled else "Not Assigned",
                    "screening_status": screening_status,
                    "enrollment_date": iso(enrollment_date),
                }
            )
            subject_dates[subject_id] = {
                "screening_date": screening_date,
                "enrollment_date": enrollment_date,
            }
            subject_counter += 1

    return subjects, subject_dates


def generate_visits(rng: random.Random, subject_dates: dict[str, dict[str, date | None]]) -> list[Row]:
    visits: list[Row] = []
    visit_counter = 1

    for subject_id, dates in subject_dates.items():
        enrollment_date = dates["enrollment_date"]
        screening_date = dates["screening_date"]
        if enrollment_date is None:
            visits.append(
                {
                    "visit_id": f"VISIT-{visit_counter:06d}",
                    "subject_id": subject_id,
                    "visit_name": "Screening",
                    "planned_date": iso(screening_date),
                    "actual_date": iso(screening_date),
                    "status": "Screen Failed",
                }
            )
            visit_counter += 1
            continue

        for visit_name, offset_days in VISIT_SCHEDULE:
            planned_date = enrollment_date + timedelta(days=offset_days)
            if visit_name in {"Screening", "Baseline"}:
                status = "Completed"
            else:
                status = rng.choices(
                    ["Completed", "Missed", "Pending"],
                    weights=[88, 8, 4],
                    k=1,
                )[0]
            actual_date = (
                planned_date + timedelta(days=rng.randint(-3, 7))
                if status == "Completed"
                else None
            )
            visits.append(
                {
                    "visit_id": f"VISIT-{visit_counter:06d}",
                    "subject_id": subject_id,
                    "visit_name": visit_name,
                    "planned_date": iso(planned_date),
                    "actual_date": iso(actual_date),
                    "status": status,
                }
            )
            visit_counter += 1

    return visits


def generate_adverse_events(
    rng: random.Random, subject_dates: dict[str, dict[str, date | None]]
) -> list[Row]:
    adverse_events: list[Row] = []

    for subject_id, dates in subject_dates.items():
        enrollment_date = dates["enrollment_date"]
        if enrollment_date is None or rng.random() > 0.32:
            continue

        for _ in range(rng.randint(1, 2)):
            severity = rng.choices(SEVERITIES, weights=[55, 35, 10], k=1)[0]
            serious = severity == "Severe" and rng.random() < 0.55
            onset_date = enrollment_date + timedelta(days=rng.randint(1, 120))
            adverse_events.append(
                {
                    "subject_id": subject_id,
                    "event_term": rng.choice(ADVERSE_EVENT_TERMS),
                    "severity": severity,
                    "serious": serious,
                    "onset_date": iso(onset_date),
                    "outcome": rng.choice(OUTCOMES),
                }
            )

    return adverse_events


def generate_screen_failures(
    fake: Faker,
    rng: random.Random,
    subjects: list[Row],
    subject_dates: dict[str, dict[str, date | None]],
) -> list[Row]:
    failed_subjects = [subject for subject in subjects if subject["screening_status"] == "Screen Failed"]
    target = min(200, max(100, len(failed_subjects) * 4))
    assigned: dict[str, set[str]] = {subject["subject_id"]: set() for subject in failed_subjects}
    rows: list[Row] = []

    while len(rows) < target:
        subject = failed_subjects[len(rows) % len(failed_subjects)]
        subject_id = subject["subject_id"]
        available = [item for item in SCREEN_FAILURE_CRITERIA if item[1] not in assigned[subject_id]]
        category, criterion_code, reason = rng.choice(available)
        assigned[subject_id].add(criterion_code)
        rows.append(
            {
                "screen_failure_id": f"SF-{len(rows) + 1:06d}",
                "subject_id": subject_id,
                "site_id": subject["site_id"],
                "screening_date": iso(subject_dates[subject_id]["screening_date"]),
                "failure_category": category,
                "criterion_code": criterion_code,
                "failure_reason": reason,
                "rescreen_allowed": rng.random() < (0.1 if category in {"Consent", "Withdrawal"} else 0.22),
                "notes": fake.sentence(nb_words=9) if rng.random() < 0.35 else "",
            }
        )

    return rows


def _site_activity_end(
    site_id: str,
    subjects: list[Row],
    visits: list[Row],
    activation_date: date,
) -> date:
    subject_ids = {subject["subject_id"] for subject in subjects if subject["site_id"] == site_id}
    activity_dates = [
        date.fromisoformat(visit["actual_date"] or visit["planned_date"])
        for visit in visits
        if visit["subject_id"] in subject_ids
    ]
    return max(activity_dates, default=activation_date + timedelta(days=365))


def generate_protocol_deviations(
    fake: Faker,
    rng: random.Random,
    sites: list[Row],
    subjects: list[Row],
    visits: list[Row],
    total: int = 180,
) -> list[Row]:
    site_by_id = {site["site_id"]: site for site in sites}
    subjects_by_site: dict[str, list[Row]] = {site_id: [] for site_id in site_by_id}
    for subject in subjects:
        if subject["enrollment_date"]:
            subjects_by_site[subject["site_id"]].append(subject)
    visits_by_subject: dict[str, list[Row]] = {}
    for visit in visits:
        visits_by_subject.setdefault(visit["subject_id"], []).append(visit)

    rows: list[Row] = []
    for sequence in range(1, total + 1):
        site = rng.choice(sites)
        site_id = site["site_id"]
        subject_pool = subjects_by_site[site_id]
        subject = rng.choice(subject_pool) if subject_pool and rng.random() < 0.86 else None
        activation_date = date.fromisoformat(site["activation_date"])

        if subject is not None:
            subject_id = subject["subject_id"]
            start_date = date.fromisoformat(subject["enrollment_date"]) - timedelta(days=7)
            end_date = max(
                date.fromisoformat(visit["actual_date"] or visit["planned_date"])
                for visit in visits_by_subject[subject_id]
            )
        else:
            subject_id = ""
            start_date = activation_date
            end_date = _site_activity_end(site_id, subjects, visits, activation_date)

        deviation_date = fake_date_between(fake, start_date, end_date)
        detected_date = deviation_date + timedelta(days=rng.randint(0, 14))
        category = rng.choice(list(DEVIATION_DESCRIPTIONS))
        severity = rng.choices(["Minor", "Major", "Critical"], weights=[72, 25, 3], k=1)[0]
        status = rng.choices(["Open", "Investigating", "Resolved"], weights=[12, 18, 70], k=1)[0]
        resolved_date = (
            detected_date + timedelta(days=rng.randint(1, 60)) if status == "Resolved" else None
        )
        reported_to_irb = (
            severity == "Critical"
            or (severity == "Major" and rng.random() < 0.9)
            or (severity == "Minor" and rng.random() < 0.12)
        )

        rows.append(
            {
                "deviation_id": f"PD-{sequence:06d}",
                "subject_id": subject_id,
                "site_id": site_id,
                "deviation_date": iso(deviation_date),
                "detected_date": iso(detected_date),
                "category": category,
                "severity": severity,
                "description": rng.choice(DEVIATION_DESCRIPTIONS[category]),
                "reported_to_irb": reported_to_irb,
                "corrective_action": rng.choice(CORRECTIVE_ACTIONS[category]),
                "status": status,
                "resolved_date": iso(resolved_date),
            }
        )

    return rows


def _lab_value(rng: random.Random, profile: LabProfile) -> float:
    draw = rng.random()
    if draw < 0.08:
        value = rng.uniform(profile.reference_low * 0.65, profile.reference_low * 0.95)
    elif draw < 0.16:
        value = rng.uniform(profile.reference_high * 1.05, profile.reference_high * 1.35)
    else:
        midpoint = (profile.reference_low + profile.reference_high) / 2
        value = rng.gauss(midpoint, (profile.reference_high - profile.reference_low) / 6)
        value = min(profile.reference_high * 0.99, max(profile.reference_low * 1.01, value))
    return round(value, profile.decimals)


def _abnormal_flag(value: float, profile: LabProfile) -> str:
    if value < profile.reference_low:
        return "Low"
    if value > profile.reference_high:
        return "High"
    return "Normal"


def generate_lab_results(rng: random.Random, subjects: list[Row], visits: list[Row]) -> list[Row]:
    enrolled_subject_ids = {subject["subject_id"] for subject in subjects if subject["enrollment_date"]}
    eligible_visits = [
        visit
        for visit in visits
        if visit["subject_id"] in enrolled_subject_ids
        and visit["status"] == "Completed"
        and visit["actual_date"]
    ]
    rows: list[Row] = []

    def append_result(visit: Row, test_code: str, value: float, clinically_significant: bool) -> None:
        profile = LAB_PROFILES[test_code]
        visit_date = date.fromisoformat(visit["actual_date"])
        rows.append(
            {
                "lab_result_id": f"LAB-{len(rows) + 1:06d}",
                "subject_id": visit["subject_id"],
                "visit_id": visit["visit_id"],
                "collection_date": iso(visit_date + timedelta(days=rng.randint(-2, 2))),
                "test_code": test_code,
                "test_name": profile.name,
                "result_value": value,
                "unit": profile.unit,
                "reference_low": profile.reference_low,
                "reference_high": profile.reference_high,
                "abnormal_flag": _abnormal_flag(value, profile),
                "clinically_significant": clinically_significant,
                "reviewed": rng.random() < 0.94,
            }
        )

    for visit in eligible_visits:
        for test_code, profile in LAB_PROFILES.items():
            value = _lab_value(rng, profile)
            flag = _abnormal_flag(value, profile)
            clinically_significant = flag != "Normal" and rng.random() < 0.13
            append_result(visit, test_code, value, clinically_significant)

    # Small studies can land just below 5,000 rows. Add controlled repeat tests,
    # which is more realistic than inventing new analytes or results for missed visits.
    repeat_index = 0
    while len(rows) < 5000:
        visit = eligible_visits[repeat_index % len(eligible_visits)]
        test_code = "CREAT" if repeat_index % 2 == 0 else "ALT"
        profile = LAB_PROFILES[test_code]
        midpoint = round((profile.reference_low + profile.reference_high) / 2, profile.decimals)
        append_result(visit, test_code, midpoint, False)
        rows[-1]["reviewed"] = True
        repeat_index += 1

    return rows


def generate_data_queries(
    fake: Faker,
    rng: random.Random,
    subjects: list[Row],
    sites: list[Row],
    visits: list[Row],
    total: int = 800,
) -> list[Row]:
    site_activation = {site["site_id"]: date.fromisoformat(site["activation_date"]) for site in sites}
    visits_by_subject: dict[str, list[Row]] = {}
    for visit in visits:
        visits_by_subject.setdefault(visit["subject_id"], []).append(visit)

    rows: list[Row] = []
    for sequence in range(1, total + 1):
        subject = rng.choice(subjects)
        subject_id = subject["subject_id"]
        subject_visits = visits_by_subject[subject_id]
        visit = rng.choice(subject_visits) if rng.random() < 0.82 else None
        query_type = rng.choices(
            ["Missing", "Inconsistent", "Outlier", "Clarification"],
            weights=[32, 28, 15, 25],
            k=1,
        )[0]
        field_name = rng.choice(QUERY_FIELDS[query_type])

        if visit is not None:
            source_date = date.fromisoformat(visit["actual_date"] or visit["planned_date"])
        elif subject["enrollment_date"]:
            source_date = date.fromisoformat(subject["enrollment_date"])
        else:
            source_date = site_activation[subject["site_id"]]

        opened_date = fake_date_between(
            fake,
            source_date,
            source_date + timedelta(days=30),
        )
        status = rng.choices(["Open", "Answered", "Closed"], weights=[20, 20, 60], k=1)[0]
        response_date = (
            opened_date + timedelta(days=rng.randint(1, 15))
            if status in {"Answered", "Closed"}
            else None
        )
        closed_date = (
            response_date + timedelta(days=rng.randint(0, 7)) if status == "Closed" else None
        )
        resolution_hours = (
            (closed_date - opened_date).days * 24 if closed_date is not None else None
        )

        rows.append(
            {
                "query_id": f"DQ-{sequence:06d}",
                "subject_id": subject_id,
                "site_id": subject["site_id"],
                "visit_id": visit["visit_id"] if visit is not None else "",
                "opened_date": iso(opened_date),
                "query_type": query_type,
                "field_name": field_name,
                "priority": rng.choices(["Low", "Medium", "High"], weights=[48, 37, 15], k=1)[0],
                "query_text": QUERY_TEXT[field_name],
                "status": status,
                "response_date": iso(response_date),
                "closed_date": iso(closed_date),
                "resolution_hours": "" if resolution_hours is None else resolution_hours,
            }
        )

    return rows


def generate_site_monitoring(
    fake: Faker,
    rng: random.Random,
    sites: list[Row],
    protocol_deviations: list[Row],
    data_queries: list[Row],
) -> list[Row]:
    deviation_counts: dict[str, int] = {}
    critical_counts: dict[str, int] = {}
    open_query_counts: dict[str, int] = {}
    for deviation in protocol_deviations:
        site_id = deviation["site_id"]
        deviation_counts[site_id] = deviation_counts.get(site_id, 0) + 1
        if deviation["severity"] == "Critical":
            critical_counts[site_id] = critical_counts.get(site_id, 0) + 1
    for query in data_queries:
        if query["status"] != "Closed":
            site_id = query["site_id"]
            open_query_counts[site_id] = open_query_counts.get(site_id, 0) + 1

    rows: list[Row] = []
    windows = [(0, 14), (90, 180), (210, 330), (340, 500)]
    for site_index, site in enumerate(sites):
        site_id = site["site_id"]
        activation_date = date.fromisoformat(site["activation_date"])
        deviations = deviation_counts.get(site_id, 0)
        critical = critical_counts.get(site_id, 0)
        open_queries = open_query_counts.get(site_id, 0)
        if critical > 0 or (deviations >= 13 and open_queries >= 22):
            risk_rating = "High"
        elif deviations >= 9 or open_queries >= 16:
            risk_rating = "Medium"
        else:
            risk_rating = "Low"

        for visit_index, (start_offset, end_offset) in enumerate(windows):
            visit_date = fake_date_between(
                fake,
                activation_date + timedelta(days=start_offset),
                activation_date + timedelta(days=end_offset),
            )
            if visit_index == 0:
                visit_type = "Initiation"
            elif visit_index == 3 and risk_rating == "High":
                visit_type = "For-cause"
            elif visit_index == 3 and rng.random() < 0.35:
                visit_type = "Closeout"
            else:
                visit_type = "Interim"

            if visit_date > DATA_CUTOFF_DATE:
                status = "Planned"
            elif visit_date > DATA_CUTOFF_DATE - timedelta(days=10):
                status = "Report Pending"
            else:
                status = rng.choices(["Completed", "Report Pending"], weights=[88, 12], k=1)[0]

            if status == "Planned":
                findings_count = 0
                critical_findings = 0
                open_actions = 0
            else:
                finding_ranges = {"Low": (0, 3), "Medium": (2, 6), "High": (5, 8)}
                findings_count = rng.randint(*finding_ranges[risk_rating])
                critical_findings = (
                    rng.randint(0, min(2, findings_count)) if risk_rating == "High" else 0
                )
                open_actions = rng.randint(0, findings_count)

            report_date = visit_date + timedelta(days=rng.randint(1, 10))
            next_visit_date = (
                None
                if visit_type == "Closeout"
                else visit_date + timedelta(days=rng.randint(30, 120))
            )
            note_prefix = {
                "Low": "Routine monitoring completed with limited follow-up.",
                "Medium": "Monitoring identified items requiring site follow-up.",
                "High": "Enhanced oversight is required based on current findings.",
            }[risk_rating]
            notes = note_prefix
            if rng.random() < 0.35:
                notes = f"{note_prefix} {fake.sentence(nb_words=8)}"

            rows.append(
                {
                    "monitoring_visit_id": f"MV-{len(rows) + 1:06d}",
                    "site_id": site_id,
                    "visit_date": iso(visit_date),
                    "visit_type": visit_type,
                    "monitor_id": f"MON-{((site_index * 2 + visit_index) % 12) + 1:03d}",
                    "risk_rating": risk_rating,
                    "findings_count": findings_count,
                    "critical_findings": critical_findings,
                    "open_actions": open_actions,
                    "report_date": iso(report_date),
                    "next_visit_date": iso(next_visit_date),
                    "status": status,
                    "notes": notes,
                }
            )

    return rows


def generate_dataset(seed: int = DEFAULT_SEED) -> dict[str, list[Row]]:
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("en_US")

    trials, sites = generate_trials_and_sites(fake, rng)
    subjects, subject_dates = generate_subjects(fake, rng, sites)
    visits = generate_visits(rng, subject_dates)
    adverse_events = generate_adverse_events(rng, subject_dates)

    screen_failures = generate_screen_failures(fake, rng, subjects, subject_dates)
    protocol_deviations = generate_protocol_deviations(fake, rng, sites, subjects, visits)
    lab_results = generate_lab_results(rng, subjects, visits)
    data_queries = generate_data_queries(fake, rng, subjects, sites, visits)
    site_monitoring = generate_site_monitoring(
        fake,
        rng,
        sites,
        protocol_deviations,
        data_queries,
    )

    return {
        "trials": trials,
        "sites": sites,
        "subjects": subjects,
        "visits": visits,
        "adverse_events": adverse_events,
        "screen_failures": screen_failures,
        "protocol_deviations": protocol_deviations,
        "lab_results": lab_results,
        "data_queries": data_queries,
        "site_monitoring": site_monitoring,
    }


def write_dataset(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = generate_dataset(seed)
    if tuple(tables) != TABLE_ORDER:
        raise ValueError("Generated table order does not match schema dependency order.")

    counts: dict[str, int] = {}
    for table_name in TABLE_ORDER:
        rows = tables[table_name]
        fieldnames = [column.name for column in TABLE_SCHEMAS[table_name]]
        for row in rows:
            if set(row) != set(fieldnames):
                raise ValueError(f"{table_name} row fields do not match the declared schema.")
        write_csv(output_dir / f"{table_name}.csv", fieldnames, rows)
        counts[table_name] = len(rows)
    return counts
