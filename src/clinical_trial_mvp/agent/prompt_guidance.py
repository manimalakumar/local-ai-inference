"""Versioned relational and business guidance for SQL generation."""

JOIN_AND_DEFINITION_GUIDE = """
Join map (use these keys; do not invent a shortcut column):
- sites.trial_id = trials.trial_id
- subjects.site_id = sites.site_id
- visits.subject_id = subjects.subject_id; a visit's site or region requires visits -> subjects -> sites. visits has NO site_id.
- adverse_events.subject_id = subjects.subject_id; its treatment_arm requires adverse_events -> subjects.
- screen_failures.subject_id = subjects.subject_id; screen_failures.site_id = sites.site_id. A subject can have several criterion-level failure records.
- protocol_deviations.site_id = sites.site_id; its optional subject_id can join subjects.subject_id.
- lab_results.visit_id = visits.visit_id; lab_results.subject_id = subjects.subject_id.
- data_queries.site_id = sites.site_id; data_queries.subject_id = subjects.subject_id; its visit_id can join visits.visit_id.
- site_monitoring.site_id = sites.site_id. The visit type column is visit_type; the date column is visit_date.

Business definitions for this synthetic operational dataset:
- Unless the user supplies another definition, "enrolled subjects" includes screening_status IN ('Enrolled', 'Randomized'); count subjects, not visit or adverse-event rows. The interactive app may ask for clarification first.
- A screen-failed subject has screening_status = 'Screen Failed'. Count DISTINCT subject_id from screen_failures because one subject may have multiple failed criteria. A site screen-failure rate uses distinct failed subjects divided by all screened subjects at that site; state the denominator.
- For a site ranking such as "which sites have the longest", return the sites with their metric ordered highest first. Use LIMIT only when the user asks for a top N or one site.
- "Overall" asks for one result row. Do not GROUP BY site, trial, arm, or month unless the question asks "by" or "for each" that dimension.
- The phrase "Site 17" refers to site_id = 'SITE-017' in this synthetic dataset. Do not infer a cause of poor performance from operational correlations.
- To compare a rate, report the numerator and denominator used. Put aggregate filters in HAVING or an outer query, never WHERE.

Illustrative DuckDB patterns (adapt to the question; do not copy extra columns or filters):
1. Join a visit to its region:
   SELECT st.region, COUNT(v.visit_id) AS completed_visits
   FROM visits AS v JOIN subjects AS sub ON v.subject_id = sub.subject_id
   JOIN sites AS st ON sub.site_id = st.site_id
   WHERE v.status = 'Completed' GROUP BY st.region;
2. Preserve zero-subject trials while counting enrolled people:
   SELECT t.trial_id, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects
   FROM trials AS t LEFT JOIN sites AS st ON t.trial_id = st.trial_id
   LEFT JOIN subjects AS sub ON st.site_id = sub.site_id
     AND sub.screening_status IN ('Enrolled', 'Randomized')
   GROUP BY t.trial_id;
3. Return an overall average without a site grouping:
   SELECT ROUND(AVG(resolution_hours), 2) AS overall_resolution_hours
   FROM data_queries WHERE status = 'Closed' AND resolution_hours IS NOT NULL;
""".strip()
