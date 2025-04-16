"""
Project 478. Clinical decision support system
Description:
A Clinical Decision Support System (CDSS) helps healthcare professionals make informed decisions by offering intelligent recommendations based on patient data. These can include diagnostic suggestions, treatment options, or alerts based on guidelines. In this project, we simulate a rule-based CDSS for chronic diseases using structured patient inputs.

About:
✅ What It Does:
Takes structured patient data as input.

Applies rule-based logic to generate actionable recommendations.

Easily extendable to:

Use AI models trained on patient outcome data

Generate personalized alerts, treatment options, or diagnostic differentials

Integrate with FHIR EHRs or deploy in clinical dashboards

Real-world integration:

Link to EHR systems using FHIR APIs

Use clinical guidelines (e.g., ADA, JNC8) or ML models trained on historical outcomes
"""

# 1. Simulated patient input
patient = {
    "age": 58,
    "gender": "male",
    "systolic_bp": 150,   # mmHg
    "diastolic_bp": 95,
    "fasting_glucose": 130,  # mg/dL
    "BMI": 29,
    "cholesterol": 220,
    "has_diabetes": True,
    "has_hypertension": True
}
 
# 2. Clinical decision rules
def decision_support(patient):
    recommendations = []
 
    # Hypertension management
    if patient["systolic_bp"] > 140 or patient["diastolic_bp"] > 90:
        recommendations.append("Suggest lifestyle changes and antihypertensive therapy.")
    else:
        recommendations.append("Blood pressure is under control. Continue current plan.")
 
    # Diabetes management
    if patient["fasting_glucose"] > 126 or patient["has_diabetes"]:
        recommendations.append("Recommend HbA1c test and review diabetes medications.")
    
    # Cardiovascular risk
    if patient["cholesterol"] > 200 or patient["BMI"] > 30:
        recommendations.append("Monitor lipid profile and discuss weight management.")
 
    # Preventive care
    if patient["age"] > 50:
        recommendations.append("Schedule colonoscopy and check vaccination status.")
 
    return recommendations
 
# 3. Output recommendations
print("Clinical Decision Support Recommendations:\n")
for rec in decision_support(patient):
    print("- " + rec)