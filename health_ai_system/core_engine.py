# ================= CORE HEALTH ENGINE =================

def analyze_patient(data):

    risks = {}
    reasons = []
    recommendations = []

    # ================= DIABETES =================
    diabetes_risk = 0

    if data.get("glucose", 0) > 140:
        diabetes_risk += 0.4
        reasons.append("High glucose level")

    if data.get("bmi", 0) > 30:
        diabetes_risk += 0.3
        reasons.append("High BMI (obesity)")

    if data.get("age", 0) > 45:
        diabetes_risk += 0.2
        reasons.append("Age risk factor")

    risks["Diabetes"] = min(diabetes_risk, 1.0)

    # ================= HEART =================
    heart_risk = 0

    if data.get("cholesterol", 0) > 200:
        heart_risk += 0.4
        reasons.append("High cholesterol")

    if data.get("smoking"):
        heart_risk += 0.3
        reasons.append("Smoking habit")

    if data.get("exercise") == "low":
        heart_risk += 0.2
        reasons.append("Low physical activity")

    risks["Heart Disease"] = min(heart_risk, 1.0)

    # ================= FUTURE RISK =================
    future_risk = {}

    for disease, value in risks.items():
        growth = 0

        if data.get("smoking"):
            growth += 0.1

        if data.get("exercise") == "low":
            growth += 0.1

        future_risk[disease] = min(value + growth, 1.0)

    # ================= RECOMMENDATIONS =================

    if data.get("glucose", 0) > 140:
        recommendations.append("Reduce sugar intake")

    if data.get("bmi", 0) > 30:
        recommendations.append("Start weight loss")

    if data.get("cholesterol", 0) > 200:
        recommendations.append("Avoid oily food")

    if data.get("exercise") == "low":
        recommendations.append("Exercise daily")

    if data.get("smoking"):
        recommendations.append("Quit smoking")

    return {
        "risks": risks,
        "future_risk": future_risk,
        "reasons": list(set(reasons)),
        "recommendations": list(set(recommendations))
    }