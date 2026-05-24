import gradio as gr
import pandas as pd
import joblib

# Load model and supporting files
model = joblib.load("customer_churn_model.pkl")
expected_features = joblib.load("feature_names.pkl")
city_list = joblib.load("city_list.pkl")

YES_NO = ["Yes", "No"]
GENDERS = ["Male", "Female"]
PAYMENT_METHODS = ["Bank Withdrawal", "Credit Card", "Mailed Check"]

SERVICE_COLS = [
    "Phone Service", "Multiple Lines", "Married", "Online Security",
    "Online Backup", "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music", "Senior Citizen",
    "Unlimited Data", "Under 30", "Dependents", "Referred a Friend",
    "Internet Service", "Paperless Billing",
]

NUMERIC_COLS = [
    "Age", "Number of Dependents", "Zip Code", "Population",
    "Number of Referrals", "Tenure in Months",
    "Avg Monthly Long Distance Charges", "Avg Monthly GB Download",
    "Monthly Charge", "Total Charges", "Total Refunds",
    "Total Extra Data Charges", "Total Long Distance Charges",
    "Total Revenue", "CLTV",
]


def predict_churn(*args):
    """Map positional Gradio inputs → DataFrame → model prediction."""
    values = list(args)
    user_inputs = dict(zip(expected_features, values))
    input_df = pd.DataFrame([user_inputs])

    probs = model.predict_proba(input_df)
    churn_prob = float(probs[0][1])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        label = "⚠️ Churn"
        confidence = churn_prob
        color = "color: #c0392b; font-size: 1.4em; font-weight: bold;"
    else:
        label = "✅ No Churn"
        confidence = float(probs[0][0])
        color = "color: #27ae60; font-size: 1.4em; font-weight: bold;"

    result_html = f"""
    <div style='padding: 16px; border-radius: 8px; background: #f9f9f9;'>
        <p style='{color}'>{label}</p>
        <p>Model confidence: <strong>{confidence * 100:.1f}%</strong></p>
        <div style='background:#e0e0e0; border-radius:4px; height:18px; width:100%;'>
            <div style='background:{"#e74c3c" if prediction==1 else "#2ecc71"};
                        width:{confidence*100:.1f}%; height:18px; border-radius:4px;'>
            </div>
        </div>
    </div>
    """
    return result_html


def build_input_components():
    """Build Gradio components in the same order as expected_features."""
    components = []

    with gr.Tabs():
        # ── Tab 1 : Demographics ──────────────────────────────────────────
        with gr.Tab("👤 Demographics"):
            with gr.Row():
                with gr.Column():
                    for col in expected_features:
                        if col in ("Age", "Number of Dependents", "Zip Code",
                                   "Population", "Number of Referrals"):
                            components.append(
                                (col, gr.Number(label=col, value=0))
                            )
                        elif col == "City":
                            components.append(
                                (col, gr.Dropdown(label="City", choices=city_list,
                                                  value=city_list[0]))
                            )
                        elif col == "Gender":
                            components.append(
                                (col, gr.Radio(label="Gender", choices=GENDERS,
                                               value="Male"))
                            )
                        elif col in ("Married", "Senior Citizen", "Under 30",
                                     "Dependents", "Referred a Friend"):
                            components.append(
                                (col, gr.Radio(label=col, choices=YES_NO,
                                               value="No"))
                            )

        # ── Tab 2 : Services ─────────────────────────────────────────────
        with gr.Tab("📡 Services"):
            SERVICE_TAB_COLS = [
                "Phone Service", "Multiple Lines", "Internet Service",
                "Online Security", "Online Backup", "Device Protection Plan",
                "Premium Tech Support", "Streaming TV", "Streaming Movies",
                "Streaming Music", "Unlimited Data",
            ]
            with gr.Row():
                with gr.Column():
                    for col in expected_features:
                        if col in SERVICE_TAB_COLS:
                            components.append(
                                (col, gr.Radio(label=col, choices=YES_NO,
                                               value="No"))
                            )

        # ── Tab 3 : Billing & Financials ─────────────────────────────────
        with gr.Tab("💳 Billing & Financials"):
            with gr.Row():
                with gr.Column():
                    for col in expected_features:
                        if col == "Payment Method":
                            components.append(
                                (col, gr.Dropdown(label="Payment Method",
                                                  choices=PAYMENT_METHODS,
                                                  value=PAYMENT_METHODS[0]))
                            )
                        elif col == "Paperless Billing":
                            components.append(
                                (col, gr.Radio(label="Paperless Billing",
                                               choices=YES_NO, value="No"))
                            )
                        elif col in ("Tenure in Months",
                                     "Avg Monthly Long Distance Charges",
                                     "Avg Monthly GB Download", "Monthly Charge",
                                     "Total Charges", "Total Refunds",
                                     "Total Extra Data Charges",
                                     "Total Long Distance Charges",
                                     "Total Revenue", "CLTV"):
                            components.append(
                                (col, gr.Number(label=col, value=0.0))
                            )

    # Catch-all: any feature not yet assigned gets a Number input
    assigned = {name for name, _ in components}
    for col in expected_features:
        if col not in assigned:
            components.append((col, gr.Number(label=col, value=0.0)))

    # Sort components to match expected_features order exactly
    order = {col: i for i, col in enumerate(expected_features)}
    components.sort(key=lambda x: order[x[0]])

    return [comp for _, comp in components]


with gr.Blocks(title="Churn Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📊 Customer Churn Predictor
        Fill in the customer details across the tabs below, then click **Predict**.
        """
    )

    input_components = build_input_components()

    predict_btn = gr.Button("🔍 Predict Churn", variant="primary", size="lg")
    output = gr.HTML(label="Prediction Result")

    predict_btn.click(
        fn=predict_churn,
        inputs=input_components,
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()