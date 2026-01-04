import streamlit as st
import pickle
import os

@st.cache_resource
def load_model():
    model_path = "spam_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload spam_model.pkl")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


# -------------------------------
# Main App Function
# -------------------------------
def main():
    st.set_page_config(
        page_title="Spam Mail Detector",
        page_icon="📧",
        layout="centered"
    )

    st.title("📧 Spam Mail Detection App")
    st.write(
        "This application uses **Machine Learning (TF-IDF + Naive Bayes)** "
        "to detect whether an email message is **Spam** or **Not Spam**."
    )

    st.markdown("---")

    # Load trained model
    model = load_model()

    # User input
    email_text = st.text_area(
        "✉️ Enter Email Content:",
        height=220,
        placeholder="Paste or type the email message here..."
    )

    # Predict button
    if st.button("🔍 Check Spam"):
        if not email_text.strip():
            st.warning("⚠️ Please enter some email text to analyze.")
        else:
            prediction = model.predict([email_text])[0]
            probability = model.predict_proba([email_text])[0]

            st.markdown("### 🔎 Result")

            if prediction == 1:
                st.error(
                    f"🚨 **SPAM EMAIL**\n\n"
                    f"Confidence: **{probability[1] * 100:.2f}%**"
                )
            else:
                st.success(
                    f"✅ **NOT SPAM (HAM)**\n\n"
                    f"Confidence: **{probability[0] * 100:.2f}%**"
                )

    st.markdown("---")
    st.caption("🔐 Model trained on Kaggle Spam Mails Dataset")

if __name__ == "__main__":
    main()
