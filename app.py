from pydantic_settings import BaseSettings
import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import (
    setup,
    compare_models,
    pull,
    save_model,
    create_model,
    models,
    finalize_model,
    predict_model,
)

# from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("automlify.jpg", use_column_width=True)
    st.title("AutoMLify")

    # Use session state to manage choice
    if "choice" not in st.session_state:
        st.session_state.choice = "Upload"  # Default choice

    st.session_state.choice = st.radio(
        "Navigation",
        [
            "Upload Dataset",
            "Profiling Data",
            "ML Training",
            "Prediction",
            "Download Model",
        ],
    )

    st.info(
        "Welcome to AutoMLify! This app automates machine learning tasks such as data profiling, preprocessing, model training, and evaluation using PyCaret. Easily upload your dataset, explore its structure, train multiple models, and download the results."
    )

# if os.path.exists("source_data.csv"):
#     df = pd.read_csv("source_data.csv", index_col=None)
# else:
#     df = None  # Handle case where dataset is not uploaded


# Initialize session state for dataset if not already set
if "df" not in st.session_state:
    st.session_state.df = None  # Initialize with None


if st.session_state.choice == "Upload Dataset":
    st.header("📁 Upload Your Dataset")
    st.markdown("### Welcome! Please upload your dataset to get started.")
    dataset = st.file_uploader("Upload your Dataset here", type=["csv"])
    if dataset:
        st.session_state.df = pd.read_csv(
            dataset, index_col=None
        )  # Store dataset in session state
        # df.to_csv("source_data.csv", index=None)

        st.success(f"✅ Your file has been uploaded and saved!")

        st.subheader("🔍 Dataset Preview")
        st.dataframe(st.session_state.df)

        st.markdown("### Next Steps:")
        st.write("Now that you've uploaded your dataset, you can:")
        st.markdown("- 📊 **Perform EDA**: Head to **Profiling Data**.")
        st.markdown("- 🤖 **Model Training**: Go to **ML Training**.")


if st.session_state.choice == "Profiling Data":
    st.header("📊 Automated Data Profiling Report")

    # Check if the dataset is available
    if st.session_state.df is None:
        st.error(
            "⚠️ No dataset found. Please upload a dataset first in the **Upload Dataset** section."
        )
    else:
        st.subheader(
            "Get insights into your dataset with our automated profiling tools."
        )
        if st.button("🔍 Start Analysis..."):
            # with st.spinner("Generating the profiling report... Please wait!"):
            profile_report = ProfileReport(
                st.session_state.df,
                title="Pandas Profiling Report",
                explorative=True,
            )
            st_profile_report(profile_report)
            st.success("The profiling report has been generated successfully!")


if st.session_state.choice == "ML Training":
    st.header("🤖 ML Model Training")

    # Check if the dataset is available
    if st.session_state.df is None:
        st.error(
            "⚠️ No dataset found. Please upload a dataset first in the **Upload Dataset** section."
        )
    else:
        st.info(
            "Select your target variable and choose from multiple training options for your models."
        )

        target = st.selectbox(
            "Select Your Target Variable", st.session_state.df.columns
        )

        # Store the target in session state
        st.session_state["target"] = target

        if st.button("Train All Models"):
            with st.spinner(
                "🚀 Training All Models... Please be patient as we work on optimizing your machine learning models!"
            ):
                setup(st.session_state.df, target=target, verbose=False)
                setup_df = pull()

                # Display setup information
                st.subheader("Setup Summary:")
                st.dataframe(setup_df)

                # Compare models and display the results
                best_model = compare_models()
                compare_df = pull()

                # Store the best model in session state
                st.session_state["best_model"] = best_model

                st.subheader("Comparison of Models:")
                st.dataframe(compare_df)

                st.success(
                    "✅ All models have been trained successfully! The best performing model has been saved. "
                    "You can proceed to the **Download Model** section to download it."
                )

                save_model(best_model, "best_model")


# Initialize session state for test dataset if not already set
if "df_test" not in st.session_state:
    st.session_state.df_test = None  # Initialize with None


if st.session_state.choice == "Prediction":
    st.header("🔮 Make Predictions with the Best Model")
    st.info("Use the best-performing model to make predictions on new data.")
    st.markdown(
        """
    - **Step 1:** Upload a dataset that matches the format of the training data for prediction.
    - **Step 2:** If no dataset is uploaded, the model will automatically use the test set generated during the training process.
    """
    )

    # Check if the best model is available
    if "best_model" not in st.session_state:
        st.error(
            "No trained model found. Please train a model first in the ML Training section."
        )
    else:
        best_model = st.session_state["best_model"]
        target = st.session_state["target"]

        # Upload Test Dataset
        test_dataset = st.file_uploader("📤 Upload Your Test Dataset for Prediction")

        if st.button("Make Predictions..."):
            if test_dataset:
                st.session_state.df_test = pd.read_csv(test_dataset, index_col=None)
                st.write(
                    "✅ Dataset uploaded successfully! Proceeding with prediction..."
                )

                try:
                    predictions = predict_model(
                        best_model,
                        data=st.session_state.df_test.drop([target], axis=1),
                        raw_score=True,
                    )
                    st.success("Prediction completed on the uploaded dataset!")
                    st.dataframe(predictions)  # Display predictions

                    # Calculate and display metrics
                    metrics = pull()  # This function pulls the performance metrics
                    st.subheader("Model Performance Metrics:")
                    st.dataframe(metrics)  # Display performance metrics

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.write("No dataset uploaded. Predicting on the default test set...")
                try:
                    predictions = predict_model(best_model)
                    st.success("Prediction completed on the test set!")
                    st.dataframe(predictions)  # Display predictions

                    # Calculate and display metrics
                    metrics = pull()  # Pull performance metrics
                    st.subheader("Model Performance Metrics:")
                    st.dataframe(metrics)  # Display performance metrics

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")


if st.session_state.choice == "Download Model":
    st.header("📥 Download Your Best Model")
    st.info(
        "The best model will be saved during training, and you will be able to download it directly. "
        "Alternatively, you can finalize another model for download. The model will be saved as a pickle file."
    )

    if "best_model" in st.session_state:
        with st.spinner("Preparing your best model for download..."):
            save_model(st.session_state["best_model"], "best_model.pkl")
            with open("best_model.pkl", "rb") as f:
                st.success("The best model is ready for download!")
                st.download_button("Download the Best Model", f, "best_model.pkl")
    else:
        st.warning(
            "⚠️ No model available for download. Please go to the **ML Training** section to train your models first."
        )

    st.subheader("🔧 Finalize Another Model")
    model_id = st.text_input("Enter the Model ID you wish to finalize:")

    with st.spinner("Finalizing your model... Please wait."):
        try:
            if st.button("Finalize and Download"):
                if model_id:
                    finalize_model(model_id)
                    save_model(model_id, "finalized_model")
                    with open("finalized_model.pkl", "rb") as f:
                        st.success(
                            f"Model {model_id} has been finalized and is ready for download!"
                        )
                        st.download_button("Download Finalized Model", f, "model.pkl")
        except:
            st.error("Please enter a valid model ID or Train the Models first.")
