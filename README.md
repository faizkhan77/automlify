# AutoMLify

![AutoMLify Logo](automlify.jpg)  <!-- Replace with your logo image path -->

## Overview

Welcome to **AutoMLify**, a modern and user-friendly web application designed to simplify your machine learning (ML) workflow. With AutoMLify, you can automate the entire ML process, from data upload to model training and evaluation, all with just a few clicks. Whether you are a beginner or an experienced data scientist, AutoMLify makes it easy to handle complex ML tasks without any hassle.

## Features

- **Smooth User Experience**: The application is built with Streamlit, offering a seamless and intuitive interface.
  
- **Automated Data Handling**: Simply upload your raw dataset, and AutoMLify will take care of the rest. 

- **Deep Exploratory Data Analysis (EDA)**: Perform comprehensive EDA with stunning visualizations that provide insights into your data. AutoMLify generates detailed reports on each column, helping you understand the underlying patterns and distributions.

- **Data Cleaning and Preprocessing**: With just one click, the application automatically cleans and preprocesses your data, ensuring it is ready for model training.

- **Model Training and Comparison**: Train a wide range of machine learning models effortlessly. AutoMLify evaluates their performance based on various metrics, including:
  - Accuracy
  - ROC AUC
  - F1 Score
  - Precision
  - Recall

- **Best Model Selection**: The application automatically saves the best-performing model for your convenience.

- **Prediction and Model Download**: Make predictions on new datasets and download your trained model as a pickle file, all with just one click.

## Technologies Used

AutoMLify is built using the following technologies:
- **Streamlit**: For building the web application.
- **PyCaret**: To simplify the model training process.
- **Pandas**: For data manipulation and analysis.
- **YData Profiling**: For generating detailed EDA reports.

## Installation

**Important:** Please use Python 3.10 or below, as newer versions of Python may lead to version mismatches with Streamlit and PyCaret, which currently work only up to Python 3.10 or 3.11.

To get started with AutoMLify, follow these simple steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/AutoMLify.git
   cd AutoMLify

2. **Install the required packages**:
Make sure you have Python installed on your machine. Then, install the dependencies from the requirements.txt file:

   ```bash
   pip install -r requirements.txt
   
3. **Run the application**:

   ```bash
   streamlit run app.py


## Usage

1. Upload your dataset using the file uploader.
2. Navigate to the **Profiling** section to perform EDA.
3. Move to the **ML Training** section to train and compare models.
4. Download the best model or make predictions on new datasets as needed.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the authors of **Streamlit**, **PyCaret**, **Pandas**, and **YData Profiling** for providing excellent libraries that make this project possible.
