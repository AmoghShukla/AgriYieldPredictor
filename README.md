**# AgriYieldPredictor**
A Python-based agriculture crop prediction system using Random Forest regression, Tkinter for GUI, and integrated data visualization for yield, profit, and fertilizer recommendations. Ideal for farmers and agricultural planners to optimize crop selection and resource utilization.



**Overview:-**

AgriYieldPredictor is an agriculture crop prediction system built using Python. This tool uses machine learning to predict crop yields, estimate profits, and provide fertilizer recommendations based on user inputs such as temperature, humidity, rainfall, soil properties, crop type, and season. The application features an interactive GUI built with Tkinter and includes data visualizations using Matplotlib.



**Features:-**

🔹Crop Yield Prediction: Predicts crop yield per hectare using a Random Forest regression model.

🔹Profit Estimation: Estimates profit based on predicted yield, market prices, and cultivation costs.

🔹Fertilizer Recommendations: Suggests fertilizers tailored to the crop type and season.

🔹Data Visualization: Provides visual insights for yield vs. area, profit vs. area, crop cost vs. profit, and the best crop for the season.

🔹User-Friendly GUI: Includes an intuitive graphical user interface for easy interaction.


**Prerequisites:-**

Python 3.8 or above

Required Python libraries:

🔹tkinter

🔹sklearn

🔹pandas

🔹numpy

🔹matplotlib



**Usage:-**

🔹Launch the application by running python main.py.

🔹Enter the required inputs such as temperature, humidity, rainfall, soil moisture, soil pH, crop type, and season.

🔹Click on the Predict button to get the results.

🔹View the predicted yield, estimated profit, recommended fertilizer, and growth duration in the results window.

🔹Analyze the provided charts for additional insights.



**Data Inputs:-**

Temperature (°C): Average temperature of the region.

Humidity (%): Humidity percentage during the growing period.

Rainfall (mm): Expected or recorded rainfall.

Soil Moisture (%): Soil moisture content.

Soil pH: Soil acidity/alkalinity level.

Crop Type: Select one of the available crops: Wheat, Rice, Corn, or Soybean.

Season: Choose from Kharif, Rabi, or Zaid.



**Visualization Insights:-**

Yield vs Area: Predicts yield for different land sizes.

Profit vs Area: Visualizes profit trends based on land size.

Best Crop for Season: Highlights the most profitable crop for the selected season.

Crop Cost vs Profit: Compares the cultivation cost with expected profit.




**Model:-**

The application uses a pre-trained Random Forest regression model for predictions. The sample dataset and dummy data are included for demonstration purposes. You can replace the model with your trained dataset for real-world applications.



**Contributing:-**

Contributions are welcome! Please follow these steps:

1)Fork the repository.

2)Create a new branch for your feature or bug fix.

3)Commit your changes and push to your branch.

4)Open a pull request.



**Contact:-**

For any questions or feedback, please contact amoghpvt1@gmail.com

