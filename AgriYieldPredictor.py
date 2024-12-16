import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sample data for scaling and feature encoding
scaler = StandardScaler()
dummy_data = pd.DataFrame(
    {
        'temperature': [25], 'humidity': [60], 'rainfall': [100],
        'soil_moisture': [30], 'soil_ph': [6.5],
        'season_Kharif': [0], 'season_Rabi': [0], 'season_Zaid': [1],
        'Crop_Category_Rice': [0], 'Crop_Category_Soybean': [0], 'Crop_Category_Wheat': [1],
    }
)
scaler.fit(dummy_data)

# Updated fertilizer recommendations
fertilizer_recommendations = {
    'Wheat': {'Rabi': 'Urea', 'Kharif': 'DAP', 'Zaid': 'NPK Complex'},
    'Rice': {'Kharif': 'MOP', 'Rabi': 'Superphosphate', 'Zaid': 'Ammonium Sulfate'},
    'Corn': {'Kharif': 'Ammonium Nitrate', 'Rabi': 'Potash', 'Zaid': 'Potash'},
    'Soybean': {'Kharif': 'NPK Complex', 'Rabi': 'Gypsum', 'Zaid': 'Biofertilizers'},
}

# Crop prices and growth duration
crop_prices = {'Wheat': 20, 'Rice': 25, 'Corn': 22, 'Soybean': 30}
crop_costs = {'Wheat': 15000, 'Rice': 20000, 'Corn': 18000, 'Soybean': 25000}
growth_duration = {'Wheat': 120, 'Rice': 150, 'Corn': 100, 'Soybean': 130}

# Sample trained Random Forest model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(dummy_data, [2000])  # Example model training


def create_gui():
    def reset_inputs():
        temperature_var.set("")
        humidity_var.set("")
        rainfall_var.set("")
        soil_moisture_var.set("")
        soil_ph_var.set("")
        crop_type_var.set("Select")
        season_var.set("Select")

    def show_results(predicted_yield, profit, fertilizer, crop_type, season):
        results_window = tk.Toplevel(root)
        results_window.title("Prediction Results")
        results_window.geometry("1024x768")
        results_window.configure(bg="#dff0d8")

        results_text = (
            f"Predicted Crop Yield: {predicted_yield:.2f} kg/hectare\n"
            f"Estimated Profit: ₹{profit:.2f} per hectare\n"
            f"Growth Duration: {growth_duration[crop_type]} days\n"
            f"Recommended Fertilizer: {fertilizer} for {season}"
        )
        results_label = tk.Label(
            results_window, text=results_text, font=("Arial", 16), bg="#dff0d8", fg="green", justify="left"
        )
        results_label.pack(pady=10)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        areas = np.linspace(1, 10, 10)  # Simulate for 1 to 10 hectares
        yields = predicted_yield * areas
        axs[0, 0].plot(areas, yields, marker='o', color='blue')
        axs[0, 0].set_title("Yield vs Area")
        axs[0, 0].set_xlabel("Area (hectares)")
        axs[0, 0].set_ylabel("Yield (kg)")

        profits = (yields * crop_prices[crop_type]) - (crop_costs[crop_type] * areas)
        axs[0, 1].plot(areas, profits, marker='o', color='green')
        axs[0, 1].set_title("Profit vs Area")
        axs[0, 1].set_xlabel("Area (hectares)")
        axs[0, 1].set_ylabel("Profit (₹)")

        best_crop_profits = {c: crop_prices[c] * predicted_yield - crop_costs[c] for c in crop_prices}
        crops = list(best_crop_profits.keys())
        profits = list(best_crop_profits.values())
        axs[1, 0].bar(crops, profits, color='orange')
        axs[1, 0].set_title("Best Crop for the Season")
        axs[1, 0].set_xlabel("Crops")
        axs[1, 0].set_ylabel("Profit (₹)")

        axs[1, 1].bar(crops, [crop_costs[c] for c in crops], label="Cost", alpha=0.7)
        axs[1, 1].bar(crops, profits, label="Profit", alpha=0.7)
        axs[1, 1].set_title("Crop Cost vs Profit")
        axs[1, 1].set_xlabel("Crops")
        axs[1, 1].set_ylabel("Amount (₹)")
        axs[1, 1].legend()

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, results_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def predict():
        try:
            temperature = float(temperature_var.get())
            humidity = float(humidity_var.get())
            rainfall = float(rainfall_var.get())
            soil_moisture = float(soil_moisture_var.get())
            soil_ph = float(soil_ph_var.get())
            crop_type = crop_type_var.get()
            season = season_var.get()

            if crop_type == "Select" or season == "Select":
                raise ValueError("Please select valid crop type and season.")

            new_data = pd.DataFrame(
                [[temperature, humidity, rainfall, soil_moisture, soil_ph, season, crop_type]],
                columns=['temperature', 'humidity', 'rainfall', 'soil_moisture', 'soil_ph', 'season', 'Crop_Category']
            )
            new_data = pd.get_dummies(new_data, columns=['season', 'Crop_Category'], drop_first=True)
            new_data = new_data.reindex(columns=dummy_data.columns, fill_value=0)
            new_data_scaled = scaler.transform(new_data)

            predicted_yield = model.predict(new_data_scaled)[0]
            profit = predicted_yield * crop_prices[crop_type] - crop_costs[crop_type]
            fertilizer = fertilizer_recommendations[crop_type][season]

            show_results(predicted_yield, profit, fertilizer, crop_type, season)
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except KeyError as e:
            messagebox.showerror("Error", f"Missing data for crop or season: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    root = tk.Tk()
    root.title("Agriculture Crop Prediction System")
    root.geometry("1024x768")
    root.configure(bg="#dff0d8")
    root.state("zoomed")

    title_label = tk.Label(root, text="Agriculture Crop Prediction System", font=("Arial", 24, "bold"), bg="#dff0d8", fg="green")
    title_label.pack(pady=20)

    input_frame = tk.Frame(root, bg="#dff0d8")
    input_frame.pack(pady=20)

    variables = {
        "Temperature (°C):": "temperature_var",
        "Humidity (%):": "humidity_var",
        "Rainfall (mm):": "rainfall_var",
        "Soil Moisture (%):": "soil_moisture_var",
        "Soil pH:": "soil_ph_var"
    }
    var_mapping = {}
    for i, (label_text, var_name) in enumerate(variables.items()):
        label = tk.Label(input_frame, text=label_text, font=("Arial", 14), bg="#dff0d8")
        label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
        var = tk.StringVar()
        entry = tk.Entry(input_frame, textvariable=var, font=("Arial", 14), width=20)
        entry.grid(row=i, column=1, padx=10, pady=5)
        var_mapping[var_name] = var

    temperature_var = var_mapping["temperature_var"]
    humidity_var = var_mapping["humidity_var"]
    rainfall_var = var_mapping["rainfall_var"]
    soil_moisture_var = var_mapping["soil_moisture_var"]
    soil_ph_var = var_mapping["soil_ph_var"]

    crop_type_var = tk.StringVar(value="Select")
    crop_type_label = tk.Label(input_frame, text="Crop Type:", font=("Arial", 14), bg="#dff0d8")
    crop_type_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    crop_type_dropdown = ttk.Combobox(input_frame, textvariable=crop_type_var, values=["Wheat", "Rice", "Corn", "Soybean"], font=("Arial", 14), state="readonly")
    crop_type_dropdown.grid(row=5, column=1, padx=10, pady=5)

    # Season dropdown
    season_var = tk.StringVar(value="Select")
    season_label = tk.Label(input_frame, text="Season:", font=("Arial", 14), bg="#dff0d8")
    season_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
    season_dropdown = ttk.Combobox(input_frame, textvariable=season_var, values=["Kharif", "Rabi", "Zaid"], font=("Arial", 14), state="readonly")
    season_dropdown.grid(row=6, column=1, padx=10, pady=5)

    # Buttons
    button_frame = tk.Frame(root, bg="#dff0d8")
    button_frame.pack(pady=20)

    predict_button = tk.Button(button_frame, text="Predict", font=("Arial", 16), bg="green", fg="white", command=predict)
    predict_button.grid(row=0, column=0, padx=10)

    reset_button = tk.Button(button_frame, text="Reset", font=("Arial", 16), bg="orange", fg="white", command=reset_inputs)
    reset_button.grid(row=0, column=1, padx=10)

    exit_button = tk.Button(button_frame, text="Exit", font=("Arial", 16), bg="red", fg="white", command=root.destroy)
    exit_button.grid(row=0, column=2, padx=10)

    root.mainloop()

# Run the GUI
create_gui()
