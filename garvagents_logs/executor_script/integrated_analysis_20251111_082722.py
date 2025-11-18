# ==========================
# Imports and Setup
# ==========================
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# Task 1: Generate Data for y = x^2
# ==========================
print("\n=== Task 1: Generating Data ===")
x_values = None
y_values = None

# Step 1: Generate x values
try:
    print("Generating x values from -10 to 10...")
    x_values = np.linspace(-10, 10, num=100)  # 100 points between -10 and 10
    print(f"x_values generated. Shape: {x_values.shape}")
except Exception as e:
    print(f"Error generating x values: {e}")

# Step 2: Compute y = x^2 for each x
try:
    if x_values is not None:
        print("Computing y = x^2 for each x value...")
        y_values = np.power(x_values, 2)
        print(f"y_values computed. Shape: {y_values.shape}")
    else:
        raise ValueError("x_values is None. Cannot compute y_values.")
except Exception as e:
    print(f"Error computing y values: {e}")

# Step 3: Save intermediate results
try:
    if x_values is not None and y_values is not None:
        np.save("x_values.npy", x_values)
        np.save("y_values.npy", y_values)
        print("Intermediate results saved to 'x_values.npy' and 'y_values.npy'.")
    else:
        print("Skipping saving intermediate results due to previous errors.")
except Exception as e:
    print(f"Error saving intermediate results: {e}")

# ==========================
# Task 2: Plot y = x^2
# ==========================
print("\n=== Task 2: Plotting Function ===")
try:
    if x_values is None or y_values is None:
        raise ValueError("x_values or y_values are None. Cannot plot.")

    print("Plotting y = x^2...")
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=r'$y = x^2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of $y = x^2$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot as a PNG file
    plot_filename = "y_equals_x_squared.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'.")
    plt.show()
    print("Plot displayed successfully.")
except Exception as e:
    print(f"Error during plotting: {e}")

print("\n=== Workflow Complete ===")