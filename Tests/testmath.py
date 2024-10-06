import re
import sympy as sp
import matplotlib.pyplot as plt
import base64
import io
import numpy as np

def process_question(question):
    # Match plotting graphs
    match = re.match(r'plot\s*(.+)', question.lower())
    if match:
        expression = match.groups()[0]
        x = sp.symbols('x')
        expr = sp.sympify(expression)

        # Create a range of x values
        x_vals = np.linspace(-10, 10, 400)
        y_vals = [expr.subs(x, val) for val in x_vals]  # Evaluate the expression

        # Create the plot using Matplotlib
        plt.figure()
        plt.plot(x_vals, y_vals)
        plt.title(f'Plot of {expression}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the plot to buf
        plt.close()  # Close the plot to free resources
        
        # Encode the plot as a base64 string
        buf.seek(0)
        encoded_string = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{encoded_string}" />'
    
    return None

# Example usage
question = "plot x**2"
result = process_question(question)
print(result)
