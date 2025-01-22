from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import re
import random
import sympy as sp
import matplotlib.pyplot as plt
import base64
import io
import numpy as np
import os
import pandas as pd
# import fireducks.pandas as pd

#Get math questions:
def handle_math_question(question):
    # Match basic arithmetic operations
    match = re.match(r'(\d+)\s*([+\-*/])\s*(\d+)', question.lower())
    if match:
        try:
            num1, operator, num2 = match.groups()
            num1, num2 = int(num1), int(num2)
            if operator == '+':
                return num1 + num2
            elif operator == '-':
                return num1 - num2
            elif operator == '*':
                return num1 * num2
            elif operator == '/':
                return num1 / num2
        except Exception as e:
            return f"Error in arithmetic operation: {str(e)}"
    
    
    # Match exponentiation
    match = re.match(r'(\d+)\s*\^\s*(\d+)', question.lower())
    if match:
        try:
            base, exponent = match.groups()
            return int(base) ** int(exponent)
        except Exception as e:
            return f"Error in exponentiation: {str(e)}"
    
    # Match logarithms
    match = re.match(r'log\s*\((\d+)\)', question.lower())
    if match:
        try:
            value = match.groups()[0]
            return sp.log(int(value))
        except Exception as e:
            return f"Error in logarithm calculation: {str(e)}"
    
    # Match differentiation
    match = re.match(r'differentiate\s*(.+)', question.lower())
    if match:
        try:
            expression = match.groups()[0]
            x = sp.symbols('x')
            expr = sp.sympify(expression)
            return sp.diff(expr, x)
        except (sp.SympifyError, TypeError) as e:
            return f"Error in differentiation: {str(e)}"
    
    
    # Match integration
    match = re.match(r'integrate\s*(.+)', question.lower())
    if match:
        try:
            expression = match.groups()[0]
            x = sp.symbols('x')
            expr = sp.sympify(expression)
            return sp.integrate(expr, x)
        except (sp.SympifyError, TypeError) as e:
            return f"Error in integration: {str(e)}"
    
    
    # Match matrix operations
    match = re.match(r'matrix\s*(.+)', question.lower())
    if match:
        try:
            expression = match.groups()[0]
            matrix = sp.Matrix(sp.sympify(expression))
            return matrix
        except (sp.SympifyError, TypeError) as e:
            return f"Error in matrix operation: {str(e)}"
    
    
    # Match solving linear equations
    match = re.match(r'solve\s*linear\s*equations\s*(.+)', question.lower())
    if match:
        try:
            equations = match.groups()[0]
            eqs = [sp.sympify(eq) for eq in equations.split(',')]
            symbols = list(eqs[0].free_symbols)
            solution = sp.linsolve(eqs, *symbols)
            return solution
        except (sp.SympifyError, TypeError) as e:
            return f"Error in solving linear equations: {str(e)}"
    
    # Match solving differential equations
    match = re.match(r'solve\s*differential\s*equation\s*(.+)', question.lower())
    if match:
        try:
            equation = match.groups()[0]
            x = sp.symbols('x')
            f = sp.Function('f')
            eq = sp.sympify(equation)
            solution = sp.dsolve(eq, f(x))
            return solution
        except (sp.SympifyError, TypeError) as e:
            return f"Error in solving differential equation: {str(e)}"
    
    
    # Match statistical operations
    match = re.match(r'statistics\s*(.+)', question.lower())
    if match:
        try:
            data = [int(num) for num in match.groups()[0].split(',')]
            mean = sp.stats.mean(data)
            variance = sp.stats.variance(data)
            return f"Mean: {mean}, Variance: {variance}"
        except Exception as e:
            return f"Error in statistical operation: {str(e)}"
    
    # Match plotting graphs
    match = re.match(r'plot\s*(.+)', question.lower())
    if match:
        expression = match.groups()[0]
        x = sp.symbols('x')
        try:
            expr = sp.sympify(expression)
        except (sp.SympifyError, TypeError):
            return "Invalid mathematical expression for plotting."

        try:
            # Create a range of x values
            x_vals = np.linspace(-100, 100, 400)
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
        except Exception as e:
            return f"Error generating plot: {str(e)}"
    


    return None
    # if match:
    #     expression = match.groups()[0]
    #     x = sp.symbols('x')
    #     expr = sp.sympify(expression)
    #     sp.plot(expr, (x, -10, 10), show = False)
    #     plt.savefig('plot.png')
    #     # plt.pause(5)
    #     plt.close()
    #     with open('plot.png', 'rb') as image_file:
    #         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    #     return f'<img src="data:image/png;base64,{encoded_string}" />'
    # return "No plot command found."

    # TOO MUCH RUNTIME
    # match = re.match(r'plot\s*(.+)', question.lower())
    # if match:
    #     expression = match.groups()[0]
    #     x = sp.symbols('x')
    #     expr = sp.sympify(expression)
    #     plot = sp.plot(expr, (x, -10, 10), show=False)
        
    #     # Save the plot to a BytesIO object
    #     buf = io.BytesIO()
    #     plot._backend.fig.savefig(buf, format='png')
    #     plt.close(plot._backend.fig)
        
    #     # Encode the plot as a base64 string
    #     buf.seek(0)
    #     encoded_string = base64.b64encode(buf.read()).decode('utf-8')
    #     return f'<img src="data:image/png;base64,{encoded_string}" />'
    
    # return None
