#utility.py
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
def extract_combined_school_info(df):
    info = {}
    for _, row in df.iterrows():
        location = row['CITY'].strip().capitalize()  # Assuming 'City' column
        school_name = row['NAME'].strip()  # Assuming 'Name' column
        school_type = row['Type']  # Public or Private

        # Combine school type and name
        school_entry = f"{school_name} ({school_type})"

        if location in info:
            info[location].append(school_entry)
        else:
            info[location] = [school_entry]
    return info


