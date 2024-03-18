import os
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import matplotlib
from langchain.llms import OpenAI
import json
import warnings
import time
import seaborn as sns
import streamlit as st
import re
import io
import contextlib
import sys
import plotly.express as px
import altair as alt

# setting configs
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.simplefilter(action='ignore')

st.set_page_config(layout="wide", page_icon='⚡', page_title="Auto Analyst Pro")
alt.themes.enable("dark")

system_message = """
You are data analyst to help answer business questions by writing python code to analyze and draw business insights 
and create a report to be put in a powerpoint presentation.
You can study the data schema and decide which variables are required for the analysis and generate python code accordingly.
You can generate python code to plot graphs and perform analysis. You also know how to generate contents for a good 
professional looking business report by analysing the given data. The data will be provided in dataframe format.
The input dataframe will be named as "df". 
Whenever generating the python code, remember to replace the dataframe df with df_full.
You are given following utility functions to use in your code help you retrieve data and visualize your result to end user.
    1. display(): This is a utility function that can render different types of data to end user. 
        - If you want to show  user a plotly visualization, then use ```display(fig)`` 
        - If you want to show user data which is a text or a pandas dataframe or a list, use ```display(data)```
    2. print(): use print() if you need to observe data for yourself. 
Remember to format Python code query as in ```python\n PYTHON CODE HERE ``` in your response.
Any explanation about the data or the key metrics derived from the analysis to be put in presentation must be formatted 
in short and concise format suitable to be put in a powerpoint presentation and formatted like ```explaination\n EXPLAINATION FOR METRICS HERE ```
Only use display() to visualize or print result to user. Only use plotly for visualization.
Please follow the <<Template>> below:\n
"""

few_shot_examples = """
<<Template>>
Question: User Question
Thought: Understand the dataset.
Action: Study the dataset and understand what each variables mean and which variable will be most important for business 
case. List out the most important variables for analysis.

Observation: Now I have the variables that I want to analyse
Thought: Now I should start to analyse the dataset and explain the metrics obtained
Action: Python code to derive important metrics for explaining the dataset and explanation of the analysis
```python
import pandas as pd
analytics_df = df_full['important ']
# Fill missing data
analytics_df['Some_Column'] = analytics_df['Some_Column'].replace(np.nan, 0)
#use pandas, statistical analysis or machine learning to analyze data to answer  business question
analytics_df = analytics_df.apply(some_transformation)
analytics_metrics = metrics
chat_history = chat_history + str(metrics)
```
Observation: I have generated the metrics now I need to visualise the data 
Thought: Let's visualize the dataset and explain the analysis for presentation in short and concise way
Action:  
```python
import plotly.express as px
fig=px.line(analytics_df) # or use any suitable visualisation, you are a data analyst figure it out
#visualize fig object to user.  
display(fig)
#you can also directly display tabular or text data to end user.
display(analytics df)
```
NOW EXPLAIN THE ANALYSIS
```explaination
EXPLANATION OF ABOVE ANALYSIS IN TEXT FORMAT
```
<<Template>>
\n
"""


def build_agent(df):
    # function to build llm pandas agent
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613'),
        df, verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    return agent


def prompt_builder(user_prompt=None, chat_history=None):

    system_prompt = f"""System Prompt : {system_message + few_shot_examples}"""

    if user_prompt:
        system_prompt += f"""
        User Question : {user_prompt} \n
        """
    if chat_history:
        system_prompt += f"""\n{chat_history}\n"""

    return (system_prompt)