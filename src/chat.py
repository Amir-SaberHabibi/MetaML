import streamlit as st
from groq import Groq
import pandas as pd
import os

groq_api_key = "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"
client = Groq(api_key=groq_api_key)

models = ["llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"]
model = models[0]

def get_groq_completion(messages, model):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            response += delta_content

    return response

st.title("Data Analyzer Chatbot")
st.write(" ")


results_dir = "/mount/src/metaml/src/results"
file_path_pso = os.path.join(results_dir, "best_result_pso.csv")
df_pso = pd.read_csv(file_path_pso)
file_path_bp = os.path.join(results_dir, "best_result_bp.csv")
df_bp = pd.read_csv(file_path_bp)
file_path_ga = os.path.join(results_dir, "best_result_ga.csv")
df_ga = pd.read_csv(file_path_ga)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": f"You are an expert helpful assistant in data analysis. Your duty is analyzing the tables that are provided for you, related to the performance of different algorithms (PSO, BP, GA). Wait for user's instructions and return valid responses. Try to be straight-forward, accurate and, be nice and heart-warming."}
    ]

with st.expander(f"Powered by **{model}**", expanded=True):
    if prompt := st.chat_input("Ask the algorithms!"):
        st.session_state.messages.append({"role": "user", "content": prompt + f" \nThe tables that you have to analyze are these: \n PSO Algorithm Results: \n{df_pso}\n\n BP Algorithm Results: \n{df_bp}\n\n GA Algorithm Results: \n{df_ga}\n\n"})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking.."):

            with st.chat_message("assistant"):
                response = get_groq_completion(st.session_state.messages, model)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
