import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load model and preprocessor
priority_model = joblib.load('priority_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load task data
df = pd.read_csv('tasks.csv')
df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce')

# Add overdue flag and progress
df['Overdue'] = df['Deadline'] < pd.to_datetime('today')
progress_map = {'Completed': 100, 'In Progress': 50, 'To Do': 0}
df['Progress'] = df['Status'].map(progress_map).fillna(0)

# Page config
st.set_page_config(page_title="Task Manager", layout="wide")
st.title("📋 Task Manager Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Filter Tasks")
status_filter = st.sidebar.multiselect("Status", options=df['Status'].unique(), default=df['Status'].unique())
priority_filter = st.sidebar.multiselect("Priority", options=df['Priority'].unique(), default=df['Priority'].unique())
assignee_filter = st.sidebar.multiselect("Assignee", options=df['Assignee'].unique(), default=df['Assignee'].unique())

filtered_df = df[
    (df['Status'].isin(status_filter)) &
    (df['Priority'].isin(priority_filter)) &
    (df['Assignee'].isin(assignee_filter))
]

# --- Tasks Table ---
st.header("🗂️ Tasks List")

def color_priority(val):
    color = ''
    if val == 'High':
        color = '#ff9999'  # red
    elif val == 'Medium':
        color = '#ffcc80'  # yellow
    elif val == 'Low':
        color = '#b3ffb3'  # green
    return f'background-color: {color}; color: black;'

styled_df = filtered_df.style.applymap(color_priority, subset=['Priority'])
st.dataframe(styled_df, height=400)

# --- Add New Task ---
st.header("➕ Add New Task")

with st.form("add_task_form"):
    new_id = st.text_input("Task ID")
    new_name = st.text_input("Task Name")
    new_assignee = st.selectbox("Assignee", df['Assignee'].unique())
    new_priority = st.selectbox("Priority", ['High', 'Medium', 'Low'])
    new_status = st.selectbox("Status", ['To Do', 'In Progress', 'Completed'])
    new_time = st.number_input("Time Taken (hrs)", min_value=1, max_value=100)
    new_deadline = st.date_input("Deadline")
    add_submit = st.form_submit_button("Add Task")

    if add_submit:
        new_task = pd.DataFrame([{
            'Task ID': new_id,
            'Task Name': new_name,
            'Assignee': new_assignee,
            'Priority': new_priority,
            'Status': new_status,
            'Time Taken (hrs)': new_time,
            'Deadline': pd.to_datetime(new_deadline),
        }])
        new_task['Overdue'] = new_task['Deadline'] < pd.to_datetime('today')
        new_task['Progress'] = new_task['Status'].map(progress_map).fillna(0)
        df = pd.concat([df, new_task], ignore_index=True)
        df.to_csv("tasks.csv", index=False)
        st.success("Task added successfully!")

# --- Delete Task ---
st.header("🗑️ Delete Task")

with st.form("delete_task_form"):
    del_id = st.text_input("Enter Task ID to delete")
    del_submit = st.form_submit_button("Delete Task")
    if del_submit:
        if del_id.strip().lower() in df['Task ID'].astype(str).str.lower().str.strip().values:
            df = df[~df['Task ID'].astype(str).str.lower().str.strip().eq(del_id.strip().lower())]
            df.to_csv("tasks.csv", index=False)
            st.success(f"Task with ID '{del_id}' deleted.")
        else:
            st.error("Task ID not found.")

# --- Task Priority Prediction ---
st.header("🔮 Predict Task Priority")

with st.form("priority_form"):
    time_taken = st.number_input("Time Taken (hours)", min_value=1, max_value=100, value=1)
    status = st.selectbox("Status", ['To Do', 'In Progress', 'Completed'])
    days_until_deadline = st.number_input("Days Until Deadline", min_value=-180, max_value=180, value=10)
    submitted = st.form_submit_button("Predict Priority")

    if submitted:
        input_df = pd.DataFrame({
            'Time Taken (hrs)': [time_taken],
            'Status': [status],
            'Days Until Deadline': [days_until_deadline]
        })
        X_input = preprocessor.transform(input_df)
        pred = priority_model.predict(X_input)

        if hasattr(priority_model, "predict_proba"):
            probs = priority_model.predict_proba(X_input)[0]
            classes = priority_model.classes_
            st.success(f"Predicted Priority: {pred[0]}")
            st.write("Prediction Confidence:")
            for cls, prob in zip(classes, probs):
                st.write(f"- {cls}: {prob*100:.1f}%")
        else:
            st.success(f"Predicted Priority: {pred[0]}")

# --- Summary KPIs ---
st.header("📊 Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tasks", len(df))
col2.metric("Filtered Tasks", len(filtered_df))
col3.metric("Overdue Tasks", df['Overdue'].sum())
col4.metric("Avg. Progress", f"{filtered_df['Progress'].mean():.1f}%")

# --- Visualizations ---
st.header("📈 Visual Insights")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Priority Distribution")
    priority_order = ['High', 'Medium', 'Low']
    priority_counts = filtered_df['Priority'].value_counts().reindex(priority_order).fillna(0)
    sns.barplot(x=priority_counts.index, y=priority_counts.values, palette=['#ff4c4c', '#ffa500', '#4caf50'])
    plt.ylabel("Number of Tasks")
    st.pyplot(plt.gcf())
    plt.clf()

with col6:
    st.subheader("Status Breakdown")
    status_counts = filtered_df['Status'].value_counts()
    plt.pie(status_counts.values, labels=status_counts.index,
            autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    st.pyplot(plt.gcf())
    plt.clf()

# --- Calendar View ---
st.header("🗓️ Calendar View")
selected_date = st.date_input("Select a Date", value=datetime.today())
tasks_on_date = df[df['Deadline'].dt.date == selected_date]

if tasks_on_date.empty:
    st.info("No tasks due on this date.")
else:
    st.write(f"Tasks due on {selected_date}:")
    st.dataframe(tasks_on_date[['Task ID', 'Task Name', 'Assignee', 'Priority', 'Status', 'Deadline']])

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Infotact Group 5** | Powered by Streamlit & scikit-learn")

