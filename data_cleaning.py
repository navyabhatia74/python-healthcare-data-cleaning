import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === File paths ===
RAW_CSV = "KaggleV2-May-2016.csv"
OUTPUT_DIR = "output"
CLEANED_CSV = os.path.join(OUTPUT_DIR, "cleaned_healthcare_data.csv")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# === Plot style ===
plt.switch_backend("Agg")
sns.set_style("whitegrid")


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    return pd.read_csv(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and fix obvious spelling issues."""
    clean_names = {
        'PatientId': 'patient_id',
        'AppointmentID': 'appointment_id',
        'Gender': 'gender',
        'ScheduledDay': 'scheduled_day',
        'AppointmentDay': 'appointment_day',
        'Age': 'age',
        'Neighbourhood': 'neighborhood',
        'Scholarship': 'scholarship',
        'Hipertension': 'hypertension',
        'Diabetes': 'diabetes',
        'Alcoholism': 'alcoholism',
        'Handcap': 'handicap',
        'SMS_received': 'sms_received',
        'No-show': 'no_show',
    }
    return df.rename(columns=clean_names)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to improve analysis and business insight."""
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'], utc=True)
    df['appointment_day'] = pd.to_datetime(df['appointment_day'], utc=True)

    age_bins = [-1, 12, 20, 40, 60, 150]
    age_labels = ['child', 'teen', 'adult', 'middle_age', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    df['age_group'] = df['age_group'].cat.add_categories('unknown').fillna('unknown')

    df['appointment_weekday'] = df['appointment_day'].dt.day_name()
    df['appointment_month'] = df['appointment_day'].dt.month_name()

    df['waiting_days'] = (
        df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()
    ).dt.days

    df['attendance_category'] = df['no_show'].map({'No': 'attended', 'Yes': 'missed'})
    return df


def save_cleaned_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset to {path}")


def ensure_plots_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plot(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_attendance_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df['attendance_category'].value_counts().index
    sns.countplot(data=df, x='attendance_category', order=order, color='steelblue', ax=ax)
    ax.set_title('Attendance Distribution')
    ax.set_xlabel('Attendance Category')
    ax.set_ylabel('Number of Appointments')
    save_plot(fig, 'attendance_distribution.png')

    counts = df['attendance_category'].value_counts(normalize=True).mul(100).round(2)
    print('\nAttendance distribution (%):')
    print(counts)
    print('Insight: this shows the overall proportion of attended versus missed appointments.')


def plot_attendance_by_gender(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x='gender',
        hue='attendance_category',
        palette='magma',
        ax=ax,
        order=sorted(df['gender'].unique()),
    )
    ax.set_title('Attendance by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Number of Appointments')
    save_plot(fig, 'attendance_by_gender.png')

    pivot = df.pivot_table(
        index='gender',
        columns='attendance_category',
        values='appointment_id',
        aggfunc='count',
        fill_value=0,
    )
    print('\nAttendance counts by gender:')
    print(pivot)
    print('Insight: comparing attendance across gender can reveal if one group misses more appointments.')


def plot_attendance_by_age_group(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    age_order = ['child', 'teen', 'adult', 'middle_age', 'senior', 'unknown']
    sns.countplot(
        data=df,
        x='age_group',
        hue='attendance_category',
        order=age_order,
        palette='rocket',
        ax=ax,
    )
    ax.set_title('Attendance by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Number of Appointments')
    save_plot(fig, 'attendance_by_age_group.png')

    percent = df.groupby(['age_group', 'attendance_category']).size().unstack(fill_value=0)
    percent = percent.div(percent.sum(axis=1), axis=0).mul(100).round(2)
    print('\nAttendance percentage by age group:')
    print(percent)
    print('Insight: age groups help identify which age brackets have higher no-show rates.')


def plot_weekday_trends(df: pd.DataFrame) -> None:
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df,
        x='appointment_weekday',
        order=weekday_order,
        color='slateblue',
        ax=ax,
    )
    ax.set_title('Appointment Volume by Weekday')
    ax.set_xlabel('Weekday')
    ax.set_ylabel('Number of Appointments')
    save_plot(fig, 'weekday_appointment_trends.png')

    weekday_rate = df.groupby('appointment_weekday')['appointment_id'].count().reindex(weekday_order)
    print('\nAppointment volume by weekday:')
    print(weekday_rate)
    print('Insight: peak weekdays can help plan staffing and reminder campaigns.')


def plot_waiting_time_vs_attendance(df: pd.DataFrame) -> None:
    # Create waiting time buckets to make the trend easy to read.
    bins = [-999, -1, 0, 3, 7, 14, 30, 999]
    labels = ['negative', 'same_day', '1-3_days', '4-7_days', '8-14_days', '15-30_days', '31+_days']
    df['waiting_bucket'] = pd.cut(df['waiting_days'], bins=bins, labels=labels, ordered=True)

    bucket_counts = df.groupby(['waiting_bucket', 'attendance_category'])['appointment_id'].count().unstack(fill_value=0)
    bucket_rates = bucket_counts.div(bucket_counts.sum(axis=1), axis=0).mul(100).round(1).reset_index()
    melted = bucket_rates.melt(id_vars='waiting_bucket', value_vars=['attended', 'missed'], var_name='attendance_category', value_name='percent')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x='waiting_bucket',
        y='percent',
        hue='attendance_category',
        palette=['seagreen', 'indianred'],
        ax=ax,
    )
    ax.set_title('Attendance Percentage by Waiting Time Bucket')
    ax.set_xlabel('Waiting Time Bucket')
    ax.set_ylabel('Percentage of Appointments')
    ax.set_ylim(0, 100)
    save_plot(fig, 'waiting_days_vs_attendance.png')

    pivot = bucket_rates.set_index('waiting_bucket')
    print('\nAttendance percentage by waiting bucket:')
    print(pivot)
    print('Insight: this plot shows whether short or long wait times are linked to higher attendance or no-show rates.')


def main() -> None:
    print('Loading dataset...')
    df = load_data(RAW_CSV)

    print('Cleaning column names...')
    df = clean_columns(df)

    print('Adding engineered features...')
    df = add_features(df)

    ensure_plots_dir(PLOTS_DIR)
    save_cleaned_data(df, CLEANED_CSV)

    print('\nStarting exploratory data analysis...')
    plot_attendance_distribution(df)
    plot_attendance_by_gender(df)
    plot_attendance_by_age_group(df)
    plot_weekday_trends(df)
    plot_waiting_time_vs_attendance(df)

    print('\nExploratory data analysis complete.')


if __name__ == '__main__':
    main()
