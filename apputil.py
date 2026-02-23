import pandas as pd
import plotly.express as px


def survival_demographics():
    """
    Loads the Titanic dataset and adds an 'Age_Group' column categorizing passengers
    into: Child (<=12), Teen (13-19), Adult (20-59), Senior (60+).

    Returns:
        pd.DataFrame: The Titanic dataset with the new 'Age_Group' column added.
    """
    # Load Titanic dataset
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Define the age bins and corresponding labels
    age_bins = [0, 13, 20, 60, float("inf")]  # 0-12.999, 13-19.999, 20-59.999, 60+
    age_labels = ["Child", "Teen", "Adult", "Senior"]

    # Create the 'Age_Group' column using pd.cut()
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )

    return df


def group_survival_rates(df=None):
    """
    Groups the Titanic passengers by Pclass, Sex, and Age_Group.
    Computes count of passengers, number of survivors, and survival rate.

    Args:
        df (pd.DataFrame, optional): DataFrame with 'Age_Group' column.
                                     If None, calls survival_demographics() to get it.

    Returns:
        pd.DataFrame: Summary table with columns: Pclass, Sex, Age_Group,
                      n_passengers, n_survivors, survival_rate.
    """
    if df is None:
        df = survival_demographics()

    # Group and calculate metrics
    summary_table = (
        df.groupby(["Pclass", "Sex", "Age_Group"], observed=True)["Survived"]
        .agg(
            n_passengers="count",
            n_survivors="sum",
            survival_rate="mean",
        )
        .reset_index()
    )

    # Round survival_rate to 3 decimal places
    summary_table["survival_rate"] = summary_table["survival_rate"].round(3)

    # Sort for readability: class → sex → age group
    summary_table = summary_table.sort_values(["Pclass", "Sex", "Age_Group"])

    return summary_table


def visualize_demographic():
    """
    Visualizes survival rate by passenger class and sex to answer:
    "How much more likely were you to survive if you were female as opposed to male?"
    """
    # Get grouped data, but aggregate across age groups for this view
    summary = group_survival_rates()

    # Aggregate to class + sex only (average survival across age groups)
    agg_summary = (
        summary.groupby(["Pclass", "Sex"], observed=True)
        .agg(
            {
                "n_passengers": "sum",
                "n_survivors": "sum",
                "survival_rate": "mean",
            }
        )
        .reset_index()
    )

    agg_summary["survival_pct"] = (agg_summary["survival_rate"] * 100).round(1)

    fig = px.bar(
        agg_summary,
        x="Pclass",
        y="survival_pct",
        color="Sex",
        barmode="group",
        title=(
            "Survival Rate by Passenger Class and Sex<br>"
            "<sub>How much more likely were females to survive?</sub>"
        ),
        labels={
            "Pclass": "Passenger Class (1 = highest, 3 = lowest)",
            "survival_pct": "Survival Rate (%)",
            "Sex": "Sex",
        },
        color_discrete_map={"female": "#D81B60", "male": "#1E88E5"},
        height=600,
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        legend_title="Sex",
    )

    # Add ratio annotations
    for pclass in [1, 2, 3]:
        female = agg_summary[
            (agg_summary["Pclass"] == pclass) & (agg_summary["Sex"] == "female")
        ]
        male = agg_summary[
            (agg_summary["Pclass"] == pclass) & (agg_summary["Sex"] == "male")
        ]

        if not female.empty and not male.empty:
            f_rate = female["survival_pct"].values[0]
            m_rate = male["survival_pct"].values[0]

            if m_rate > 0:
                ratio = round(f_rate / m_rate, 1)
                text = f"{ratio}x more likely"
            else:
                text = "Only females survived"

            fig.add_annotation(
                x=pclass,
                y=max(f_rate, m_rate) + 5,
                text=text,
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )

    return fig


def visualize_families():
    """
    Visualizes how family size affects average ticket fare within each passenger class.
    Answers: "How does family size affect average ticket fare within each passenger class?"
    
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    # Get the summary table from Step 2/3
    summary = family_class_fare_summary()

    fig = px.line(
        summary,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        facet_col="Pclass",
        title="Average Ticket Fare by Family Size and Passenger Class",
        labels={
            "family_size": "Family Size (including self)",
            "avg_fare": "Average Ticket Fare ($)",
            "Pclass": "Passenger Class",
        },
        height=600,
        width=1000,
    )

    fig.update_layout(
        showlegend=False,  # hide legend since facets show class
        yaxis_title="Average Ticket Fare ($)",
        xaxis_title="Family Size (including the passenger)",
        hovermode="x unified",
    )

    return fig


# Exercise 2
def family_groups():
    """
    Loads the Titanic dataset and adds a 'family_size' column.
    
    family_size = number of siblings/spouses aboard (SibSp)
                + number of parents/children aboard (Parch)
                + 1 (the passenger themselves)
    
    Returns:
        pd.DataFrame: The Titanic dataset with the new 'family_size' column added.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Create family_size column
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    return df


def family_class_fare_summary(df=None):
    """
    Groups passengers by family_size and Pclass.
    Calculates:
    - n_passengers: total number of passengers in the group
    - avg_fare: average ticket fare
    - min_fare: minimum ticket fare in the group
    - max_fare: maximum ticket fare in the group
    
    Returns:
        pd.DataFrame: Summary table with the requested statistics.
    """
    if df is None:
        df = family_groups()

    summary = (
        df.groupby(["family_size", "Pclass"], observed=True)["Fare"]
        .agg(
            n_passengers="count",
            avg_fare="mean",
            min_fare="min",
            max_fare="max",
        )
        .reset_index()
    )

    # Round monetary values for readability
    summary["avg_fare"] = summary["avg_fare"].round(2)
    summary["min_fare"] = summary["min_fare"].round(2)
    summary["max_fare"] = summary["max_fare"].round(2)

    # Sort: first by passenger class (1–3), then by family size
    summary = summary.sort_values(["Pclass", "family_size"])

    return summary


def last_names():
    """
    Extracts the last name from each passenger's Name column and returns
    a pandas Series with last names as index and their counts as values.
    
    Returns:
        pd.Series: Last name counts (sorted descending by frequency).
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Extract last name: everything before the comma
    df["last_name"] = df["Name"].str.split(",").str[0].str.strip()

    # Get counts, sorted descending
    name_counts = df["last_name"].value_counts().sort_values(ascending=False)

    return name_counts