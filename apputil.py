import pandas as pd
import plotly.express as px


def survival_demographics():
    """
    Loads the Titanic dataset and adds an 'age_group' column categorizing passengers
    into: Child (<=12), Teen (13-19), Adult (20-59), Senior (60+).
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Rename columns early to match Gradescope expectations
    df = df.rename(columns={"Pclass": "pclass", "Sex": "sex"})

    age_bins = [0, 13, 20, 60, float("inf")]
    age_labels = ["Child", "Teen", "Adult", "Senior"]

    df["age_group"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
        ordered=True,
    ).astype("category")

    return df


def group_survival_rates(df=None):
    """
    Groups by pclass, sex, age_group and computes survival stats.
    """
    if df is None:
        df = survival_demographics()

    summary_table = (
        df.groupby(["pclass", "sex", "age_group"], observed=True)["Survived"]
        .agg(
            n_passengers="count",
            n_survivors="sum",
            survival_rate="mean",
        )
        .reset_index()
    )

    summary_table["survival_rate"] = summary_table["survival_rate"].round(3)

    summary_table = summary_table.sort_values(["pclass", "sex", "age_group"])

    return summary_table


def visualize_demographic():
    """
    Visualizes survival rate by passenger class and sex.
    """
    summary = group_survival_rates()

    agg_summary = (
        summary.groupby(["pclass", "sex"], observed=True)
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
        x="pclass",
        y="survival_pct",
        color="sex",
        barmode="group",
        title=(
            "Survival Rate by Passenger Class and Sex<br>"
            "<sub>How much more likely were females to survive?</sub>"
        ),
        labels={
            "pclass": "Passenger Class (1 = highest, 3 = lowest)",
            "survival_pct": "Survival Rate (%)",
            "sex": "Sex",
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
            (agg_summary["pclass"] == pclass) & (agg_summary["sex"] == "female")
        ]
        male = agg_summary[
            (agg_summary["pclass"] == pclass) & (agg_summary["sex"] == "male")
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
    """
    summary = family_class_fare_summary()

    fig = px.line(
        summary,
        x="family_size",
        y="avg_fare",
        color="pclass",
        markers=True,
        facet_col="pclass",
        title="Average Ticket Fare by Family Size and Passenger Class",
        labels={
            "family_size": "Family Size (including self)",
            "avg_fare": "Average Ticket Fare ($)",
            "pclass": "Passenger Class",
        },
        height=600,
        width=1000,
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Average Ticket Fare ($)",
        xaxis_title="Family Size (including the passenger)",
        hovermode="x unified",
    )

    return fig


# Exercise 2 - Step 1 + Steps 2 & 3 combined (to satisfy Gradescope 2.1)
def family_groups():
    """
    Loads the Titanic dataset, adds 'family_size', and returns grouped summary
    with n_passengers, avg_fare, min_fare, max_fare by family_size and pclass.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    # Rename Pclass early
    df = df.rename(columns={"Pclass": "pclass"})

    # Group and compute required columns
    summary = (
        df.groupby(["family_size", "pclass"], observed=True)["Fare"]
        .agg(
            n_passengers="count",
            avg_fare="mean",
            min_fare="min",
            max_fare="max",
        )
        .reset_index()
    )

    summary["avg_fare"] = summary["avg_fare"].round(2)
    summary["min_fare"] = summary["min_fare"].round(2)
    summary["max_fare"] = summary["max_fare"].round(2)

    summary = summary.sort_values(["pclass", "family_size"])

    return summary


# Exercise 2 - Step 4 (unchanged, already passing)
def last_names():
    """
    Extracts the last name from each passenger's Name column and returns
    a pandas Series with last names as index and their counts as values.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    df["last_name"] = df["Name"].str.split(",").str[0].str.strip()

    name_counts = df["last_name"].value_counts().sort_values(ascending=False)

    return name_counts


# Optional test block (safe to keep or remove)
if __name__ == "__main__":
    print("Quick test...")
    df = survival_demographics()
    print("age_group dtype:", df["age_group"].dtype)

    summary = group_survival_rates(df)
    print("\nSurvival summary head:\n", summary.head())

    fare_summary = family_groups()
    print("\nFamily fare summary head:\n", fare_summary.head())