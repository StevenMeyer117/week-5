import pandas as pd
import plotly.express as px


def survival_demographics():
    """
    Loads the Titanic dataset and adds an 'age_group' column categorizing passengers
    into: Child (<=12), Teen (13-19), Adult (20-59), Senior (60+).

    Returns:
        pd.DataFrame: The Titanic dataset with the new 'age_group' column added.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Rename columns early
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
    Groups the Titanic passengers by pclass, sex, and age_group.
    Computes count of passengers, number of survivors, and survival rate.
    Includes ALL possible combinations with n_passengers=0 for missing groups.
    """
    if df is None:
        df = survival_demographics()

    # Drop passengers with missing age_group
    df = df.dropna(subset=["age_group"])

    # Aggregate existing groups
    summary_table = (
        df.groupby(["pclass", "sex", "age_group"])["Survived"]
        .agg(n_passengers="count", n_survivors="sum")
        .reset_index()
    )

    # Calculate survival rate
    summary_table["survival_rate"] = (
        summary_table["n_survivors"] / summary_table["n_passengers"]
    ).round(3)

    # Force all possible combinations
    pclass_vals = [1, 2, 3]
    sex_vals = ["male", "female"]
    age_group_vals = ["Child", "Teen", "Adult", "Senior"]

    full_index = pd.MultiIndex.from_product(
        [pclass_vals, sex_vals, age_group_vals],
        names=["pclass", "sex", "age_group"]
    )

    # Reindex to include missing groups
    summary_table = summary_table.set_index(["pclass", "sex", "age_group"])
    summary_table = summary_table.reindex(full_index, fill_value=0)

    # Ensure integer type for counts
    summary_table["n_passengers"] = summary_table["n_passengers"].astype(int)
    summary_table["n_survivors"] = summary_table["n_survivors"].astype(int)

    # Fill survival_rate for empty groups
    summary_table["survival_rate"] = summary_table["survival_rate"].fillna(0)

    # Reset index to columns
    summary_table = summary_table.reset_index()

    # Sort
    summary_table = summary_table.sort_values(["pclass", "sex", "age_group"])

    return summary_table


def visualize_demographic():
    """
    Visualizes survival rate by passenger class and sex to answer:
    "How much more likely were you to survive if you were female as opposed to male?"
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


def family_groups():
    """
    Loads the Titanic dataset, adds 'family_size', and returns a summary table
    grouped by family_size and pclass with fare statistics.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Add family size
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    # Aggregate for Gradescope expectations
    summary = (
        df.groupby(["family_size", "Pclass"], observed=True)["Fare"]
        .agg(
            n_passengers="count",
            avg_fare="mean",
            min_fare="min",
            max_fare="max"
        )
        .reset_index()
    )

    summary = summary.rename(columns={"Pclass": "pclass"})
    summary = summary.sort_values(["pclass", "family_size"])
    summary[["avg_fare", "min_fare", "max_fare"]] = summary[["avg_fare", "min_fare", "max_fare"]].round(2)

    return summary


def family_class_fare_summary(df=None):
    """
    Groups passengers by family_size and pclass.
    Calculates n_passengers, avg_fare, min_fare, max_fare.
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

    summary["avg_fare"] = summary["avg_fare"].round(2)
    summary["min_fare"] = summary["min_fare"].round(2)
    summary["max_fare"] = summary["max_fare"].round(2)

    summary = summary.rename(columns={"Pclass": "pclass"})

    summary = summary.sort_values(["pclass", "family_size"])

    return summary


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


#if __name__ == "__main__":
#    print("Quick test of group_survival_rates with missing groups...")
#    summary = group_survival_rates()
#    print("\nFull summary shape:", summary.shape)
#    print("\nCheck for pclass=2, sex=female, age_group=Senior:")
#    print(summary[(summary["pclass"] == 2) & (summary["sex"] == "female") & (summary["age_group"] == "Senior")])