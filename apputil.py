import pandas as pd
import plotly.express as px

# ----------------------------
# Exercise 1: Survival Demographics
# ----------------------------
def survival_demographics():
    """
    Loads Titanic dataset, adds 'age_group', and returns a grouped summary table
    by pclass, sex, and age_group including n_passengers, n_survivors, and survival_rate.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    # Drop rows with missing Age
    df = df.dropna(subset=["Age"])

    # Add age_group
    age_bins = [0, 12, 19, 59, float("inf")]
    age_labels = ["Child", "Teen", "Adult", "Senior"]
    df["age_group"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True,
        ordered=True
    ).astype("category")

    # Aggregate existing groups
    summary = df.groupby(["Pclass", "Sex", "age_group"])["Survived"].agg(
        n_passengers="count",
        n_survivors="sum"
    )

    # Ensure all combinations exist
    pclass_vals = [1, 2, 3]
    sex_vals = ["male", "female"]
    age_group_vals = pd.Categorical(age_labels, categories=age_labels, ordered=True)

    full_index = pd.MultiIndex.from_product(
        [pclass_vals, sex_vals, age_group_vals],
        names=["Pclass", "Sex", "age_group"]
    )

    summary = summary.reindex(full_index, fill_value=0)

    # Add survival_rate
    summary["n_passengers"] = summary["n_passengers"].astype("int64")
    summary["n_survivors"] = summary["n_survivors"].astype("int64")
    summary["survival_rate"] = (summary["n_survivors"] / summary["n_passengers"].replace(0, 1)).round(3)
    summary.loc[summary["n_passengers"] == 0, "survival_rate"] = 0.0

    # Reset index and rename
    summary = summary.reset_index()
    summary = summary.rename(columns={"Pclass": "pclass", "Sex": "sex"})
    summary = summary.sort_values(["pclass", "sex", "age_group"])

    return summary


def group_survival_rates(df=None):
    """
    Returns the Titanic summary table grouped by pclass, sex, and age_group.
    """
    return survival_demographics()


# ----------------------------
# Exercise 2: Family Size and Fare
# ----------------------------
def family_groups():
    """
    Loads Titanic dataset, adds 'family_size', and returns summary grouped by family_size
    and pclass with fare statistics: n_passengers, avg_fare, min_fare, max_fare.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

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
    summary[["avg_fare", "min_fare", "max_fare"]] = summary[["avg_fare", "min_fare", "max_fare"]].round(2)
    summary = summary.sort_values(["pclass", "family_size"])

    return summary


def family_class_fare_summary(df=None):
    """
    Wrapper that returns the summary of family size vs fare.
    """
    if df is None:
        df = family_groups()
    return df


def last_names():
    """
    Extracts last names from the Name column and returns a Series with counts.
    """
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
    df = pd.read_csv(url)

    df["last_name"] = df["Name"].str.split(",").str[0].str.strip()
    name_counts = df["last_name"].value_counts().sort_values(ascending=False)

    return name_counts


# ----------------------------
# Optional Visualization Functions
# ----------------------------
def visualize_demographic():
    summary = group_survival_rates()
    agg_summary = (
        summary.groupby(["pclass", "sex"], observed=True)
        .agg(
            n_passengers=("n_passengers", "sum"),
            n_survivors=("n_survivors", "sum"),
            survival_rate=("survival_rate", "mean"),
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
        title="Survival Rate by Passenger Class and Sex",
        labels={"pclass": "Passenger Class", "survival_pct": "Survival Rate (%)", "sex": "Sex"},
        color_discrete_map={"female": "#D81B60", "male": "#1E88E5"},
        height=600,
    )

    for pclass in [1, 2, 3]:
        female = agg_summary[(agg_summary["pclass"] == pclass) & (agg_summary["sex"] == "female")]
        male = agg_summary[(agg_summary["pclass"] == pclass) & (agg_summary["sex"] == "male")]
        if not female.empty and not male.empty:
            f_rate = female["survival_pct"].values[0]
            m_rate = male["survival_pct"].values[0]
            text = f"{round(f_rate / m_rate, 1)}x more likely" if m_rate > 0 else "Only females survived"
            fig.add_annotation(x=pclass, y=max(f_rate, m_rate)+5, text=text, showarrow=True, arrowhead=1,
                               ax=0, ay=-30, font=dict(size=12), bgcolor="white", bordercolor="gray", borderwidth=1)

    return fig


def visualize_families():
    summary = family_class_fare_summary()

    fig = px.line(
        summary,
        x="family_size",
        y="avg_fare",
        color="pclass",
        markers=True,
        facet_col="pclass",
        title="Average Ticket Fare by Family Size and Passenger Class",
        labels={"family_size": "Family Size", "avg_fare": "Average Ticket Fare ($)", "pclass": "Passenger Class"},
        height=600,
        width=1000,
    )

    fig.update_layout(showlegend=False, hovermode="x unified")
    return fig