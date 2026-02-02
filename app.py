import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Mentoring Visit Dashboard", layout="wide")

# Your raw files have these exact columns (verified):
# District: District Name, Employee Code, Employee Name, Role Name, Targeted Visits, Completed, Details
# Block:    District Name, Block Name, Employee Code, Employee Name, Role Name, Targeted Visits, Completed, Details
# Cluster:  District Name, Block Name, Cluster Name, Employee Code, Employee Name, Role Name, Targeted Visits, Completed, Details

REQ_DIST = {"District Name", "Employee Code", "Employee Name", "Role Name", "Targeted Visits", "Completed"}
REQ_BLOCK = {"District Name", "Block Name", "Employee Code", "Employee Name", "Role Name", "Targeted Visits", "Completed"}
REQ_CLUSTER = {"District Name", "Block Name", "Cluster Name", "Employee Code", "Employee Name", "Role Name", "Targeted Visits", "Completed"}

DIST_ROLES = ["DPC", "DIET Principal", "DIET Academic", "APC"]
BLOCK_ROLES = ["BRCC", "BAC"]
CLUSTER_ROLES = ["CAC"]

ROLE_CANON = {
    "BRC": "BRCC",  # mapping if your block file uses BRC
    "BRCC": "BRCC",
    "BAC": "BAC",
    "CAC": "CAC",
    "DPC": "DPC",
    "APC": "APC",
    "DIET Principal": "DIET Principal",
    "DIET Academic": "DIET Academic",
}

# ============================================================
# HELPERS
# ============================================================
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Clean strings
    text_cols = ["District Name", "Block Name", "Cluster Name", "Employee Name", "Role Name", "Employee Code"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "None", "NaN", ""]), c] = np.nan

    # Numeric coercion
    for c in ["Targeted Visits", "Completed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Canonical roles
    if "Role Name" in df.columns:
        df["Role Name"] = df["Role Name"].map(lambda x: ROLE_CANON.get(str(x).strip(), str(x).strip()))

    return df


def validate(df: pd.DataFrame, required: set, label: str):
    missing = required - set(df.columns)
    if missing:
        st.error(f"{label} missing columns: {sorted(list(missing))}")
        st.stop()


def mentor_100_flag(df: pd.DataFrame) -> pd.Series:
    # A mentor counts as 100% if Completed >= Targeted and Targeted > 0
    return (df["Targeted Visits"] > 0) & (df["Completed"] >= df["Targeted Visits"])


def safe_pct(numer, denom):
    if denom <= 0:
        return np.nan
    return (numer / denom) * 100


def style_pct_cols(df: pd.DataFrame, pct_cols: list[str]):
    d = df.copy()
    for c in pct_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(1)
    return d


def agg_role_wide(df: pd.DataFrame, group_cols: list[str], role: str) -> pd.DataFrame:
    """
    For state table: gives role Target/Completed/% by group_cols.
    """
    d = df[df["Role Name"] == role].copy()
    if d.empty:
        # return empty structure so merge doesn't break
        out_cols = group_cols + [f"{role} Target", f"{role} Completed", f"{role} %"]
        return pd.DataFrame(columns=out_cols)

    g = d.groupby(group_cols, dropna=False).agg(
        Target=("Targeted Visits", "sum"),
        Completed=("Completed", "sum")
    ).reset_index()

    g[f"{role} Target"] = g["Target"]
    g[f"{role} Completed"] = g["Completed"]
    g[f"{role} %"] = np.where(g["Target"] > 0, (g["Completed"] / g["Target"]) * 100, np.nan)

    return g[group_cols + [f"{role} Target", f"{role} Completed", f"{role} %"]]


def overall_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["is100"] = mentor_100_flag(d)

    g = d.groupby(group_cols, dropna=False).agg(
        Total_Target=("Targeted Visits", "sum"),
        Total_Completed=("Completed", "sum"),
        Mentors=("Employee Code", "nunique"),
        Mentors100=("is100", "sum"),
    ).reset_index()

    g["Total %"] = np.where(g["Total_Target"] > 0, (g["Total_Completed"] / g["Total_Target"]) * 100, np.nan)
    g["% of Mentors Completing 100% Mentoring Visit"] = np.where(
        g["Mentors"] > 0, (g["Mentors100"] / g["Mentors"]) * 100, np.nan
    )
    return g


def to_excel_download(sheets: dict[str, pd.DataFrame], filename: str):
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    buff.seek(0)
    st.download_button(
        "⬇️ Download tables as Excel",
        data=buff,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# UI: SIDEBAR UPLOADS
# ============================================================
st.title("📊 Mentoring Visit Dashboard")
st.caption("State → District → Block drill-down | sorted performance tables | cadre-wise compliance")

with st.sidebar:
    st.header("Upload raw files (CSV)")
    f_dist = st.file_uploader("District file", type=["csv"])
    f_block = st.file_uploader("Block file", type=["csv"])
    f_cluster = st.file_uploader("Cluster file", type=["csv"])

    st.divider()
    st.header("Options")
    show_preview = st.checkbox("Show file previews", value=False)
    enable_export = st.checkbox("Enable Excel export", value=True)

if not (f_dist and f_block and f_cluster):
    st.info("Upload all 3 files: District, Block and Cluster.")
    st.stop()

dist = standardize(pd.read_csv(f_dist))
block = standardize(pd.read_csv(f_block))
cluster = standardize(pd.read_csv(f_cluster))

validate(dist, REQ_DIST, "District file")
validate(block, REQ_BLOCK, "Block file")
validate(cluster, REQ_CLUSTER, "Cluster file")

if show_preview:
    st.subheader("District file preview")
    st.dataframe(dist.head(30), use_container_width=True)
    st.subheader("Block file preview")
    st.dataframe(block.head(30), use_container_width=True)
    st.subheader("Cluster file preview")
    st.dataframe(cluster.head(30), use_container_width=True)

# Unified for totals and mentor-100 calculations
unified = pd.concat(
    [
        dist[list(REQ_DIST)].assign(**{"Block Name": np.nan, "Cluster Name": np.nan}),
        block[list(REQ_BLOCK)].assign(**{"Cluster Name": np.nan}),
        cluster[list(REQ_CLUSTER)],
    ],
    ignore_index=True
)

# ============================================================
# TABS
# ============================================================
tab_state, tab_district, tab_block = st.tabs(
    ["🏛️ State (All Districts)", "🏢 District Drilldown", "🏫 Block Drilldown"]
)

# ============================================================
# 1) STATE TAB
# ============================================================
with tab_state:
    st.subheader("State Level: District-wise Performance (sorted high → low)")

    all_districts = sorted(
        set(dist["District Name"].dropna().tolist())
        | set(block["District Name"].dropna().tolist())
        | set(cluster["District Name"].dropna().tolist())
    )
    base = pd.DataFrame({"District Name": all_districts})

    # Merge role-wide stats
    for r in DIST_ROLES:
        base = base.merge(agg_role_wide(dist, ["District Name"], r), on="District Name", how="left")
    for r in BLOCK_ROLES:
        base = base.merge(agg_role_wide(block, ["District Name"], r), on="District Name", how="left")
    for r in CLUSTER_ROLES:
        base = base.merge(agg_role_wide(cluster, ["District Name"], r), on="District Name", how="left")

    # Totals + mentor100%
    totals = overall_by_group(unified, ["District Name"])
    base = base.merge(totals, on="District Name", how="left")

    # Rank and sort
    base = base.sort_values(["Total %", "District Name"], ascending=[False, True]).reset_index(drop=True)
    base.insert(0, "Rank", np.arange(1, len(base) + 1))

    # Arrange columns in your format
    ordered_cols = ["Rank", "District Name"]
    for r in ["DPC", "DIET Principal", "DIET Academic", "APC", "BRCC", "BAC", "CAC"]:
        ordered_cols += [f"{r} Target", f"{r} Completed", f"{r} %"]
    ordered_cols += ["Total_Target", "Total_Completed", "Total %", "% of Mentors Completing 100% Mentoring Visit"]

    state_table = base[ordered_cols].copy()

    # Pretty table
    pct_cols = [c for c in state_table.columns if c.endswith(" %") or c.endswith("%") or c == "Total %"]
    st.dataframe(style_pct_cols(state_table, pct_cols), use_container_width=True)

    # Optional chart: Top 20
    top20 = state_table.sort_values("Total %", ascending=False).head(20)
    fig = px.bar(top20, x="District Name", y="Total %", title="Top 20 Districts by Total % Completion")
    fig.update_layout(height=420, xaxis_title="", yaxis_title="Total %")
    st.plotly_chart(fig, use_container_width=True)

    if enable_export:
        to_excel_download(
            {"State_District_Performance": state_table},
            filename="state_district_performance.xlsx"
        )

# ============================================================
# 2) DISTRICT TAB
# ============================================================
with tab_district:
    st.subheader("District Drilldown (sorted high → low)")

    all_districts = sorted(cluster["District Name"].dropna().unique().tolist())
    sel_d = st.selectbox("Select District", options=all_districts)

    # ---------------------------
    # Table 1: Overall district performance
    # (A) District mentors (DPC/DIET/APC)
    # ---------------------------
    st.markdown("### 1) Overall District Performance")

    st.markdown("#### District Mentors (role-wise)")
    dm = dist[dist["District Name"] == sel_d].copy()
    dm["is100"] = mentor_100_flag(dm)

    dm_sum = dm.groupby("Role Name", dropna=False).agg(
        Number_of_Mentors=("Employee Code", "nunique"),
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
        Mentors=("Employee Code", "nunique"),
        Mentors100=("is100", "sum"),
    ).reset_index()

    dm_sum["% Completion"] = np.where(dm_sum["Target_Visit"] > 0, (dm_sum["Completed_Visit"] / dm_sum["Target_Visit"]) * 100, np.nan)
    dm_sum["% of Mentors Completing 100% Visits"] = np.where(dm_sum["Mentors"] > 0, (dm_sum["Mentors100"] / dm_sum["Mentors"]) * 100, np.nan)

    dm_sum = dm_sum.rename(columns={"Role Name": "District Mentors"})
    dm_sum = dm_sum[dm_sum["District Mentors"].isin(DIST_ROLES)].copy()
    dm_sum = dm_sum.sort_values("% Completion", ascending=False)
    dm_sum = dm_sum.drop(columns=["Mentors", "Mentors100"])

    st.dataframe(style_pct_cols(dm_sum, ["% Completion", "% of Mentors Completing 100% Visits"]), use_container_width=True)

    # (B) Block mentors combined rollup (BAC+BRCC+CAC)
    st.markdown("#### Block-wise Combined Mentors (BAC + BRCC + CAC)")
    blk = block[block["District Name"] == sel_d].copy()
    cac = cluster[cluster["District Name"] == sel_d].copy()

    combo = pd.concat([blk, cac], ignore_index=True)
    combo["is100"] = mentor_100_flag(combo)

    bsum = combo.groupby("Block Name", dropna=False).agg(
        Number_of_Mentors=("Employee Code", "nunique"),
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
        Mentors=("Employee Code", "nunique"),
        Mentors100=("is100", "sum"),
    ).reset_index()

    bsum["% Completion"] = np.where(bsum["Target_Visit"] > 0, (bsum["Completed_Visit"] / bsum["Target_Visit"]) * 100, np.nan)
    bsum["% of Mentors Completing 100% Visits"] = np.where(bsum["Mentors"] > 0, (bsum["Mentors100"] / bsum["Mentors"]) * 100, np.nan)
    bsum = bsum.drop(columns=["Mentors", "Mentors100"]).sort_values("% Completion", ascending=False)

    st.dataframe(style_pct_cols(bsum, ["% Completion", "% of Mentors Completing 100% Visits"]), use_container_width=True)

    # ---------------------------
    # Table 2: BRCC Compliance (Block-wise)
    # ---------------------------
    st.markdown("### 2) BRCC Compliance (Block-wise)")
    brcc = block[(block["District Name"] == sel_d) & (block["Role Name"] == "BRCC")].copy()
    brcc_sum = brcc.groupby("Block Name", dropna=False).agg(
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
    ).reset_index()
    brcc_sum["% Completion"] = np.where(brcc_sum["Target_Visit"] > 0, (brcc_sum["Completed_Visit"] / brcc_sum["Target_Visit"]) * 100, np.nan)
    brcc_sum = brcc_sum.sort_values("% Completion", ascending=False)
    st.dataframe(style_pct_cols(brcc_sum, ["% Completion"]), use_container_width=True)

    # ---------------------------
    # Table 3: BAC Compliance (Block-wise)
    # ---------------------------
    st.markdown("### 3) BAC Compliance (Block-wise)")
    bac = block[(block["District Name"] == sel_d) & (block["Role Name"] == "BAC")].copy()
    bac["is100"] = mentor_100_flag(bac)
    bac_sum = bac.groupby("Block Name", dropna=False).agg(
        Number_of_Mentors=("Employee Code", "nunique"),
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
        Mentors=("Employee Code", "nunique"),
        Mentors100=("is100", "sum"),
    ).reset_index()
    bac_sum["% Completion"] = np.where(bac_sum["Target_Visit"] > 0, (bac_sum["Completed_Visit"] / bac_sum["Target_Visit"]) * 100, np.nan)
    bac_sum["% of Mentors Completing 100% Visits"] = np.where(bac_sum["Mentors"] > 0, (bac_sum["Mentors100"] / bac_sum["Mentors"]) * 100, np.nan)
    bac_sum = bac_sum.drop(columns=["Mentors", "Mentors100"]).sort_values("% Completion", ascending=False)
    st.dataframe(style_pct_cols(bac_sum, ["% Completion", "% of Mentors Completing 100% Visits"]), use_container_width=True)

    # ---------------------------
    # Table 4: CAC Compliance (Block-wise)
    # ---------------------------
    st.markdown("### 4) CAC Compliance (Block-wise)")
    cac_d = cluster[cluster["District Name"] == sel_d].copy()
    cac_d["is100"] = mentor_100_flag(cac_d)
    cac_sum = cac_d.groupby("Block Name", dropna=False).agg(
        Number_of_Mentors=("Employee Code", "nunique"),
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
        Mentors=("Employee Code", "nunique"),
        Mentors100=("is100", "sum"),
    ).reset_index()
    cac_sum["% Completion"] = np.where(cac_sum["Target_Visit"] > 0, (cac_sum["Completed_Visit"] / cac_sum["Target_Visit"]) * 100, np.nan)
    cac_sum["% of Mentors Completing 100% Visits"] = np.where(cac_sum["Mentors"] > 0, (cac_sum["Mentors100"] / cac_sum["Mentors"]) * 100, np.nan)
    cac_sum = cac_sum.drop(columns=["Mentors", "Mentors100"]).sort_values("% Completion", ascending=False)
    st.dataframe(style_pct_cols(cac_sum, ["% Completion", "% of Mentors Completing 100% Visits"]), use_container_width=True)

    if enable_export:
        to_excel_download(
            {
                "District_Mentors": dm_sum,
                "Block_Combined_Mentors": bsum,
                "BRCC_Compliance": brcc_sum,
                "BAC_Compliance": bac_sum,
                "CAC_Compliance": cac_sum,
            },
            filename=f"{sel_d}_district_drilldown.xlsx"
        )

# ============================================================
# 3) BLOCK TAB
# ============================================================
with tab_block:
    st.subheader("Block Drilldown (sorted high → low)")

    block_pairs = cluster[["District Name", "Block Name"]].dropna().drop_duplicates()
    block_pairs = block_pairs.sort_values(["District Name", "Block Name"])
    options = (block_pairs["District Name"] + " | " + block_pairs["Block Name"]).tolist()

    sel = st.selectbox("Select Block (District | Block)", options=options)
    sel_d, sel_b = sel.split(" | ", 1)

    cb = cluster[(cluster["District Name"] == sel_d) & (cluster["Block Name"] == sel_b)].copy()

    # ---------------------------
    # Table 1: Cluster Compliance
    # ---------------------------
    st.markdown("### 1) Cluster Compliance")
    clus = cb.groupby("Cluster Name", dropna=False).agg(
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
    ).reset_index()
    clus["% Completion"] = np.where(clus["Target_Visit"] > 0, (clus["Completed_Visit"] / clus["Target_Visit"]) * 100, np.nan)
    clus = clus.sort_values("% Completion", ascending=False)
    st.dataframe(style_pct_cols(clus, ["% Completion"]), use_container_width=True)

    # ---------------------------
    # Table 2: CAC Leaderboard
    # ---------------------------
    st.markdown("### 2) CAC Leaderboard")
    lb = cb.groupby(["Employee Name", "Cluster Name"], dropna=False).agg(
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
    ).reset_index()
    lb["% Completion"] = np.where(lb["Target_Visit"] > 0, (lb["Completed_Visit"] / lb["Target_Visit"]) * 100, np.nan)
    lb = lb.sort_values("% Completion", ascending=False)
    lb = lb.rename(columns={"Employee Name": "Mentor Name"})

    st.dataframe(style_pct_cols(lb, ["% Completion"]), use_container_width=True)

    if enable_export:
        to_excel_download(
            {
                "Cluster_Compliance": clus,
                "CAC_Leaderboard": lb,
            },
            filename=f"{sel_d}_{sel_b}_block_drilldown.xlsx"
        )

st.success("✅ Updated dashboard code loaded. If you still see an older UI, stop Streamlit (Ctrl+C) and run again.")
