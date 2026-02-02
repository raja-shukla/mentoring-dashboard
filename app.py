import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Mentoring Visit Dashboard", layout="wide")

# ============================================================
# CSS (clean + readable sidebar)
# ============================================================
st.markdown(
    """
<style>
.main { background: linear-gradient(180deg, #fbfbff 0%, #ffffff 60%); }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1380px; }

h1 { font-size: 2.15rem !important; font-weight: 900 !important; letter-spacing: -0.6px; margin-bottom: 0.3rem; }
.small-muted { color: #6b7280; font-size: 0.92rem; }

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 14px 14px 10px 14px;
    border-radius: 16px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}
div[data-testid="stMetric"] label { color: #6b7280 !important; font-weight: 650 !important; }
div[data-testid="stMetric"] div { font-weight: 850 !important; }

section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] * { color: #0f172a !important; }
section[data-testid="stSidebar"] small { color: #64748b !important; }

section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 10px;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
}

.callout {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}
.callout-title { font-weight: 900; font-size: 1.02rem; margin-bottom: 0.2rem; }
.callout-sub { color: #6b7280; font-size: 0.9rem; }

.section-title { font-size: 1.05rem; font-weight: 900; margin-top: 1.05rem; margin-bottom: 0.2rem; }
.section-sub { color: #6b7280; font-size: 0.9rem; margin-bottom: 0.7rem; }

hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 1.1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
REQ_DIST = {"District Name", "Employee Code", "Employee Name", "Role Name", "Targeted Visits", "Completed"}
REQ_BLOCK = {"District Name", "Block Name", "Employee Code", "Employee Name", "Role Name", "Targeted Visits", "Completed"}
REQ_CLUSTER = {"District Name", "Block Name", "Cluster Name", "Employee Code", "Employee Name", "Role Name", "Targeted Visits", "Completed"}

DIST_ROLES = ["DPC", "DIET Principal", "DIET Academic", "APC"]
BLOCK_ROLES = ["BRCC", "BAC"]
CLUSTER_ROLES = ["CAC"]

ROLE_CANON = {
    "BRC": "BRCC",
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

    text_cols = ["District Name", "Block Name", "Cluster Name", "Employee Name", "Role Name", "Employee Code"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "None", "NaN", ""]), c] = np.nan

    for c in ["Targeted Visits", "Completed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "Role Name" in df.columns:
        df["Role Name"] = df["Role Name"].map(lambda x: ROLE_CANON.get(str(x).strip(), str(x).strip()))

    return df


def validate(df: pd.DataFrame, required: set, label: str):
    missing = required - set(df.columns)
    if missing:
        st.error(f"{label} missing columns: {sorted(list(missing))}")
        st.stop()


def add_true_pending_and_extra(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Pending_i"] = (d["Targeted Visits"] - d["Completed"]).clip(lower=0)
    d["Extra_i"] = (d["Completed"] - d["Targeted Visits"]).clip(lower=0)
    return d


def overall_stats(df: pd.DataFrame) -> dict:
    d = add_true_pending_and_extra(df)
    total_target = float(d["Targeted Visits"].sum())
    total_completed = float(d["Completed"].sum())
    completion = (total_completed / total_target * 100) if total_target > 0 else np.nan

    mentors_total = int(d["Employee Code"].nunique()) if "Employee Code" in d.columns else int(d["Employee Name"].nunique())

    m = d.groupby("Employee Code", dropna=False).agg(
        t=("Targeted Visits", "sum"),
        c=("Completed", "sum")
    )
    mentors_100 = int(((m["t"] > 0) & (m["c"] >= m["t"])).sum())
    mentors_100_pct = (mentors_100 / mentors_total * 100) if mentors_total > 0 else np.nan

    pending_true = int(d["Pending_i"].sum())

    return {
        "completion": completion,
        "total_target": int(total_target),
        "total_completed": int(total_completed),
        "mentors_total": mentors_total,
        "mentors_100": mentors_100,
        "mentors_100_pct": mentors_100_pct,
        "mentors_not_100": max(0, mentors_total - mentors_100),
        "pending_true": pending_true,
    }


def agg_role_wide(df: pd.DataFrame, group_cols: list[str], role: str) -> pd.DataFrame:
    d = df[df["Role Name"] == role].copy()
    if d.empty:
        out_cols = group_cols + [f"{role} Target", f"{role} Completed", f"{role} %"]
        return pd.DataFrame(columns=out_cols)

    g = d.groupby(group_cols, dropna=False).agg(
        Target=("Targeted Visits", "sum"),
        Completed=("Completed", "sum")
    ).reset_index()

    g[f"{role} Target"] = g["Target"].astype(int)
    g[f"{role} Completed"] = g["Completed"].astype(int)
    g[f"{role} %"] = np.where(g["Target"] > 0, (g["Completed"] / g["Target"]) * 100, np.nan)

    return g[group_cols + [f"{role} Target", f"{role} Completed", f"{role} %"]]


def overall_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = add_true_pending_and_extra(df)

    g = d.groupby(group_cols, dropna=False).agg(
        Total_Target=("Targeted Visits", "sum"),
        Total_Completed=("Completed", "sum"),
        Pending_True=("Pending_i", "sum"),
        Mentors=("Employee Code", "nunique"),
    ).reset_index()

    g["Total_Target"] = g["Total_Target"].astype(int)
    g["Total_Completed"] = g["Total_Completed"].astype(int)
    g["Pending_True"] = g["Pending_True"].astype(int)

    g["Total %"] = np.where(g["Total_Target"] > 0, (g["Total_Completed"] / g["Total_Target"]) * 100, np.nan)

    # mentor-level 100% per group
    d["_key"] = d[group_cols].astype(str).agg("||".join, axis=1)
    m = d.groupby(["_key", "Employee Code"], dropna=False).agg(
        t=("Targeted Visits", "sum"),
        c=("Completed", "sum")
    ).reset_index()
    m["is100m"] = (m["t"] > 0) & (m["c"] >= m["t"])
    msum = m.groupby("_key").agg(Mentors100=("is100m", "sum")).reset_index()

    g["_key"] = g[group_cols].astype(str).agg("||".join, axis=1)
    g = g.merge(msum, on="_key", how="left").drop(columns=["_key"])
    g["Mentors100"] = g["Mentors100"].fillna(0).astype(int)
    g["Mentors NOT 100%"] = (g["Mentors"] - g["Mentors100"]).clip(lower=0).astype(int)
    g["% Mentors 100%"] = np.where(g["Mentors"] > 0, (g["Mentors100"] / g["Mentors"]) * 100, np.nan)

    return g


def kpi_row(metrics: list[tuple[str, str, str]]):
    cols = st.columns(len(metrics))
    for i, (label, value, delta) in enumerate(metrics):
        cols[i].metric(label, value, delta if delta else None)


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = str(name)[:31]  # excel sheet name limit
            df.to_excel(writer, index=False, sheet_name=safe)
    buff.seek(0)
    return buff.getvalue()


def mentor_exception_table(df: pd.DataFrame, scope_label: str) -> tuple[dict, pd.DataFrame]:
    """
    Returns:
      - summary dict for metrics row
      - exception table with mentor details (no mentor id)
    """
    d = add_true_pending_and_extra(df).copy()

    # Attach block/cluster "best available" for each mentor (mode of non-null)
    def pick_mode(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        return s.mode().iloc[0] if not s.mode().empty else s.iloc[0]

    info = d.groupby("Employee Code", dropna=False).agg(
        Mentor_Name=("Employee Name", pick_mode),
        Role_Name=("Role Name", pick_mode),
        Block_Name=("Block Name", pick_mode),
        Cluster_Name=("Cluster Name", pick_mode),
    ).reset_index(drop=False)

    agg = d.groupby("Employee Code", dropna=False).agg(
        Target_Visit=("Targeted Visits", "sum"),
        Completed_Visit=("Completed", "sum"),
        Pending_Visits=("Pending_i", "sum"),
    ).reset_index()

    out = agg.merge(info, on="Employee Code", how="left")

    out["% Completion"] = np.where(
        out["Target_Visit"] > 0,
        (out["Completed_Visit"] / out["Target_Visit"]) * 100,
        np.nan
    )

    # Mentors NOT 100%
    exc = out[(out["Target_Visit"] > 0) & (out["Completed_Visit"] < out["Target_Visit"])].copy()
    exc = exc.sort_values(["% Completion", "Pending_Visits"], ascending=[True, False])

    # User wants: block name, cluster name; remove mentor id
    exc_view = exc.rename(columns={
        "Mentor_Name": "Mentor Name",
        "Role_Name": "Role Name",
        "Block_Name": "Block Name",
        "Cluster_Name": "Cluster Name",
        "Target_Visit": "Target Visit",
        "Completed_Visit": "Completed Visit",
    })[[
        "Mentor Name", "Role Name", "Block Name", "Cluster Name",
        "Target Visit", "Completed Visit", "% Completion", "Pending_Visits"
    ]].rename(columns={"Pending_Visits": "Pending Visits"})

    # Summary metrics
    summary = {
        "mentors_not_100": int(exc_view.shape[0]),
        "pending_total": int(exc_view["Pending Visits"].sum()) if not exc_view.empty else 0,
        "median_pct": float(exc_view["% Completion"].median()) if not exc_view.empty else np.nan,
        "worst_pct": float(exc_view["% Completion"].min()) if not exc_view.empty else np.nan,
        "label": scope_label
    }
    return summary, exc_view


def role_compliance_table(df: pd.DataFrame, group_col: str, role: str) -> pd.DataFrame:
    d = df[df["Role Name"] == role].copy()
    d = add_true_pending_and_extra(d)

    if d.empty:
        return pd.DataFrame(columns=[
            group_col, "Number of Mentors", "Target Visit", "Completed Visit",
            "% Completion", "% of Mentors Completing 100% Mentoring Visit",
            "Number of Mentors not completing visit Target", "Pending Visits"
        ])

    g = d.groupby(group_col, dropna=False).agg(
        Target=("Targeted Visits", "sum"),
        Completed=("Completed", "sum"),
        Pending=("Pending_i", "sum"),
        Mentors=("Employee Code", "nunique"),
    ).reset_index()

    g["Target"] = g["Target"].astype(int)
    g["Completed"] = g["Completed"].astype(int)
    g["Pending"] = g["Pending"].astype(int)

    g["% Completion"] = np.where(g["Target"] > 0, (g["Completed"] / g["Target"]) * 100, np.nan)

    # mentor-level 100%
    m = d.groupby([group_col, "Employee Code"], dropna=False).agg(
        t=("Targeted Visits", "sum"),
        c=("Completed", "sum")
    ).reset_index()
    m["is100m"] = (m["t"] > 0) & (m["c"] >= m["t"])
    msum = m.groupby(group_col).agg(Mentors100=("is100m", "sum")).reset_index()

    g = g.merge(msum, on=group_col, how="left")
    g["Mentors100"] = g["Mentors100"].fillna(0).astype(int)
    g["% of Mentors Completing 100% Mentoring Visit"] = np.where(g["Mentors"] > 0, (g["Mentors100"] / g["Mentors"]) * 100, np.nan)
    g["Number of Mentors not completing visit Target"] = (g["Mentors"] - g["Mentors100"]).clip(lower=0).astype(int)

    out = pd.DataFrame({
        group_col: g[group_col],
        "Number of Mentors": g["Mentors"].astype(int),
        "Target Visit": g["Target"],
        "Completed Visit": g["Completed"],
        "% Completion": g["% Completion"],
        "% of Mentors Completing 100% Mentoring Visit": g["% of Mentors Completing 100% Mentoring Visit"],
        "Number of Mentors not completing visit Target": g["Number of Mentors not completing visit Target"],
        "Pending Visits": g["Pending"],
    })

    out = out.sort_values("% Completion", ascending=False)
    return out


def district_mentors_table(dm: pd.DataFrame) -> pd.DataFrame:
    d = dm.copy()
    d = add_true_pending_and_extra(d)
    d = d[d["Role Name"].isin(DIST_ROLES)].copy()

    if d.empty:
        return pd.DataFrame(columns=[
            "District Mentors", "Number of Mentors", "Target Visit", "Completed Visit",
            "% Completion", "% of Mentors Completing 100% Mentoring Visit",
            "Number of Mentors not completing visit Target", "Pending Visits"
        ])

    g = d.groupby("Role Name", dropna=False).agg(
        Mentors=("Employee Code", "nunique"),
        Target=("Targeted Visits", "sum"),
        Completed=("Completed", "sum"),
        Pending=("Pending_i", "sum"),
    ).reset_index()

    g["Target"] = g["Target"].astype(int)
    g["Completed"] = g["Completed"].astype(int)
    g["Pending"] = g["Pending"].astype(int)
    g["% Completion"] = np.where(g["Target"] > 0, (g["Completed"] / g["Target"]) * 100, np.nan)

    m = d.groupby(["Role Name", "Employee Code"], dropna=False).agg(
        t=("Targeted Visits", "sum"),
        c=("Completed", "sum")
    ).reset_index()
    m["is100m"] = (m["t"] > 0) & (m["c"] >= m["t"])
    msum = m.groupby("Role Name").agg(Mentors100=("is100m", "sum")).reset_index()

    g = g.merge(msum, on="Role Name", how="left")
    g["Mentors100"] = g["Mentors100"].fillna(0).astype(int)

    out = pd.DataFrame({
        "District Mentors": g["Role Name"],
        "Number of Mentors": g["Mentors"].astype(int),
        "Target Visit": g["Target"],
        "Completed Visit": g["Completed"],
        "% Completion": g["% Completion"],
        "% of Mentors Completing 100% Mentoring Visit": np.where(g["Mentors"] > 0, (g["Mentors100"] / g["Mentors"]) * 100, np.nan),
        "Number of Mentors not completing visit Target": (g["Mentors"] - g["Mentors100"]).clip(lower=0).astype(int),
        "Pending Visits": g["Pending"],
    })

    role_order = {r: i for i, r in enumerate(DIST_ROLES)}
    out["__ord"] = out["District Mentors"].map(lambda x: role_order.get(x, 999))
    out = out.sort_values("__ord").drop(columns="__ord")
    return out


# ============================================================
# HEADER
# ============================================================
st.title("📊 Mentoring Visit Dashboard")
st.markdown(
    "<div class='small-muted'>State → District → Block drill-down | actual completion % | role-wise compliance | mentor exception lists</div>",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR (UPLOADS)
# ============================================================
with st.sidebar:
    st.header("Upload raw files (CSV)")
    f_dist = st.file_uploader("District file", type=["csv"])
    f_block = st.file_uploader("Block file", type=["csv"])
    f_cluster = st.file_uploader("Cluster file", type=["csv"])

    st.divider()
    st.header("Options")
    show_preview = st.checkbox("Show file previews", value=False)

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
    st.markdown('<div class="section-title">File previews</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.caption("District")
        st.dataframe(dist.head(25), use_container_width=True, height=330)
    with b:
        st.caption("Block")
        st.dataframe(block.head(25), use_container_width=True, height=330)
    with c:
        st.caption("Cluster")
        st.dataframe(cluster.head(25), use_container_width=True, height=330)

unified = pd.concat(
    [
        dist[list(REQ_DIST)].assign(**{"Block Name": np.nan, "Cluster Name": np.nan}),
        block[list(REQ_BLOCK)].assign(**{"Cluster Name": np.nan}),
        cluster[list(REQ_CLUSTER)],
    ],
    ignore_index=True,
)

# ============================================================
# TABS
# ============================================================
tab_state, tab_district, tab_block = st.tabs(
    ["🏛️ State (All Districts)", "🏢 District Drilldown", "🏫 Block Drilldown"]
)

# ============================================================
# STATE TAB
# ============================================================
with tab_state:
    st.markdown('<div class="section-title">State Level: District-wise performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Actual completion % (not capped). Pending Visits = TRUE pending (individual shortfalls).</div>', unsafe_allow_html=True)

    s = overall_stats(unified)
    kpi_row([
        ("State Completion %", f"{s['completion']:.1f}%", ""),
        ("Total Mentors", f"{s['mentors_total']:,}", ""),
        ("Mentors with 100%", f"{s['mentors_100']:,} ({s['mentors_100_pct']:.1f}%)", ""),
        ("Pending Visits", f"{s['pending_true']:,}", ""),
    ])

    all_districts = sorted(
        set(dist["District Name"].dropna().tolist())
        | set(block["District Name"].dropna().tolist())
        | set(cluster["District Name"].dropna().tolist())
    )
    base = pd.DataFrame({"District Name": all_districts})

    for r in DIST_ROLES:
        base = base.merge(agg_role_wide(dist, ["District Name"], r), on="District Name", how="left")
    for r in BLOCK_ROLES:
        base = base.merge(agg_role_wide(block, ["District Name"], r), on="District Name", how="left")
    for r in CLUSTER_ROLES:
        base = base.merge(agg_role_wide(cluster, ["District Name"], r), on="District Name", how="left")

    totals = overall_by_group(unified, ["District Name"])
    base = base.merge(totals, on="District Name", how="left")

    base = base.sort_values(["Total %", "District Name"], ascending=[False, True]).reset_index(drop=True)
    base.insert(0, "Rank", np.arange(1, len(base) + 1))

    state_table = pd.DataFrame({
        "Rank": base["Rank"],
        "District Name": base["District Name"],

        "DPC Target": base.get("DPC Target", np.nan),
        "DPC Completed": base.get("DPC Completed", np.nan),
        "DPC %": base.get("DPC %", np.nan),

        "DIET Principal Target": base.get("DIET Principal Target", np.nan),
        "DIET Principal Completed": base.get("DIET Principal Completed", np.nan),
        "DIET Principal %": base.get("DIET Principal %", np.nan),

        "DIET Academic Target": base.get("DIET Academic Target", np.nan),
        "DIET Academic Completed": base.get("DIET Academic Completed", np.nan),
        "DIET Academic %": base.get("DIET Academic %", np.nan),

        "APC Target": base.get("APC Target", np.nan),
        "APC Completed": base.get("APC Completed", np.nan),
        "APC %": base.get("APC %", np.nan),

        "BRCC Target": base.get("BRCC Target", np.nan),
        "BRCC Completed": base.get("BRCC Completed", np.nan),
        "BRCC %": base.get("BRCC %", np.nan),

        "BAC Target": base.get("BAC Target", np.nan),
        "BAC Completed": base.get("BAC Completed", np.nan),
        "BAC %": base.get("BAC %", np.nan),

        "CAC Target": base.get("CAC Target", np.nan),
        "CAC Completed": base.get("CAC Completed", np.nan),
        "CAC %": base.get("CAC %", np.nan),

        "Total Target": base["Total_Target"].astype("Int64"),
        "Total Completed": base["Total_Completed"].astype("Int64"),
        "Total %": base["Total %"],

        "% of Mentors Completing 100% Mentoring Visit": base["% Mentors 100%"],
        "Number of  Mentors not completing visit Target": base["Mentors NOT 100%"].astype("Int64"),
        "Pending Visits": base["Pending_True"].astype("Int64"),
    })

    st.markdown('<div class="section-title">District performance table (ranked)</div>', unsafe_allow_html=True)
    st.dataframe(state_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">District completion charts</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top 15 (green) and Bottom 15 (red) by Total % (actual).</div>', unsafe_allow_html=True)

    chart_src = state_table[["District Name", "Total %"]].dropna().copy()
    top15 = chart_src.sort_values("Total %", ascending=False).head(15).sort_values("Total %", ascending=True)
    bot15 = chart_src.sort_values("Total %", ascending=True).head(15).sort_values("Total %", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig_top = px.bar(
            top15, x="Total %", y="District Name", orientation="h",
            text=top15["Total %"].map(lambda v: f"{v:.1f}%"),
            title="Top 15 Districts by Completion % (Actual)"
        )
        fig_top.update_traces(marker_color="rgba(34,197,94,0.85)", textposition="outside", cliponaxis=False)
        fig_top.update_layout(height=520, margin=dict(l=18, r=18, t=55, b=20),
                              paper_bgcolor="white", plot_bgcolor="white",
                              xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                              xaxis_title="Completion % (Actual)", yaxis_title="")
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        fig_bot = px.bar(
            bot15, x="Total %", y="District Name", orientation="h",
            text=bot15["Total %"].map(lambda v: f"{v:.1f}%"),
            title="Bottom 15 Districts by Completion % (Actual)"
        )
        fig_bot.update_traces(marker_color="rgba(239,68,68,0.85)", textposition="outside", cliponaxis=False)
        fig_bot.update_layout(height=520, margin=dict(l=18, r=18, t=55, b=20),
                              paper_bgcolor="white", plot_bgcolor="white",
                              xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                              xaxis_title="Completion % (Actual)", yaxis_title="")
        st.plotly_chart(fig_bot, use_container_width=True)

    st.divider()
    state_xlsx = to_excel_bytes({
        "State_District_Performance": state_table,
        "Top15": top15,
        "Bottom15": bot15
    })
    st.download_button(
        "⬇️ Download State Report (Excel)",
        data=state_xlsx,
        file_name="state_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ============================================================
# DISTRICT TAB
# ============================================================
with tab_district:
    st.markdown('<div class="section-title">District drilldown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">District mentors + block mentors + role-wise compliance + mentor exception list.</div>', unsafe_allow_html=True)

    all_districts = sorted(cluster["District Name"].dropna().unique().tolist())
    sel_d = st.selectbox("Select District", options=all_districts)

    dm = dist[dist["District Name"] == sel_d].copy()
    blk = block[block["District Name"] == sel_d].copy()
    cac_d = cluster[cluster["District Name"] == sel_d].copy()

    district_all = pd.concat(
        [dm.assign(**{"Block Name": np.nan, "Cluster Name": np.nan}),
         blk.assign(**{"Cluster Name": np.nan}),
         cac_d],
        ignore_index=True
    )

    ds = overall_stats(district_all)
    kpi_row([
        (f"{sel_d} Completion %", f"{ds['completion']:.1f}%", ""),
        ("Total Mentors", f"{ds['mentors_total']:,}", ""),
        ("Mentors with 100%", f"{ds['mentors_100']:,} ({ds['mentors_100_pct']:.1f}%)", ""),
        ("Pending Visits", f"{ds['pending_true']:,}", ""),
    ])

    st.markdown('<div class="section-title">1) District Mentors</div>', unsafe_allow_html=True)
    dmt = district_mentors_table(dm)
    st.dataframe(dmt, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">2) Block Mentors Combined (BRCC + BAC + CAC)</div>', unsafe_allow_html=True)
    combo = pd.concat([blk, cac_d], ignore_index=True)
    bsum = overall_by_group(combo, ["Block Name"]).rename(columns={
        "Total_Target": "Target Visit",
        "Total_Completed": "Completed Visit",
        "Pending_True": "Pending Visits",
        "Total %": "% Completion",
        "% Mentors 100%": "% of Mentors Completing 100% Mentoring Visit",
        "Mentors NOT 100%": "Number of Mentors not completing visit Target",
        "Mentors": "Number of Mentors",
        "Mentors100": "Mentors with 100%",
    }).sort_values("% Completion", ascending=False)

    bsum_view = bsum[[
        "Block Name",
        "Number of Mentors",
        "Mentors with 100%",
        "% of Mentors Completing 100% Mentoring Visit",
        "Target Visit",
        "Completed Visit",
        "Pending Visits",
        "% Completion",
        "Number of Mentors not completing visit Target",
    ]].copy()

    st.dataframe(bsum_view, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">3) BRCC (Block-wise)</div>', unsafe_allow_html=True)
    brcc_t = role_compliance_table(blk, "Block Name", "BRCC")
    st.dataframe(brcc_t, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">4) BAC (Block-wise)</div>', unsafe_allow_html=True)
    bac_t = role_compliance_table(blk, "Block Name", "BAC")
    st.dataframe(bac_t, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">5) CAC (Block-wise)</div>', unsafe_allow_html=True)
    cac_t = role_compliance_table(cac_d, "Block Name", "CAC")
    st.dataframe(cac_t, use_container_width=True, hide_index=True)

    # ✅ Mentor exception list (District)
    st.divider()
    st.markdown('<div class="section-title">Mentor Exception List (District)</div>', unsafe_allow_html=True)
    summary_d, exc_d = mentor_exception_table(district_all, scope_label=sel_d)

    kpi_row([
        ("Mentors not at 100%", f"{summary_d['mentors_not_100']:,}", ""),
        ("Total pending visits", f"{summary_d['pending_total']:,}", ""),
        ("Median % completion", f"{summary_d['median_pct']:.1f}%" if not np.isnan(summary_d["median_pct"]) else "—", ""),
        ("Worst % completion", f"{summary_d['worst_pct']:.1f}%" if not np.isnan(summary_d["worst_pct"]) else "—", ""),
    ])

    st.markdown(f"### 🚨 {sel_d}: Mentors NOT completing 100% target")
    st.caption("Sorted worst-first: lowest % completion, then highest pending visits.")
    st.dataframe(exc_d, use_container_width=True, hide_index=True)

    st.divider()
    district_xlsx = to_excel_bytes({
        "District_Mentors": dmt,
        "Block_Mentors_Combined": bsum_view,
        "BRCC_Blockwise": brcc_t,
        "BAC_Blockwise": bac_t,
        "CAC_Blockwise": cac_t,
        "Mentor_Exception_List": exc_d,
    })
    st.download_button(
        "⬇️ Download District Report (Excel)",
        data=district_xlsx,
        file_name=f"{sel_d}_district_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ============================================================
# BLOCK TAB
# ============================================================
with tab_block:
    st.markdown('<div class="section-title">Block drilldown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cluster compliance + CAC leaderboard + mentor exception list.</div>', unsafe_allow_html=True)

    block_pairs = cluster[["District Name", "Block Name"]].dropna().drop_duplicates().sort_values(["District Name", "Block Name"])
    options = (block_pairs["District Name"] + " | " + block_pairs["Block Name"]).tolist()

    sel = st.selectbox("Select Block (District | Block)", options=options)
    sel_d, sel_b = sel.split(" | ", 1)

    cb = cluster[(cluster["District Name"] == sel_d) & (cluster["Block Name"] == sel_b)].copy()
    blk_rows = block[(block["District Name"] == sel_d) & (block["Block Name"] == sel_b)].copy()

    block_all = pd.concat([blk_rows.assign(**{"Cluster Name": np.nan}), cb], ignore_index=True)
    if block_all.empty:
        st.warning("No data found for this block.")
        st.stop()

    bs = overall_stats(block_all)
    kpi_row([
        (f"{sel_b} Completion %", f"{bs['completion']:.1f}%", ""),
        ("Total Mentors", f"{bs['mentors_total']:,}", ""),
        ("Mentors with 100%", f"{bs['mentors_100']:,} ({bs['mentors_100_pct']:.1f}%)", ""),
        ("Pending Visits", f"{bs['pending_true']:,}", ""),
    ])

    st.markdown('<div class="section-title">1) Cluster compliance</div>', unsafe_allow_html=True)
    if cb.empty:
        st.info("No CAC/Cluster data available for this block.")
        clus_view = pd.DataFrame(columns=["Cluster Name", "Target Visit", "Completed Visit", "% Completion", "Pending Visits"])
        lb_view = pd.DataFrame(columns=["Mentor Name", "Cluster Name", "Target_Visit", "Completed_Visit", "% Completion"])
    else:
        cbc = add_true_pending_and_extra(cb)

        clus = cbc.groupby("Cluster Name", dropna=False).agg(
            Target_Visit=("Targeted Visits", "sum"),
            Completed_Visit=("Completed", "sum"),
        ).reset_index()

        clus["Pending Visits"] = (clus["Target_Visit"] - clus["Completed_Visit"]).clip(lower=0).astype(int)
        clus["% Completion"] = np.where(
            clus["Target_Visit"] > 0, (clus["Completed_Visit"] / clus["Target_Visit"]) * 100, np.nan
        )

        clus["Target_Visit"] = clus["Target_Visit"].astype(int)
        clus["Completed_Visit"] = clus["Completed_Visit"].astype(int)
        clus = clus.sort_values(["% Completion", "Pending Visits"], ascending=[False, False])

        clus_view = clus.rename(columns={"Target_Visit": "Target Visit", "Completed_Visit": "Completed Visit"})
        st.dataframe(clus_view, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">2) CAC leaderboard</div>', unsafe_allow_html=True)
        lb = cbc.groupby(["Employee Name", "Cluster Name"], dropna=False).agg(
            Target_Visit=("Targeted Visits", "sum"),
            Completed_Visit=("Completed", "sum"),
        ).reset_index()

        lb["% Completion"] = np.where(
            lb["Target_Visit"] > 0, (lb["Completed_Visit"] / lb["Target_Visit"]) * 100, np.nan
        )
        lb["Target_Visit"] = lb["Target_Visit"].astype(int)
        lb["Completed_Visit"] = lb["Completed_Visit"].astype(int)

        lb = lb.rename(columns={"Employee Name": "Mentor Name"})
        lb = lb.sort_values(["% Completion", "Completed_Visit"], ascending=[False, False])

        lb_view = lb[["Mentor Name", "Cluster Name", "Target_Visit", "Completed_Visit", "% Completion"]].copy()
        st.dataframe(lb_view, use_container_width=True, hide_index=True)

    # ✅ Mentor exception list (Block)
    st.divider()
    st.markdown('<div class="section-title">Mentor Exception List (Block)</div>', unsafe_allow_html=True)
    summary_b, exc_b = mentor_exception_table(block_all.assign(**{"Block Name": sel_b}), scope_label=f"{sel_d} | {sel_b}")

    kpi_row([
        ("Mentors not at 100%", f"{summary_b['mentors_not_100']:,}", ""),
        ("Total pending visits", f"{summary_b['pending_total']:,}", ""),
        ("Median % completion", f"{summary_b['median_pct']:.1f}%" if not np.isnan(summary_b["median_pct"]) else "—", ""),
        ("Worst % completion", f"{summary_b['worst_pct']:.1f}%" if not np.isnan(summary_b["worst_pct"]) else "—", ""),
    ])

    st.markdown(f"### 🚨 {sel_b}: Mentors NOT completing 100% target")
    st.caption("Sorted worst-first: lowest % completion, then highest pending visits.")
    st.dataframe(exc_b, use_container_width=True, hide_index=True)

    st.divider()
    block_xlsx = to_excel_bytes({
        "Cluster_Compliance": clus_view,
        "CAC_Leaderboard": lb_view,
        "Mentor_Exception_List": exc_b,
    })
    st.download_button(
        "⬇️ Download Block Report (Excel)",
        data=block_xlsx,
        file_name=f"{sel_d}_{sel_b}_block_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
