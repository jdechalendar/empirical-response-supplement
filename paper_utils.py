import pathlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import pandas as pd
import numpy as np
import statsmodels.api as sm


# USER INPUTS ############################
base_path = pathlib.Path(__file__).parent.absolute()
data_path = base_path / "data"
save_fig = True
##########################################


fig_path = base_path / "figures" / "main"
si_path = base_path / "figures" / "si"


if save_fig:
    fig_path.mkdir(exist_ok=True, parents=True)
    si_path.mkdir(exist_ok=True)

PROJECTS = ["CONF-1", "OFF-2", "LIB-3", "OFF-4", "LAB-5", "LAB-6"]
FILE_NAME_COOLING = data_path / "building_level_cooling.csv"

# Units
sqft_to_sqm = 0.092903  # m^2/ft^2
tonhr_to_MJ = 1.0 / 0.07898476
kW_to_ton = 0.284345
MJ_to_kWh = 1 / 3.6


def cop_to_kWe_per_MJc(cop):
    """kWh of electricity per MJ of cooling"""
    return 1 / cop / kW_to_ton / tonhr_to_MJ


PAGE_WIDTH = 7.22  # in
ROW_HEIGHT = 2.0  # in
COLORS = sns.color_palette("colorblind")

MY_CMAP = ListedColormap([COLORS[0], COLORS[3]])

dark_grey = np.array([66, 66, 66]) / 256
light_grey = np.array([115, 115, 115]) / 256

sage = np.array([210, 194, 149]) / 256
custom_blue = np.array([0, 152, 219]) / 256
custom_green = np.array([0, 155, 118]) / 256
custom_orange = np.array([233, 131, 0]) / 256
cardinal_red = np.array([140, 21, 21]) / 255
grey1 = np.array([210, 210, 210]) / 256
grey2 = np.array([150, 150, 150]) / 256
MY_CMAP2 = ListedColormap([dark_grey, custom_blue, custom_green])
MY_CMAP4 = ListedColormap(COLORS[0:4])

plot_numbers = "ABCDEFGHIJ"

# Floorspace data in 1000 m^2
floorspace = (
    pd.Series(
        {  # kft^2
            "LIB-3": 170.0,
            "OFF-2": 28.0,
            "OFF-4": 105.0,
            "CONF-1": 145.0,
            "LAB-5": 75.0,
            "LAB-6": 77.0,
        }
    )
    * sqft_to_sqm
)  # 1000 m^2

choose_weekend = {
    "CONF-1": False,
    "OFF-2": True,
    "LIB-3": False,
    "OFF-4": True,
    "LAB-5": True,
    "LAB-6": True,
}

cooling_ylabels = {
    "ktonhr": "Served cooling (kton-hr/day)",
    "tonhr_per_m2": "Served cooling\n(ton-hr/m$^2$/day)",
    "MJ_per_m2": "Served cooling\n(MJ/m$^2$/day)",
    "kWhe_per_m2": "Electric load for cooling\n(kWh/m$^2$/day)",
}


param_renamer = {
    "Treated": "SP (Treatment)",
    "Tmean": "OAT",
    "const": "Intercept",
    "Weekend": "Weekend",
}


def load(project):
    return pd.read_csv(data_path / f"{project}.csv", parse_dates=True, index_col=0)


def set_plots():
    """
    Set up custom plotting style.

    Returns
    -------
    COLORS
    PAGE_WIDTH
    ROW_HEIGHT
    """
    pd.plotting.register_matplotlib_converters()
    plt.rcParams.update(
        {
            "figure.figsize": [PAGE_WIDTH, ROW_HEIGHT],
            "axes.grid": True,
            "font.size": 6,
            "axes.prop_cycle": cycler("color", COLORS),
            "grid.linewidth": 0.2,
            "grid.color": "grey",
            "figure.dpi": 300,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1,
            "savefig.transparent": True,
            "legend.fontsize": 5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.markerscale": 3.0,
        }
    )


def plot_base(
    df_,
    x="Tmean",
    y="cooling",
    c="scheduled_sp",
    ylabel=cooling_ylabels["ktonhr"],
    xlabel=r"Mean Daily Temp ($\degree$F)",
    clabel=r"Cool SP ($\degree$F)",
    cmap=MY_CMAP,
    fax=None,
    weekend=True,
    s=7,
    with_cbar=False,
    cooling_load_unit=None,
):
    if cooling_load_unit is not None:
        ylabel = cooling_ylabels[cooling_load_unit]
    if fax is None:
        f, ax = plt.subplots(1, 1, figsize=(PAGE_WIDTH * 0.3, ROW_HEIGHT / 2))
    else:
        f, ax = fax

    df = df_.loc[:, [x, y, c]].copy(deep=True)

    weekday = df.index.weekday.isin(range(5))
    if not weekend:
        df = df.loc[weekday, :]
        weekday = df.index.weekday.isin(range(5))

    sc = ax.scatter(
        df.loc[weekday, x], df.loc[weekday, y], c=df.loc[weekday, c], s=s, cmap=cmap
    )

    if with_cbar:
        cbar = f.colorbar(sc, ax=ax)
        cbar.ax.set_title(clabel, fontsize="medium")

    if weekend:
        sc = ax.scatter(
            df.loc[~weekday, x],
            df.loc[~weekday, y],
            c=df.loc[~weekday, c],
            marker="x",
            s=s,
            cmap=cmap,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if fax is None:
        f.tight_layout()

    return f, ax


def _form_y_X(df_, x_vars, y_var, treated_var, log, weekend, interact):
    """Helper function to form X and y for statsmodels fit and predict methods"""
    weekday = df_.index.weekday.isin(range(5))
    if not weekend:
        df = df_.loc[weekday, x_vars + [y_var, treated_var]].copy(deep=True)
    else:
        df = df_.loc[:, x_vars + [y_var, treated_var]].copy(deep=True)

    x_vars = {x: df[x] for x in x_vars}
    x_vars["Treated"] = df[treated_var]
    if weekend:
        x_vars["Weekend"] = ~weekday
    if interact:
        x_vars["Interact"] = x_vars["Treated"] * x_vars["Tmean"]
    X = pd.DataFrame(x_vars).astype(float)
    X = sm.add_constant(X)
    if log:
        return np.log(df[y_var]), X
    else:
        return df[y_var], X


def y_transform(x, log):
    if log:
        return np.exp(x)
    else:
        return x


def _plot_model(fax, modres, xvals, log, weekend, interact, ls):
    f, ax = fax
    ax.plot(
        xvals,
        y_transform(xvals * modres.params["Tmean"] + modres.params.const, log=log),
        color=MY_CMAP(0),
        ls=ls[0],
    )
    yvals = xvals * modres.params["Tmean"] + modres.params.Treated + modres.params.const
    if interact:
        yvals += xvals * modres.params["Interact"]
    ax.plot(xvals, y_transform(yvals, log=log), color=MY_CMAP(1), ls=ls[0])
    if weekend:
        ax.plot(
            xvals,
            y_transform(
                xvals * modres.params["Tmean"]
                + modres.params.const
                + modres.params.Weekend,
                log=log,
            ),
            color=MY_CMAP(0),
            ls=ls[1],
        )
        yvals = (
            xvals * modres.params["Tmean"]
            + modres.params.Treated
            + modres.params.const
            + modres.params.Weekend
        )
        if interact:
            yvals += xvals * modres.params["Interact"]
        ax.plot(
            xvals,
            y_transform(yvals, log=log),
            color=MY_CMAP(1),
            ls=ls[1],
        )


def create_model(
    df_,
    x_variables=["Tmean"],
    y="cooling",
    treated="scheduled_sp_bin",
    log=True,
    weekend=True,
    fax=None,
    interact=False,
    ls=["-", "--"],
):
    if (fax is not None) and not (x_variables == ["Tmean"]):
        raise ValueError("I don't know how to make this plot")
    if interact and not (x_variables == ["Tmean"]):
        raise ValueError("Only supporting this with Tmean")

    y, X = _form_y_X(df_, x_variables, y, treated, log, weekend, interact)
    mod = sm.OLS(y, X)
    fii = mod.fit()

    if fax is not None:
        xvals = np.linspace(X.Tmean.min(), X.Tmean.max(), 100)
        _plot_model(fax, fii, xvals, log, weekend, interact, ls)
    return fii


def predict(
    modelres,
    df_,
    x_variables=["Tmean"],
    y="cooling",
    treated="scheduled_sp_bin",
    log=True,
    weekend=True,
    interact=False,
):
    """
    Returns ytrue, ypred
    """
    ytrue, X = _form_y_X(df_, x_variables, y, treated, log, weekend, interact)
    return y_transform(ytrue, log=log), y_transform(modelres.predict(X), log=log)


def transform_coeff(y, log):
    if log:
        return 100 * (np.exp(y) - 1)
    else:
        return y


def ci_plot(axis, x, mod, var, color, ms=2, alpha=1, lw=1, log=True, verb=False):
    """Plot coefficient data from a model"""
    y = transform_coeff(mod.params[var], log=log)
    ci = transform_coeff(
        mod.summary2().tables[1].loc[var, ["0.975]", "[0.025"]], log=log
    ).values
    axis.plot(
        [x, x],
        ci,
        color=color,
        alpha=alpha,
        lw=lw,
    )
    axis.plot([x], [y], color=color, marker="o", ms=ms)
    if verb:
        print(f"{var}: {y:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")


def hourly_table(s_in):
    s = s_in.copy(deep=True)
    if type(s_in) == pd.Series:
        s = s.to_frame()
    cols = s.columns
    # special case: we don't want to end up with a hierarchical index
    if len(cols) == 1:
        cols = cols[0]
    s["Date"] = s.index.date
    s["hour"] = s.index.hour
    s = s.pivot(index="Date", values=cols, columns="hour")
    s.index = pd.to_datetime(s.index)
    return s


def add_legend(fax, ncol=4, bbox_to_anchor=(0.5, 0.01), bottom=0.14):
    f, ax = fax
    ax[-1].plot([], [], "o", color=MY_CMAP(0), label=r"Cooling SP: 74 $\degree$F", ms=2)
    ax[-1].plot([], [], "o", color=MY_CMAP(1), label=r"Cooling SP: 76 $\degree$F", ms=2)
    ax[-1].plot([], [], "o", color=dark_grey, label="Weekday", ms=2)
    ax[-1].plot([], [], "x", color=dark_grey, label="Weekend", ms=2)
    plt.subplots_adjust(bottom=bottom)
    ax[-1].legend(
        loc=8, bbox_transform=f.transFigure, bbox_to_anchor=bbox_to_anchor, ncol=ncol
    )


def _table(models_, projects_):
    text = "   "
    for p in projects_:
        text += f" & \\multicolumn{{3}}{{|c|}}{{{p}}}"
    text += "\\\\\n"
    text += "\\hline\n"
    text += "& Value & Std. Err & P-Value & Value & Std. Err & P-Value &  Value & Std. Err & P-Value \\\\"
    text += "\n\\hline\n"

    for param in ["Treated", "Tmean", "Weekend", "const"]:
        line = param_renamer[param] + " & "
        for p in projects_:
            tt = models_[p].summary2().tables[1][["Coef.", "Std.Err.", "P>|t|"]]
            if param in tt.index:
                line += " & ".join(list(tt.loc[param].map("{:.2g}".format).values))
            else:
                line += " & ".join(["-", "-", "-"])
            line += " & "
        text += line[:-2] + "\\\\\n"
        text += "\\hline\n"
    return text


def table2(models_, projects_):
    text = "\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n"
    text += "\\hline\n"
    text += _table(models_, projects_[0:3])
    text += "\\hline\n"
    text += _table(models_, projects_[3:])
    text += "\\end{tabular}\n"
    return text


def table1(models_, projects_):
    nobs = [models_[p].summary2().tables[0].loc[3, 1] for p in projects_]
    R2 = [models_[p].summary2().tables[0].loc[6, 1] for p in projects_]
    PF = [models_[p].summary2().tables[0].loc[5, 3] for p in projects_]
    text = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    text += "\\hline\n"
    text += " & " + " & ".join([p for p in projects_]) + "\\\\\n"
    text += "\\hline\n"
    text += "Number of observations & " + " & ".join(nobs) + "\\\\\n"
    text += "\\hline\n"
    text += "R$^2$ & " + " & ".join(R2) + "\\\\\n"
    text += "\\hline\n"
    text += "Prob (F-statistic) & " + " & ".join(PF) + "\\\\\n"
    text += "\\hline\n"
    text += "\\end{tabular}\n"
    return text


def si_table(models_, projects_, caption="", label=""):
    text = "\\begin{table*}[t]\n"
    text += "\\centering\n"
    text += table1(models_, projects_)
    text += table2(models_, projects_)
    text += f"\\caption{{{caption}}}\n"
    text += f"\\label{{{label}}}\n"
    text += "\\end{table*}\n"
    return text


def add_weekend(ax, x1, x2, color=dark_grey):
    dates = pd.date_range(x1, x2, freq="D")
    dates = dates[~dates.weekday.isin(range(5))]
    for dt in dates:
        ax.axvspan(
            dt,
            dt + pd.Timedelta("24h"),
            color=color,
            alpha=0.1,
            ec="w",
        )
