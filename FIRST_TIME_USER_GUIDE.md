# TraceISO - First-Time User Guide

This guide is for users launching TraceISO from the repository folder.
It covers **Windows** (section A) and **macOS** (section B).

## 0. Getting TraceISO From GitHub

The TraceISO source code is available at https://github.com/sppranav/TraceISO.

Choose either method. Both give you the same files.

### Option 1: Download the ZIP (no extra software needed)

1. Open https://github.com/sppranav/TraceISO
2. Click the green **Code** button, then **Download ZIP**.
3. Extract the ZIP.
   - Windows: right-click the file, choose **Extract All**.
   - macOS: double-click the file.
4. You now have a folder named `TraceISO-main`. Everything below happens
   inside that folder.

Extract the ZIP before running anything. Windows can open a ZIP as if it were a
folder, but launchers started from inside a ZIP preview cannot create the `venv`
folder and will fail.

On macOS, if the `.command` files do not run after extracting, open Terminal in
the TraceISO folder and run this once:

```bash
chmod +x *.command
```

The macOS application launcher starts a new process on an available local port.
It skips occupied ports instead of opening an existing service. Stop the previous
session with **Control+C** before relaunching if you want only one instance.

### Option 2: Clone with git (best if you want updates later)

```bash
git clone https://github.com/sppranav/TraceISO.git
cd TraceISO
```

To update to the newest version afterwards:

```bash
git pull
```

Then run `00_Setup_Update_Dependencies` again, so any new dependency is
installed.

### Where to put the folder

Choose a normal working location such as Documents. Avoid:

- a path inside a ZIP file that was never extracted
- a synced cloud folder that may lock files while the app runs
  (OneDrive, iCloud Drive, Dropbox) — TraceISO writes into the folder
- a path containing characters your Python installation may not handle well

## 1. Before You Start (both platforms)

1. Keep the full TraceISO folder together.
2. Install **Python 3.10 or newer** if you do not have it
   (https://www.python.org/downloads/). This is the declared minimum; see
   [Python compatibility](README.md#python-compatibility)
   for dependency-pin limitations and optional desktop-tool requirements.
3. Do not move these folders out of the project:
   - `config`
   - `domain`
   - `file_io`
   - `ui`
   - `tools`
4. First run may take several minutes, and normally needs internet while
   dependencies are installed. Later runs work offline.
5. Windows Defender, antivirus, or firewall may ask for permission.
   Allow local access.

### In-app review flow

The Session Configuration ratio selector limits the ratios available in Sample
Inspector, Instrumental Drift, Results & Statistics, Uncertainty Budgets, and
Export. In Uncertainty Budgets, the **Budget review** subview shows the current
budgets; budget settings take effect immediately, while profile and per-sample
applicability edits require their Apply controls. Review Results & Statistics
again after changing uncertainty settings, then export.

Use the upload box’s **×** to remove the loaded file and clear its session data.
Your original file on disk is retained. **Reset to Original** keeps the dataset,
restores its initial sample edits, and discards processed results.

Sidebar search and type filters affect only the **Inspect samples** selector.
Its run/index labels distinguish duplicate names; selecting a sample there always
opens Sample Inspector. Loaded-file details start collapsed. Plot-layer,
threshold, outlier and statistics controls are in Sample Inspector beside their plots.
Each sample entry includes its type code and a distinct shape as well as colour.
At narrow browser widths or increased zoom, follow the navigation scroll cue to
reach later sections; the section controls remain keyboard accessible.

In Instrumental Drift, you can fit and review the drift model whether or not
drift correction is currently enabled: the fit, the before/after preview and the
statistics are all read-only until you commit. The review leads with the
before/after RSD statistics, with the model equation in an expander below.
When you are satisfied, **Apply drift correction and reprocess** (or **Commit
Drift Correction** if it is already enabled) re-runs the full reduction from the
tab — no trip back to Session Configuration. The button is available both in the
settings panel and after the review detail.

In Session Configuration, scroll the page normally: settings are grouped as
Data & cycles, Corrections and Reference material, custom ratios live in the
collapsed **Create custom ratio** expander, and **Execute Data Reduction** is
below both panels. In Sample Inspector, the cycle window and applied exclusions
appear beside the isotope and ratio selectors; "Pending" means staged exclusions
still need **Apply Exclusions**. Use **Discard pending changes** to restore the
staged list to the applied exclusions without changing masks; **Clear applied
exclusions** is a separate action. Export's advanced sheet options list each
worksheet name in parentheses. Panels stack on narrower windows.

For Sr data, **Calibrate against Sr standard** is off by default. Enable it and
select the calibration standard runs explicitly to calibrate 87Sr/86Sr against
the selected reference material. Sample Inspector keeps separate internally
normalized and Sr-standard-corrected traces. The calibrated layer flows into
Results, Uncertainty, Drift and Export; optional drift follows calibration.
Calibration uses the selected standards' cycle windows and exclusions when you
execute data reduction. If a selected standard is unusable, normalized results
remain available for inspection and the app explains why calibrated final
results are unavailable. Changing calibration settings or standard-cycle
selections requires executing data reduction again.

For Pb data, the Hg correction control is available for both ordinary SSB and
Pb–Tl external normalization when a 202Hg monitor is present. On SSB, TraceISO
uses usable Tl channels automatically for the Hg mass-bias factor and otherwise
uses the managed natural Hg ratio; incomplete or failed Tl is reported as
unavailable. Pb–Tl sessions can optionally calibrate final ratios against
explicitly assigned Pb standards using local brackets or a session mean of
individual K factors. This route keeps separate Tl-only and final traces and
disables separate drift application while retaining the saved drift request.
Calibrated absolute-ratio budgets and Monte Carlo are available when their inputs
are complete. Calibrated delta combined uncertainty and Monte Carlo are shown as
**Not calculated**; any displayed SD or SE is cycle scatter/precision only.
Every configured Pb/Pb ratio, including a custom ratio, follows the same Tl
normalization and optional calibration path. If its calibration reference cannot
be resolved, the Tl-normalized trace remains available and the calibration reason
is stated explicitly.

For Sr internal normalization, Excel automatically uses the Sr scientific
report. **Summary** and **Results** are always present. **Include uncertainty
detail** adds Budget_Detail (on by default); turning it off keeps uncertainty
in Results. **Cycle_Data workbook view** adds one combined cycle table (off by
default) containing every recorded observation and series, including blanks,
unselected ratios and excluded values. Calibration and drift stages remain
distinct. Reprocess after changing a calibration standard's accepted cycles.

For Pb with external Tl normalization, Excel offers **Include uncertainty detail**
(on by default) and **Include cycle data** (off by default). Summary and Results
are always included. Optional Pb-standard calibration appears immediately after
the Tl-normalized stage, with explicit roles and unavailable-result reasons.
Calibrated delta shows the selected SD/SE precision and **Not calculated** for
combined uncertainty. Reprocess when a calibration-standard or other calibration
dependency change makes the session stale. The cycle sheet preserves all recorded
observations and series, including unselected ratios and rejected cycles.

In an Engine-B session with SSB enabled, Excel automatically uses the compact
SSB scientific report. **Summary** and **Results** are always included;
**Budget_Detail** is selected by default, and the optional **Cycle_Data** view is
one combined sheet containing every observation, including blanks, with each
series' accepted/excluded status. In other workflows, choose the Excel
**Workbook layout**: *Diagnostic* keeps the familiar
workbook led by the Results sheet; *Analyst report* adds **Reported Results**, one
row per sample, ratio and output mode with its U and Coverage k, and keeps Results
as an optional diagnostic. Contributor-level budgets (Uncertainty_Budgets) and the
optional wide view (Budget_Detail) describe the same budgets. Contributor Profiles
lists the profiles your samples use unless you tick the complete library for audit.
Per-sample audit cycle sheets are not written for blanks. For every format, choose
the options and select **Prepare report**; the download appears only while that
prepared file still matches the current session. For CSV, choose
**Complete audit CSV** or **Compact CSV** independently of **All selected ratios**
or **One ratio**, then use the single download action. Complete audit carries the
full provenance on every row; Compact keeps the same results without those repeated
records. In the Results summary table, U is shown
beside its Coverage k and u_c is the combined standard uncertainty.
Results leads with delta when delta reporting is active and otherwise with the
ratio; the aligned ratio comparison and intensity diagnostics are optional.
Results plots start compact; use **Enlarged export view** to inspect and download
the publication-styled PNG or SVG. Long sessions initially plot by run order and
show sample names on hover. In Sample Inspector, **Reset ratio view** restores
the full plot range without altering the cycle window or exclusions.

---

# A. Windows

Use the `.bat` files. The `.command` files are for macOS and can be ignored.

## A1. First Launch

1. Double-click `00_Setup_Update_Dependencies.bat` and wait for
   `Setup completed successfully.` This step is required once — the launcher
   does **not** install anything by itself.
2. Double-click `01_TraceISO_Launcher.bat`.
3. A window named **TraceISO Setup + Launcher** should open.
4. Click **Launch TraceISO** if it does not start automatically.
5. Your browser should open to the local TraceISO page.

If the browser does not open:

1. Click **Open in Browser** in the setup window.
2. Or open one of these manually:
   - `http://localhost:8501`
   - `http://localhost:8502`
   - `http://localhost:8503`

## A2. What The Launcher Does

`00_Setup_Update_Dependencies.bat`:

1. finds Python
2. creates the `venv` folder
3. installs the application and GUI dependencies

`01_TraceISO_Launcher.bat`:

1. checks that `venv` exists and has the GUI dependency
2. opens the TraceISO setup/launch window
3. sends you back to `00_Setup_Update_Dependencies.bat` if setup is incomplete

The main Streamlit app then runs from that launcher window.

## A3. Daily Use

1. Double-click `01_TraceISO_Launcher.bat`.
2. Click **Launch TraceISO** if needed.
3. Work in the browser once the local app opens.

## A4. The Desktop Tools

The other `.bat` files open the standalone desktop tools:

- `02_CRM_Library_Manager_Launcher.bat`
- `03_Neptune_Data_Extractor.bat`
- `05_Global_Uncertainty_Manager_Launcher.bat`

If the Global Uncertainty Manager reports that setup is incomplete or its GUI dependency is unavailable, run `00_Setup_Update_Dependencies.bat`, wait for successful completion, then launch it again.

They can also be opened from within TraceISO's sidebar.

## A5. Safe Troubleshooting (Windows)

### A. The setup window opens but the app page does not appear

1. Click **Launch TraceISO**.
2. Click **Open in Browser**.
3. Try the local URLs manually (8501, 8502, 8503).

### B. The launcher says Python was not found

1. Install Python 3.10 or newer.
2. Enable **Add Python to PATH** during installation.
3. Run `00_Setup_Update_Dependencies.bat` again.

### C. The launcher says TraceISO is not set up yet

Run `00_Setup_Update_Dependencies.bat`, wait for it to finish, then launch again.

### D. Setup or dependency repair keeps failing

1. Close all TraceISO windows.
2. Delete the local `venv` folder in the TraceISO project directory.
3. Run `00_Setup_Update_Dependencies.bat` again.

### E. A local port is already in use

Close old TraceISO windows and try again.
The launcher can use another local port if needed.

---

# B. macOS

Use the `.command` files. The `.bat` files are Windows-only and can be ignored.
No coding is required: these are double-clickable files, the same as on Windows.

## B1. First Launch

1. Double-click `00_Setup_Update_Dependencies.command`.
   A Terminal window opens and installs everything. Wait for
   `Setup completed successfully.`
2. Double-click `01_TraceISO_Launcher.command`.
3. Your browser opens to the local TraceISO page.
4. Leave the Terminal window open while you work.

If macOS refuses to open the file because it is from an unidentified developer:

1. Right-click (or Control-click) the file.
2. Choose **Open**.
3. Choose **Open** again in the dialog.

This is only needed the first time for each file.

If macOS says the file cannot be executed, open Terminal in the TraceISO folder
and run this once:

```bash
chmod +x 00_Setup_Update_Dependencies.command 01_TraceISO_Launcher.command
```

## B2. What The Launcher Does

`00_Setup_Update_Dependencies.command`:

1. finds Python 3.10 or newer
2. creates the `venv` folder
3. installs the application dependencies
4. installs the optional PyQt5 desktop tools if a wheel is available

`01_TraceISO_Launcher.command`:

1. checks that setup has been done
2. picks a free local port (8501, 8502, or 8503)
3. starts TraceISO and opens your browser
4. reuses an already running TraceISO instead of starting a second copy

Unlike Windows, macOS has no separate setup window: TraceISO runs directly in
the Terminal window that the launcher opens.

## B3. Daily Use

1. Double-click `01_TraceISO_Launcher.command`.
2. Work in the browser once the local app opens.
3. To stop TraceISO, press **Control-C** in the Terminal window, or close it.

## B4. Using TraceISO From The Terminal (optional)

The double-click files are the recommended route. If you prefer the Terminal,
the equivalent commands are:

```bash
cd /path/to/TraceISO
python3 -m venv venv                       # first time only
source venv/bin/activate                   # first time only
python -m pip install --upgrade pip        # first time only
python -m pip install -r requirements.txt  # first time only
streamlit run TraceISO.py
```

For later sessions only the last two lines are needed:

```bash
cd /path/to/TraceISO && source venv/bin/activate
streamlit run TraceISO.py
```

## B5. The Desktop Tools On macOS

The standalone tools need PyQt5, which `00_Setup_Update_Dependencies.command`
installs when a wheel is available for your Mac.

Each tool has its own double-clickable launcher, matching the Windows `.bat`
files one for one:

- `02_CRM_Library_Manager_Launcher.command`
- `03_Neptune_Data_Extractor.command`
- `05_Global_Uncertainty_Manager_Launcher.command`

The tool opens in its own window and detaches from the Terminal, so you can
close the Terminal window it came from. They can also be opened from TraceISO's
sidebar while the app is running.

From Terminal, the equivalent commands are:

```bash
source venv/bin/activate
python -m tools.crm_manager                  # CRM Library Manager
python -m tools.neptune_data_extractor.main  # Neptune Data Extractor
python -m tools.global_uncertainty_manager   # Global Uncertainty Manager
```

If a tool fails to start, its startup log is written to
`$TMPDIR/traceiso_<tool>.log`.

If setup reported that PyQt5 could not be installed, TraceISO itself can still
be used in the browser; these separate tools require PyQt5.
Check the setup log for the installation failure. See
[Python compatibility](README.md#python-compatibility)
for dependency-pin limitations; no specific Python version is recommended
as a verified desktop-tool environment.

## B6. Safe Troubleshooting (macOS)

### A. Setup says Python 3.10 or newer was not found

The Python included with macOS is too old. Install a current version from
https://www.python.org/downloads/, then run
`00_Setup_Update_Dependencies.command` again.

### B. The browser does not open

Open one of these manually:

- `http://localhost:8501`
- `http://localhost:8502`
- `http://localhost:8503`

Use the port printed in the Terminal window.

### C. The launcher says TraceISO is not set up yet

Run `00_Setup_Update_Dependencies.command`, wait for it to finish, then launch
again.

### D. Setup keeps failing

1. Close all TraceISO windows.
2. Delete the `venv` folder in the TraceISO project directory.
3. Run `00_Setup_Update_Dependencies.command` again.

### E. A local port is already in use

Close old TraceISO Terminal windows and try again.
The launcher picks another local port if needed.

---

# C. If You Need To Send A Problem Report

Enable **Advanced → Developer mode**, reproduce the issue, then open
**Developer diagnostics** in the sidebar and select **Download diagnostic report**.
The JSON contains settings, versions, captured errors, performance timings and the
latest GUM/MC debug reports you viewed. It may include file/sample names and error
paths; inspect it before sharing. Timings are recorded only while the mode is on.
The panel shows runtime cache hits/misses; nested timings overlap. **Reset diagnostic
history** starts a fresh measurement history without clearing your analysis.

Please include:

1. your operating system (Windows or macOS) and Python version
2. a screenshot of the error, or the text from the Terminal / setup window
3. whether you used the ZIP download or `git clone`
4. whether this was first launch or later use
5. whether setup completed successfully
6. whether the browser opened

### Desktop tool workflows

All three desktop tools—CRM Library Manager, Neptune Data Extractor, and Global
Uncertainty Manager—use one TraceISO light theme: white panels, muted teal primary
actions, consistent fonts and tables, visible keyboard focus, and a shared window
icon/title. Their workflow-specific layouts are retained. Reopen a tool to load the
updated style. Larger tables support scrolling rather than squeezing every column.

In CRM Manager, **Apply to Library** validates the current form in memory;
**Save Library to Disk / Ctrl+S** includes the current form and persists the library.
Pending edits are protected when navigating or closing. **Replace Library from JSON**
previews record and element-setting differences and shows the destination before
replacing the in-memory library. Search filters names/elements; validation issues
open the affected record. Derived Ratios includes a read-only standard uncertainty
where its semantics permit conversion. Blank uncertainty remains different from zero.

In Neptune Data Extractor, choose a built-in template or Custom Mapping, use the
visible mapping controls, and undo/redo mapping changes with Ctrl+Z/Ctrl+Shift+Z.
File/template changes start a new history. Sample-naming presets preview filename
rules; mapped Sample Name cells can override them. **Convert All N Files** and
**Convert Selected** explicitly choose the batch scope. Cancel works during validation
and conversion. **Results** retains per-file statuses and complete diagnostics, with
copy/export and Open Output Folder. **Retry Failed Files** uses the original mapping
and requires a new output file. Reader verification failures are labelled separately
from conversion; closing during work requests cancellation and keeps the window
open until the worker has stopped, then it can be closed again.

Windows desktop launchers preserve the local `venv` and supervise startup. Import,
initialization, and native Qt failures are reported with an **Open Log** action when
the fallback GUI is available. Logs live under `%LOCALAPPDATA%/TraceISO/logs`
(or a temporary writable location), with bounded rotation and ten runs retained per
tool. A launch that does not signal main-window readiness within 60 seconds is stopped
and reported. The GUI launcher also supervises the main application and Global
Uncertainty Manager. Keep that launcher open while using the children it starts;
closing it stops them. Its per-launch runtime-log path appears in the launch log.
Shutdown and timeout also stop descendants started by those children. Other
TraceISO sessions launched separately remain outside that launcher's ownership.
