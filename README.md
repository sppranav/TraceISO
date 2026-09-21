# TraceISO

TraceISO is an open-source application for processing solution-mode MC-ICP-MS isotope-ratio measurements. Import your data, select corrections, review results and uncertainty budgets, and export reports from a browser-based interface.

**Supported isotope systems:** Li, B, Mg, Sr, Cd and Pb.

---

## Key features

- Six isotope systems: Li, B, Mg, Sr, Cd and Pb
- SSB, Sr internal normalisation and Pb–Tl external normalisation
- Cycle-level inspection and filtering
- Configurable GUM-based uncertainty budgets
- Optional Monte Carlo uncertainty cross-check
- Excel, CSV and JSON export with processing records

---

## Quick Start

### Before you begin

1. Install **Python 3.10 or newer** (the declared minimum; see [Python compatibility](#python-compatibility)).
2. Download the source toolkit from the repository or clone it:

```bash
git clone https://github.com/sppranav/TraceISO.git
cd TraceISO
```

Alternatively, select **Code → Download ZIP** in the repository.
Extract the ZIP completely and open the extracted TraceISO folder. Do not run
TraceISO from inside the ZIP preview.

### Python compatibility

TraceISO requires **Python 3.10 or newer**. The application is designed for
Python 3.10–3.13. Compatibility may depend on the availability of binary
packages for the selected operating system and Python version.

The main browser-based TraceISO application does not require PyQt5. The optional
desktop tools require PyQt5, whose availability may vary by Python version and
platform.

`requirements.txt` specifies dependency ranges for normal installation.
`requirements-lock.txt` contains exact pins for the listed application
dependencies, but a successful release test run using those pins has not been
established. It is not a fully resolved environment lock: transitive dependencies
and optional desktop packages are not pinned there. Do not treat it as a
verified release environment.

### Recommended launch method

No terminal commands are required for the standard setup:

- **Windows:** double-click `00_Setup_Update_Dependencies.bat`. After setup finishes, double-click `01_TraceISO_Launcher.bat`.
- **macOS:** double-click `00_Setup_Update_Dependencies.command`. After setup finishes, double-click `01_TraceISO_Launcher.command`.

If macOS reports that a `.command` file is not executable after ZIP extraction,
open Terminal in the TraceISO folder and run `chmod +x *.command` once.

The setup launcher creates a separate Python environment in the `venv` folder and installs the required packages. The application launcher starts TraceISO and opens it in your browser, normally at `http://localhost:8501`.

On macOS, each launch starts a new TraceISO process on an available local port.
If port 8501 is occupied, the launcher selects another port. Close the previous
session with **Control+C** before launching again if you want only one instance.

### Terminal setup (optional)

From a terminal opened in the TraceISO folder, create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install the dependencies and launch TraceISO:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run TraceISO.py
```

For later sessions, activate the existing virtual environment and rerun only:

```bash
python -m streamlit run TraceISO.py
```

Keep the terminal or launcher window open while using TraceISO. Stop a terminal session with **Ctrl+C**.

More detailed platform-specific instructions and troubleshooting are provided in [FIRST_TIME_USER_GUIDE.md](FIRST_TIME_USER_GUIDE.md).

TraceISO runs locally on your computer. The main application opens in your browser; the optional tools open in separate desktop windows.

---

## Scope

**Current release: TraceISO v1.0.0**

TraceISO v1.0.0 supports solution-mode MC-ICP-MS measurements for Li, B, Mg, Sr, Cd and Pb. Laser-ablation and other transient-signal workflows are outside the scope of the current release.

---

## Workflow

1. Upload an HDF5 file in the sidebar.
2. Classify samples and configure processing in **Session Configuration**, then run data reduction.
3. Adjust cycle windows and manual exclusions in **Sample Inspector**.
4. Review drift in **Instrumental Drift**. To apply a correction, select **Apply drift correction and reprocess** (optional).
5. Review summary tables and QC plots in **Results & Statistics**.
6. Configure uncertainty, then use **Budget review** in **Uncertainty Budgets**;
   revisit **Results & Statistics** after uncertainty-setting changes.
7. In **Export**, choose the format/options, select **Prepare report**, then download the ready file.

Changing cycle windows or exclusions in Sample Inspector updates the displayed statistics and uncertainty budgets. To repeat the complete processing sequence, use **Execute Data Reduction** or **Apply drift correction and reprocess**. Reprocess after changing correction settings or reference values.

---

## Processing Flows

- **Li, B, Mg, Cd and Pb standard-sample bracketing (SSB):** blank correction, cycle filtering, optional drift correction, and comparison with bracketing standards. Results can be reported as ratios or delta values. Pb also supports mercury-interference correction.
- **Sr:** blank and Rb/Kr interference corrections, internal normalisation, optional calibration against selected Sr standards, and optional drift correction.
- **Pb–Tl:** blank correction, optional mercury-interference correction, Tl-based external normalisation, and optional calibration against selected Pb standards. Separate drift correction is not applied when Pb-standard calibration is used.


---

## Input Data

Upload cycle-resolved measurements in TraceISO's HDF5 format. The application identifies the isotope system from the recorded isotope channels.

Use the **Neptune Data Extractor** to convert supported instrument exports with a built-in template or custom column mapping.

---

## Uncertainty Evaluation

Configure uncertainty contributions to match your measurement method. TraceISO combines them into a GUM-based uncertainty budget and reports expanded uncertainty **U** with its coverage factor **k**.

An optional **Monte Carlo cross-check** compares simulated uncertainty with the analytical calculation. It does not independently validate the complete measurement or processing method.

For isotope-amount-ratio reporting using SSB, the budget includes the uncertainty of the certified or assigned reference isotope ratio. This contribution cancels for delta values and is therefore excluded from delta budgets.

---

## Reference Data

Use **CRM Manager** to view and edit certified or assigned reference values, isotope masses and natural ratios. Check that the selected reference material and its values are appropriate for your measurements.

---

## Exports

| Format                  | What you get                                                                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excel (.xlsx)** | Formatted reports with summary results and optional uncertainty details and cycle data.                                                        |
| **CSV**           | One row per sample and ratio. Choose **Complete audit CSV** for detailed processing records or **Compact CSV** for a smaller table. |
| **JSON**          | Structured results and processing records, with optional cycle data and uncertainty budgets.                                                   |

**Li, B, Mg, Cd and Pb SSB workflows share the same Excel structure.** Delta-only workflows using the SSB/delta uncertainty engine also use this report, including Cd with only **Report delta values** selected:

- **Summary** — session and processing information.
- **Results** — isotope ratios, delta values and available uncertainties.
- **Budget_Detail** — uncertainty contributions; enabled by default when suitable budgets are available.
- **Cycle_Data** — one combined sheet of cycle measurements; optional and off by default.

Columns adapt to the selected isotope ratios and corrections. Sr internal-normalisation and Pb–Tl workflows use dedicated reports with the same main sheet names. Other processing modes offer **Diagnostic** and **Analyst report** layouts.

Exports reflect the applied cycle windows and exclusions. **U** is expanded uncertainty and **k** is its coverage factor; **Delta 2SE** describes precision and is a separate quantity.

Excel downloads include saved Monte Carlo results when available, but omit the separate Provenance, Uncertainty Scope and Long Payloads sheets. Use JSON when you need complete processing records, including large records omitted from Excel.

---

## Standalone Tools

| Tool                                 | Purpose                                    |
| ------------------------------------ | ------------------------------------------ |
| **CRM Manager**                | View and edit reference-material values.   |
| **Neptune Data Extractor**     | Convert supported instrument data to HDF5. |
| **Global Uncertainty Manager** | Manage shared uncertainty settings.        |
| **Desktop Launcher**           | Open the application and its tools.        |

Run the setup script before opening the numbered tool launchers. For a manual installation, install the optional desktop packages from the TraceISO folder:

```bash
python -m pip install -e ".[tools]"
```

---

## Citation

If you use TraceISO in published work, please cite the associated TraceISO
publication and the software release used in your analysis. Citation details
for the publication will be added after publication.

---

## License

TraceISO is distributed under the MIT License. See [LICENSE](LICENSE).
