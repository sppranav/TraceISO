# TraceISO Test Data

Ten MC-ICP-MS measurement files for trying TraceISO without your own data.
All files listed below load directly in the application's HDF5 uploader.
Sequence counts and channel lists were checked with TraceISO's HDF5 reader on
2026-09-21; all ten loaded without reader warnings. This is an input-loading
check, not certification of the measurements or their processed results.

| File | System | Sequences | Recorded isotope channels |
| --- | --- | ---: | --- |
| `B-RMTest09-10-14_20141009-114513.h5` | B | 94 | 10B, 11B |
| `Boron_IAPSO_NASS6_Metropoem__20250410-171928_Uncorrected_Data_V2.h5` | B | 91 | 10B, 11B |
| `Cd-Stds_20231110-120209_Uncorrected_Data.h5` | Cd | 21 | 106Cd, 110Cd, 111Cd, 112Cd, 113Cd, 114Cd, 116Cd, 117Sn |
| `Li_Data_eg.h5` | Li | 169 | 6Li, 7Li |
| `Mg-delta-1-_20160112-152031.h5` | Mg | 159 | 24Mg, 25Mg, 26Mg |
| `Mg_Data_eg.h5` | Mg | 12 | 24Mg, 25Mg, 26Mg |
| `Pb-_20240315-160902.h5` | Pb | 26 | 202Hg, 204Pb, 206Pb, 207Pb, 208Pb |
| `Pb-Tl-Referenzvergleich.h5` | Pb | 38 | 203Tl, 204Pb, 205Tl, 206Pb, 207Pb, 208Pb |
| `Uncorrected_Data_Li-MR-Pra_20240216-124219_through_033.h5` | Li | 33 | 6Li, 7Li |
| `Uncorrected_Data_Sr-ILC_20211130-173900.h5` | Sr | 96 | 82Kr, 83Kr, 84Sr, 85Rb, 86Sr, 87Sr, 88Sr |

## How to use

1. Start TraceISO (`01_TraceISO_Launcher.bat` on Windows,
   `01_TraceISO_Launcher.command` on macOS).
2. Upload `Mg_Data_eg.h5` for a short, 12-sequence starting example.
3. Review sample classifications, reference values and correction settings
   before executing data reduction.
4. Inspect the results and uncertainty settings before preparing an export.

The two Li files have different sequence counts: `Li_Data_eg.h5` contains 169,
while the file ending in `_through_033.h5` contains 33.

The Pb files offer different channels: `Pb-_20240315-160902.h5` includes a
202Hg monitor, while `Pb-Tl-Referenzvergleich.h5` includes 203Tl and 205Tl.
Choose corrections appropriate to the available channels and measurement method.
