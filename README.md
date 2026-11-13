# Acoustic Energy Research: paper and reproducible analysis

**[Read the paper](Acoustic_Energy_Research.pdf).** This is the complete paper. The `code/` folder is for readers who want to inspect or rerun the calculations.

## What is included

- `Acoustic_Energy_Research.pdf`: the complete paper.
- `reproduce.py`: one command to run the analysis and checks.
- `requirements.txt` and `environment.json`: dependencies and the versions used for the study.
- `code/`: the model, seven analysis scripts and five test modules.
- `code/source_data/`: the published figure used for digitization, with attribution and source details.
- `code/extended_results/` and `code/revised_results/`: selected reference outputs, digitized readings, provenance and numerical checks. The package includes 12 reference/data files; running `reproduce.py` regenerates the complete set of 25 result/data files. Some files retain their original names so the scripts work without changes.
- `LICENSE`: the MIT License for the Python code, with scope explained below.

The package omits generated equation images, separate figure exports, duplicate result exports, editable manuscript copies and document-layout code. The scientific scripts regenerate all six research figures and the detailed result tables. Rebuilding the PDF's typesetting is outside this compact package.

## Reproduce the calculations

Use Python 3.12 and install the dependencies from the extracted package folder:

    python -m pip install -r requirements.txt
    python reproduce.py

The runner executes the analysis scripts in dependency order, then runs the 60-test suite and the capacitance checks. Source calibration is the slowest step. It writes progress to the terminal and detailed output to `code/run.log`.

Generated tables are written to `code/revised_results/` and `code/extended_results/`; all six figures are written to `code/revised_figures/`. Rerunning overwrites these generated results, including the bundled reference results. Keep an untouched extracted copy if you want to compare them. The supplied paper and source image are not modified.

For a quick check of the supplied reference results and model:

    python reproduce.py --check-only

`environment.json` records the versions actually used. Other dependency versions have not been verified. Scientific figures use Times New Roman when available, with a fallback font; a font difference can change appearance without changing the calculations.

## Code map

| File in `code/` | Purpose |
| --- | --- |
| `model.py` | Capacitor energy balance and charging relations |
| `revision_analysis.py` | Leakage assumptions, basic comparisons, Figures 1-2 |
| `empirical_analysis.py` | Published-curve digitization and fits, Figure 3 |
| `intermittent_analysis.py` | Equal-energy schedules and event checks, Figure 4 |
| `restart_analysis.py` | Restart costs and shutdown policies, Figure 6 |
| `component_analysis.py` | Component-based sensor budget |
| `robustness_analysis.py` | Threshold controls, source model, fit windows, acoustic budget and leakage range; Figure 5 |
| `tolerance_analysis.py` | Effective-capacitance sensitivity at 2.8 V and the 2.55 V energy boundary |
| `test_*.py` | 60 numerical checks |

## Interpretation and provenance

This is a computational study, not a demonstrated acoustic sensor. Scenario inputs are assumptions unless explicitly identified as published values. One digitized experimental trace is used; simulated outputs are separate from those readings. The 80-120 microfarad capacitance range and leakage sweeps are deterministic assumptions, not measured distributions. Voltage-source cases calibrate source resistance separately to match delivered electrical energy, not incident acoustic energy. See the paper for the full limitations and references.

The source figure's attribution and CC BY provenance are retained in `code/source_data/README.md`; digitization coordinates, tolerance and the image hash are in `code/extended_results/digitization_metadata.json`.

Public repository: [Acoustic Energy Research](https://github.com/muradqu5-dev/Acoustic-energy-study). No DOI has been assigned.

## License

The Python source files under `code/` and the top-level `reproduce.py` are provided under the MIT License in `LICENSE`, copyright 2026 Murad Guliyev. The license text follows the Open Source Initiative's MIT template: https://opensource.org/license/mit.

This code license does not relicense the research paper, result/data files or third-party source material. The Duan figure retains its existing CC BY terms and attribution in `code/source_data/README.md`.

## Author contribution and AI assistance

The author researched acoustic energy harvesting and searched for relevant published papers. ChatGPT (OpenAI) assisted with checking numerical values, developing and debugging the Python code, numerical analysis, and figure generation. Published charging readings were digitized from the cited source; scenario outputs are model calculations. Responsibility for evaluating the methods, verifying the results and approving the final manuscript remains with the author.
