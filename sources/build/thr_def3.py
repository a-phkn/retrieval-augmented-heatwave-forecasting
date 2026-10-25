# -*- coding: utf-8 -*-
from thr_def import T
NI="NIOSH_2016_106_HEAT"; SP="springer_paper"
W="NIOSH WBGT (°C-WBGT), US occupational guidance; not an Indian standard. M = metabolic rate in watts, for a 70 kg 'standard man'."
THR3=[
T("NIOSH RAL (unacclimatized workers)","RAL [°C-WBGT] = 59.9 − 14.1 log10 M [W]","Recommended heat stress alert limit as a function of metabolic rate","Yes: "+W,NI,125,"RAL [°C-WBGT] = 59.9 − 14.1 log10 M [W]","Printed as the 60-minute-per-hour curve; shorter work periods have higher limits (Figures 8-1, 8-2). Equation typed in the source with U+2212 minus."),
T("NIOSH REL (acclimatized workers)","REL [°C-WBGT] = 56.7 − 11.5 log10 M [W]","Recommended heat stress exposure limit as a function of metabolic rate","Yes: "+W,NI,125,"REL [°C-WBGT] = 56.7 − 11.5 log10 M [W]","Equation only; no India-specific values. At the same M the two equations give different limits, so read Figures 8-1 and 8-2 before comparing them."),
T("Worked example: moderate work, acclimatized, 60 min/h","27.8 °C WBGT (82 °F) at about 300 kcal/h (348.9 W)","NIOSH's own worked example of the REL","Yes: "+W,NI,36,"REL curve at a WBGT of 27.8°C","Only worked number NIOSH gives in this section. Do not generalise to other workloads."),
T("India occupational WBGT limit","None found: Indian OSH Code sets no actionable heat exposure limits","Gap finding (secondary source)","n/a (gap)",SP,11,"fails to set actionable exposure limits or standards","Secondary: peer-reviewed review in Discover Public Health (2026), not an official document. Supports the conflict/gap entry that no Indian WBGT limits were located."),
]
