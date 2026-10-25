# -*- coding: utf-8 -*-
from thr_def import T
TM="NPCCHH_TRAINING_MANUAL_MO"
NA="n/a (not WBGT)"
PG="NPCCHH Training Manual for Medical Officers (PDF page = printed page). "
THR5=[
T("Heat stroke core temperature (manual, section 3)","≥ 40 °C, stated with '(103°F or higher)'","Heat stroke sign",NA,TM,10,"core temperature ≥ 40°C, (103°F","INTERNAL INCONSISTENCY: 40 °C is 104 °F, and the manual's own treatment protocol (p.13) and algorithm (p.16) use 104 °F / 40 °C. The National Action Plan says ≥40°C or ≥104°F. Treat 103 °F as a source typo; the quote is kept as printed. "+PG),
T("Cooling target for heat stroke","Core temperature 39 °C or below within 10-40 min, and 38.5 °C or below within 2 hours","Clinical cooling target",NA,TM,15,"cool the core temperature to 39°C or below within 10–40 min and to 38.5 °C or below within 2hrs","The NAP says not to cool below 38 °C (NAP p.26 PDF). Text layer lacks word spaces; quote matches ignoring spaces. "+PG),
T("Sickroom temperature during heat stroke cooling","Adjust room temperature to 20–24 °C","Clinical setting for treating heat stroke, not a public advice value",NA,TM,15,"Adjust the room temperature to 20°C–24°C","Not comparable to the 24/25 °C public AC advice (BEE/CEA 24 °C, Delhi HAP 25 degree). Text layer lacks word spaces. "+PG),
]
