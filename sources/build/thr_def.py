# -*- coding: utf-8 -*-
H25="DELHI_HAP_2025"; H24="DELHI_HAP_2024_25"; IMD="IMD_HW_FAQ"; NDMA="NDMA_HW_SLIDES"
def T(q,thr,meaning,var,doc,page,quote,notes=""):
    return dict(quantity=q,threshold=thr,meaning=meaning,wbgt_variant=var,doc_id=doc,page=page,verbatim_excerpt=quote,notes=notes)
NA="n/a (not WBGT)"
THR=[
T("Tmax (plains station)","≥ 40 °C","Minimum Tmax before a heat wave is considered at a plains station",NA,IMD,1,"Heat wave is considered if maximum temperature of a station reaches at least","Value is printed in the source as '40⁰C'; text extraction renders the degree mark as '0', so the numeral is not quoted."),
T("Departure of Tmax from normal","4.5 to 6.4 °C","Heat wave (IMD FAQ)",NA,IMD,1,"Heat Wave: Departure from normal is 4.5","The task brief says 4.5/6.5; the IMD FAQ and NDMA slides say 6.4. Degree mark renders as '0' in extraction."),
T("Departure of Tmax from normal","> 6.4 °C","Severe heat wave (IMD FAQ)",NA,IMD,1,"Severe Heat Wave: Departure from normal is >6.4","Same note."),
T("Actual Tmax","≥ 45 °C","Heat wave (absolute criterion)",NA,IMD,1,"Heat Wave: When actual maximum temperature ≥ 45","Degree mark renders as '0' (e.g. '450C')."),
T("Actual Tmax","≥ 47 °C","Severe heat wave (absolute criterion)",NA,IMD,1,"Severe Heat Wave: When actual maximum temperature ≥47","Same note."),
T("Declaration rule","≥ 2 stations in a met sub-division, ≥ 2 consecutive days","Heat wave is declared on the second day",NA,IMD,2,"If above criteria met at least in 2 stations in a Meteorological sub-division for at least two consecutive days and it declared on the second day.",""),
T("Coastal stations","Tmax departure ≥ 4.5 °C and actual Tmax ≥ 37 °C","Heat wave may be described (coastal)",NA,IMD,2,"When maximum temperature departure is 4.5","Delhi is not coastal; recorded for completeness."),
T("Hot & humid weather","Tmax ≥ 3 °C above normal with above-normal RH","'Hot & Humid Weather' label (IMD)",NA,IMD,2,"it is termed as Hot & Humid Weather over that station.","Not a WBGT or heat-index threshold."),
T("Tmin departure (warm night)","4.5 to 6.4 °C (warm); > 6.4 °C (very warm)","Warm night / very warm night; only declared when Tmax ≥ 40 °C",NA,IMD,2,"Warm night: minimum temperature departure is 4.5","Degree mark renders as '0' in extraction."),
T("Colour code – Yellow","Heat wave conditions at isolated pockets persist on 2 days","IMD yellow alert (Be updated)",NA,IMD,8,"Heat wave conditions at isolated pockets persists on 2 days",""),
T("Colour code – Orange","(i) Severe heat wave conditions persist for 2 days, or (ii) heat wave persists for 4 days or more","IMD orange alert (Be prepared)",NA,IMD,8,"(i) Severe heat wave conditions persists for 2 days","Second condition quoted in notes: '(ii) Through not severe, but heat wave persists for 4 days or more' (source spelling)."),
T("Colour code – Red","(i) Severe heat wave persists for more than 2 days, or (ii) total heat/severe heat wave days exceed 6","IMD red alert (Take action)",NA,IMD,8,"(ii) Total number of heat/severe heat wave days exceeding 6 days.","Condition (i) is 'Severe heat wave persists for more than 2 days' but its words are interleaved with neighbouring table cells in the PDF text, so only condition (ii) is quoted."),
T("Tmax – Delhi HAP alert table (Yellow)","≥ 40 °C","'Hot day advisory' (Yellow, Be updated)",NA,H25,10,"Yellow(Be updated) Hot day advisory >= 40 °C","Table caption in the plan: 'Heat Alert Thresholds for Delhi City (source: NDMA)'."),
T("Tmax – Delhi HAP alert table (Orange)","≥ 45 °C","'Heat alert day' (Orange, Be prepared)",NA,H25,10,">= 45°C","The same table gives Red as ≥ 45 °C too, so Orange and Red are not separable by temperature in this plan — see conflicts."),
T("Tmax – Delhi HAP alert table (Red)","≥ 45 °C (as printed)","'Extreme heat alert day' (Red, Take action)",NA,H25,10,"Red Alert (Take Action) Extreme heat","Likely a table error in the plan; recorded as printed."),
T("Normal Tmax ≤ 40 °C – departure criterion","Heat wave 5–6 °C; severe ≥ 7 °C","Older IMD criterion as reproduced in the plan",NA,H25,9,"When normal maximum temperature of a station is less than or equal to 40°C","Differs from the IMD FAQ criterion (4.5/6.4). See conflicts."),
T("Normal Tmax > 40 °C – departure criterion","Heat wave 4–5 °C; severe ≥ 6 °C","Older IMD criterion as reproduced in the plan",NA,H25,9,"When normal maximum temperature of a station is more than 40°C","Differs from the IMD FAQ."),
T("Actual Tmax – declare irrespective of normal","≥ 45 °C","Heat waves should be declared",NA,H25,9,"When actual maximum temperature remains 45°C or more irrespective of normal maximum temperature, heat waves should be declared.",""),
T("Percentile thresholds (NDMA recommendation)","75th, 85th, 95th percentile","Recommended local thresholds on Tmax/Tmin/Tmean, Humidex, WBGT, UTCI, heat index",
  "WBGT is only listed as a candidate variable; no WBGT value or variant is given",H25,43,"NDMA recommends using the 75th, 85th and 95th percentile for Indian context",""),
T("Heat index (worked example)","Air 34 °C and RH 75% → heat index 49 °C","Illustrative example, not a threshold","Heat index (apparent temperature)",H25,10,"if the air temperature is 34°C and the relative humidity is 75%, the heat index--how hot it feels--is 49°C.",""),
T("Environmental temperature (body heat gain)","37 °C","Below this, 'no harm to the human body' per the plan","Air/environmental temperature",H24,11,"There will be no harm to the human body if the environmental temperature remains at 37° C.","Statement of the plan, not an operational alert level."),
T("Clinical – body temperature","≥ 40 °C","Heat stroke / start IV rehydration and transport","Body (core) temperature, not weather",H25,66,"If temperature is 40°C or above, initiate IV rehydration and immediately transport to emergency department for stabilization.","Clinical protocol."),
]
