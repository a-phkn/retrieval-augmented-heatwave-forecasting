# -*- coding: utf-8 -*-
# Part E: colour (tier_min) assignments, only where a source states the colour.
AHD="AHMEDABAD_HAP_2018"; IMD="IMD_HW_FAQ"
PRE="ahmedabad_precedent"; NAT="national_source"
Q_COOL="Activate “cooling centers,” such as temples, public buildings, malls, during a heat alert."
Q_SHADE="Expand access to shaded areas for outdoor workers, slum communities, and other vulnerable populations."
Q_WATER="Increase efforts to distribute fresh drinking water to the public."
Q_ICE="Pilot project to provide emergency ice packs and heat-illness prevention materials to traffic police, BRTS transit staff and construction workers."
Q_HRS="Change working hours of labourers."
Q_PORT="Provide sufficient portable drinking water."
Q_POWER="Communicate the local utility protocol to prioritize maintaining power to critical facilities (such as hospitals and UHCs)."
# action_id: (colour, doc, quote, match, basis)
TIERS={
"MUNI_COOLING_CENTRES_ACTIVATE":("RED",AHD,Q_COOL,"equivalent action (Delhi text also lists night shelters)",PRE),
"MUNI_SHADE_ACCESS_VULNERABLE":("RED",AHD,Q_SHADE,"same action",PRE),
"MUNI_DRINKING_WATER_DISTRIBUTION":("RED",AHD,Q_WATER,"same action",PRE),
"COMMS_ALERT_VIA_WHATSAPP_SMS":("RED",AHD,"Circulate warnings via text alerts or WhatsApp mobile messages","equivalent action",PRE),
"COMMS_ALERT_TO_HOSPITALS_PRACTITIONERS":("RED",AHD,"Develop an SMS alert system to send direct messages to private practitioners in addition to the medical professionals at public hospitals and UHCs.","same action",PRE),
"HEALTH_HOSPITAL_SURGE_STAFF":("RED",AHD,"Increase staffing at hospitals and UHCs to attend to the influx of patients during a heat alert, if feasible.","same action",PRE),
"HEALTH_WEEKLY_IMPACT_REPORT":("ORANGE",AHD,"Produce weekly reports of the public health impact for AMC Nodal Officer during a heat alert.","same action (listed at both Orange and Red; lowest colour used)",PRE),
"HEALTH_EMS_ICE_PACKS_IV":("RED",AHD,"Ensure adequate supply of ice packs and IV fluids.","same action",PRE),
"HEALTH_108_SMS_VULNERABLE_AREAS":("RED",AHD,"Disseminate SMS text messages to warn local residents during a heat alert.","same action",PRE),
"PUBLIC_CHECK_VULNERABLE_NEIGHBOURS":("RED",AHD,"Check on vulnerable neighbours, particularly during a heat alert.","same action",PRE),
"WORK_SHIFT_AWAY_1_5PM":("RED",AHD,"Encourage employers to shift outdoor workers’ schedules away from peak afternoon hours (1pm – 5pm) during a heat alert.","same action",PRE),
"WORK_EMERGENCY_ICE_PACKS_PROTECTION":("RED",AHD,Q_ICE,"same action",PRE),
"TRANSPORT_SHADE_WATER_ORS_STANDS":("RED",AHD,"Ensure availability of shade / shelters, drinking water, ORS packets etc., in bus stands, auto stands etc.","same action",PRE),
"SCHOOL_NO_OPERATION_12_4":("RED",AHD,"Ensure that all school and colleges should be closed during heatwave days.","equivalent action (Ahmedabad closes schools for the whole day; Delhi 12-4 pm)",PRE),
"SCHOOL_RESCHEDULE_TIMINGS":("ORANGE",AHD,"Restrict the school timings, if necessary","equivalent action",PRE),
"SCHOOL_PREMISES_AS_SHELTER":("RED",AHD,"If school is not functioning, permit use of school premises as shelter during day time.","same action",PRE),
"ENERGY_POWER_PRIORITY_CRITICAL_FACILITIES":("RED",AHD,Q_POWER,"same action",PRE),
"ENERGY_DISCOM_HOSPITAL_SUPPLY_CARE":("RED",AHD,Q_POWER,"equivalent action",PRE),
"NDMA_ENERGY_RESCHEDULE_LOAD_SHEDDING":("YELLOW",AHD,"Power shedding should be cut down/reduced during severe heat (frequency and timing).","equivalent action (Ahmedabad Red also says no power cuts)",PRE),
"ENERGY_DISCOM_AC_25C_AWARENESS":("YELLOW",AHD,"Create awareness among people on energy conservation.","equivalent action (general energy-conservation awareness, no AC setting named)",PRE),
"WORK_WATER_AT_SITES":("RED",AHD,Q_PORT,"equivalent action",PRE),
"DELHI_LABOUR_DRINKING_WATER_WORKPLACE":("RED",AHD,Q_PORT,"equivalent action",PRE),
"DELHI_LABOUR_SHIFT_AVOID_12_4PM":("RED",AHD,Q_HRS,"equivalent action",PRE),
"MAN_WORKERS_CHANGE_SHIFT_TIME_HOURS":("RED",AHD,Q_HRS,"equivalent action",PRE),
"MOLE_RESCHEDULE_WORKING_HOURS":("RED",AHD,Q_HRS,"equivalent action",PRE),
"WORK_RESCHEDULE_PEAK_HOURS_ADVISORY":("RED",AHD,Q_HRS,"equivalent action",PRE),
"DELHI_LABOUR_EMERGENCY_KIT_CONSTRUCTION":("RED",AHD,Q_ICE,"equivalent action",PRE),
"MOLE_EMERGENCY_ICE_PACKS_HEAT_ILLNESS_MATERIAL":("RED",AHD,Q_ICE,"equivalent action",PRE),
"HEALTH_SURVEILLANCE_HEAT_CASES":("YELLOW",AHD,"Hospitals should be instructed to update their admissions and emergency case records to track heat-related morbidity and mortality.","equivalent action",PRE),
"NAP_HOSP_AMBULANCE_ICE_PACKS_COLD_WATER":("ORANGE",AHD,"Keep sufficient stocks of reusable soft plastic ice packs for the 108 citywide emergency ambulances.","equivalent action (ice packs only; no cold water)",PRE),
"PUBLIC_DRINK_WATER":("ORANGE",IMD,"Drink sufficient water- even if not thirsty.","same action",NAT),
"PUBLIC_AVOID_SUN_12_3":("YELLOW",IMD,"Avoid heat exposure.","equivalent action (IMD: general; no clock window)",NAT),
}
OBS={"CLINICAL_REMOVE_FROM_HEAT","CLINICAL_40C_IV_TRANSPORT","PUBLIC_HEAT_STROKE_EMERGENCY","PUBLIC_NO_ORAL_FLUIDS_UNCONSCIOUS","NDMA_HEAT_STROKE_EMERGENCY_CALL_108_102","NDMA_HEAT_EXHAUSTION_FIRST_AID","NDMA_PHC_STABILISE_BEFORE_REFERRAL","NDMA_HOSP_REPORT_HRI_TO_IDSP","NDMA_NEWCOMERS_STAY_INDOORS_ONE_WEEK","NCDC_AMBULANCE_COOL_FIRST_TRANSPORT_SECOND","NCDC_COOL_FIRST_TRANSPORT_SECOND","NCDC_RAPID_COOLING_WITHIN_30_MIN","NCDC_DO_NOT_DELAY_COOLING_FOR_CLOTHING","NCDC_ICE_PACKS_NECK_AXILLA_GROIN_NOT_PRIMARY","NCDC_AVOID_WARM_WATER_FOR_COOLING","NCDC_AC_ROOM_ALONE_INADEQUATE_FOR_HEATSTROKE","NCDC_CLASSIC_HEATSTROKE_EVAPORATIVE_COOLING","NAP_PHC_CALL_108_COOL_BEFORE_REFER","NAP_PHC_NO_FLUID_IF_UNCONSCIOUS","NAP_PHC_COLD_COMPRESS_NECK_AXILLA_GROIN","NAP_ED_TEMP_MONITOR_DO_NOT_OVERCORRECT","NCDC_CLUSTER_DEATHS_DISTRICT_TASK_FORCE_MEETING","MAN_HEATEXH_REMOVE_FROM_HEAT_STOP_ACTIVITY","MAN_HEATEXH_BELOW_104F_REASSESS_5MIN","MAN_HEATEXH_104F_IV_TRANSPORT_ED","MAN_HEATSTROKE_ONSITE_COOLING_SHADY_PLACE","MAN_HEATSTROKE_AMBULANCE_AC_OR_WINDOWS","NIOSH_MEDICAL_MONITORING_ABOVE_RAL","WHO_FANS_CAUTION_ABOVE_35C","NDMA_ORANGE_SUNSTROKE_FIRST_AID"}
STANDING={"HEALTH_HOSPITAL_READINESS","HEALTH_HEAT_ILLNESS_PROTOCOLS","HEALTH_EMS_ICE_PACKS_IV","HEALTH_AMBULANCE_DYNAMIC_DEPLOYMENT","HEALTH_SURVEILLANCE_HEAT_CASES","HEALTH_AC_WARDS","WORK_WATER_AT_SITES","TRANSPORT_SHADE_WATER_ORS_STANDS","ENERGY_DISCOM_HOSPITAL_SUPPLY_CARE","ENERGY_DISCOM_AC_25C_AWARENESS","NDMA_HOSP_SURGE_CAPACITY_HEATSTROKE_BEDS","NDMA_HOSP_RAPID_RESPONSE_TEAMS","NDMA_HOSP_SUPPLIES_STAFF_IV_ORS","NDMA_ENERGY_UNINTERRUPTED_POWER_BEFORE_DURING_SUMMER","NDMA_ENERGY_MOP_ADVISORY_UNINTERRUPTED_SUPPLY","NDMA_ENERGY_RESCHEDULE_LOAD_SHEDDING","NCDC_PUBLIC_COOLING_AND_DRINKING_WATER_FACILITIES","NCDC_HEATSTROKE_UNITS_EXTERNAL_COOLING","NCDC_FACILITY_DRINKING_WATER","NCDC_FACILITY_COOLING_APPLIANCES","NCDC_ENERGY_HOSPITAL_UNINTERRUPTED_ELECTRICITY","NCDC_ENERGY_HEALTH_FACILITY_LOAD_AUDITS","NCDC_ENERGY_HEALTH_FACILITY_CONSERVATION","NCDC_EVENTS_AVOID_12_3","NCDC_EVENTS_WATER_OUTLETS","NAP_HOSP_DEDICATED_BED_COOLER_AREA","NAP_HOSP_COOLING_EQUIPMENT_AC_COOLER_FAN","NAP_HOSP_AMBULANCE_ICE_PACKS_COLD_WATER","BEE_AC_DEFAULT_24C_MANDATORY","BEE_AC_NOTIFICATION_TEXT_24C","HAP25_ELECTRICITY_VULNERABILITY_ANALYSIS","HAP25_POWER_DEMAND_HEAT_RISK_MAP","MOLE_ESIC_DGLW_HEATSTROKE_HELP_DESKS","MOLE_AWARENESS_LABOUR_CHOWKS","MAN_LOW_SES_PUBLIC_ACCESS_COOL_PLACES","CEA_CONSUMERS_LOAD_WITHIN_SANCTIONED","CEA_DISCOMS_MONITOR_EQUIPMENT_LOADING","CEA_SWITCH_OFF_UNNEEDED_LOADS","CEA_AC_NOT_CONTINUOUS_24C","CEA_SERVICE_AC_REGULARLY","CEA_NO_MULTIPLE_APPLIANCES_ONE_SOCKET","CEA_DISCOMS_DISSEMINATE_TO_CONSUMERS","DELHI_LABOUR_SWITCH_OFF_ELECTRICAL_EQUIPMENT","DELHI_LABOUR_RCD_FOR_2KW_AND_ABOVE","DELHI_LABOUR_ELECTRICAL_SAFETY_OFFICER_250KW","DELHI_LABOUR_FIRE_SWITCH_OFF_SUPPLY"}
