# VestasV27
**Introduction:**
Code used to process the data from a Vestas V27 Wind Turbine and apply different Environmental and Operational Mitigation Procedures.
The dataset used corresponds to an experimental test carried out on a fully operational Vestas V27 between 2014 and 2015 in Roskilde, Denmark.
Researchers implemented a set of accelerometers along one of the three blades. In total, 12 monoaxial piezoelectric accelerometers (BK Type 4507B). The nominal sensitivity of accelerometers was selected based on their location and expected vibration.
In addition, the experimental campaign included an electromechanical actuator, which would excite the turbine blade with a plunger. Although, the use of active excitation tends to be favorable in any Structural Health Monitoring. Our study aims to use solely operational excitation from the turbine (passive excitation).

More information regarding the experimental campaign can be found in the [original paper](https://backend.orbit.dtu.dk/ws/portalfiles/portal/128004294/32_Tcherniak.pdf)

Some additional documents related to the V27 dataset, which I found helpful:

[Predictive health monitoring of wind turbine blades](https://energiforskning.dk/files/slutrapporter/eudp_phm_final_report_v4_-_full_id_494016_id_494018.pdf)

[General specifications Vestas V27-225kW, 50Hz wind turbine](http://www.husdesign.no/lars/V27-Teknisk%20spesifikasjon/gen%20specification%20v27.pdf)

[DTU Report Wind turbine test Vestas V27-225kW](http://www.husdesign.no/lars/V27-Teknisk%20spesifikasjon/gen%20specification%20v27.pdf)

This is a previous study carried out by the Technical University of Denmark (DTU). The report describes fundamental measurements performed on Vestas V27.

## Hybrid implicit-explicit regression
Mitigation of Environmental and Operational Variabilities (EOVs) remains one of the main challenges to adopting Structural Health Monitoring (SHM) as part of wind turbine maintenance. To this end, two main strategies have been proposed: explicit and implicit procedures, which attempt to mitigate EOV effects in different ways. Explicit methods build regression models using environmental and/or operational parameters, which are later used to correct Damage-sensitive Features (DSFs), while Implicit methods disregard certain DSFs due to their influence on EOVs. This work proposes the use of so-called hybrid implicit-explicit procedures to minimize the drawbacks of conventional procedures. In comparison to explicit methods, this hybrid approach only corrects DSFs that are influenced by EOVs. Therefore, avoiding poor corrections of DSFs. In addition, it does not disregard DSFs as it happens on implicit methods, thus eliminating the loss of damage information from certain DSFs. These two factors are considered to be advantageous for improving damage detection performance. The work results are validated using vibrational data from an operational Vestas V27 wind turbine with different induced damages on one of its blades. The proposed method outperforms fully explicit procedures using different regression models, implying that models can benefit from a selective DSF correction.

## Functions available
Original data from the Vestas V27 is not published, as it was shared by Dmitri Tcherniak from HBK. Nonetheless, all functions coded for this work as provided.
They could be repurposed for other wind turbines or civil structures datasets with small effort.

### Cite our work:
I will add the citation once the work is published


