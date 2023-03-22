# Reese-et-al_2023_Elbe

**Supplementary dataset and Python scripts used for the creation of figures presented in [1]. Contains:**

*riverinfo.dat -- river and tributary freshwater input grid locations as grid cell indices

*rivers.nc -- freshwater forcing used in setup, contains sources from all included tributaries and Geesthacht. Data retrieved from [2,3,4] in combination with personal communication via the Bundesanstalt für Wasserbau (BAW). 
              
*topo_smoothed_v20_z0.nc -- setup topography and curvilinear grid. Topography created from datasets [5,6], for the latter see also [7].

./py_scripts -- Python scripts used for the creation of the figures, namely:

     *paper_plot_grid: Figure 2
     
     *paper_plot_forcing_validation: Figure 3
     
     *paper_plot_tidal_analysis: Figure 4
     
     *paper_plot_salinity_correlation: Figure 5
     
     *paper_plot_salinity_distribution: Figure 6
     
     *paper_longitudinal_TEF_Mixing: Figures 7,8
     
     *paper_plot_ULEM: Figure 9
     
     *paper_plot_temp_mixing: Figure 10
     
     *paper_plot_horizontal: Figure 11
     
     *paper_plot_salt_tidally-resolved: Tidally resolved stratification, not shown in paper
     
     
The model data used for the figures and loaded within the Python scripts is available through <link> or, alternatively, from the corresponding author of the study [1].

Note that *paper_plot_forcing_validation.py, *paper_plot_tidal_analysis.py, and *paper_plot_salinity_correlation.py require observational data that is freely available via portal-tideelbe.de; for in-depth documentation see [1].
*paper_plot_forcing_validation.py additionally requires meteorological forcing data for the first figure panel, which cannot be made public due to its proprietary nature.
The use of shapefiles to colour landmasses and plot the coastline in Figures 2, 6, and 11 has been commented out in the respective Python scripts due to unclear copyright regarding the underlying data. However, omission of these shapefiles does not change the interpretation of the model results. 


[1] N. Reese, U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.
    
[2] Wasserstraßen- und Schifffahrtsamt Magdeburg, 2021: Abflussstation Neu Darchau.
    Wasserstraßen- und Schifffahrtsverwaltung des Bundes, accessed 10 March 2021,
    http://www.portal-tideelbe.de.
    
[3] Landesamt für Landwirtschaft, Umwelt und ländliche Räume Schleswig-Holstein, 2022:
    Daily averaged freshwater runoff at Kellinghusen-Parkplatz and Kölln-Reisiek A23,
    2012–2014. Hochwasser- und Sturmflutinformation Schleswig-Holstein, accessed
    2 February 2022,
    https://www.umweltdaten.landsh.de/pegel/jsp/pegel.jsp?gui=ganglinie&thema=q&mstnr=114377;https://www.umweltdaten.landsh.de/pegel/jsp/pegel.jsp?mstnr=114527&wsize=free.
    
[4] Niedersächsicher Landesbetrieb für Wasserwirtschaft, Küsten- und Naturschutz, 2022:
    Daily averaged freshwater runoff at gauges no. 5963101 (Oersdorf),
    5945125 (Bienenbüttel), 5983110 (Rockstedt), and 5972105 (Schwinge), 2012–2014.
    Niedersächsische Landesdatenbank für wasserwirtschaftliche Daten, accessed 27 April 2022,
    http://www.wasserdaten.niedersachsen.de/cadenza/pages/home/welcome.xhtml.
    
[5] Wasserstraßen- und Schifffahrtsamt Hamburg, 2011: DGM-W 2010 Unter- und Außenelbe
    (Digitales Geländemodell des Wasserlaufes - Multifunktionsmodell). Wasserstraßen-
    und Schifffahrtsverwaltung des Bundes, accessed 12 January 2021,
    https://www.kuestendaten.de/Tideelbe/DE/Service/Kartenthemen/Kartenthemen_node.html
    
[6] Bundesamt für Seeschifffahrt und Hydrographie, 2019: AufMod Bathymetries
    German Bight 1982-2012 - WMS. Wasserstraßen- und Schifffahrtsverwaltung des
    Bundes, accessed 28 May 2013,
    https://gdk.gdi-de.org/geonetwork/srv/api/records/3543c490-d251-4948-830c-1da723a1a8eb.
    
[7] Heyer, H., and K. Schrottke, 2015: Einführung, Aufgabenstellung und
    Bearbeitungsstruktur im KFKI-Projekt AufMod. Die Küste, 83, 1–18,
    ISBN: 978-3-939230-40-3.
