# -*- coding: utf-8 -*-
"""Build the Journal of Hydrology submission package as Word files.

Produces three .docx files that follow the Journal of Hydrology Guide for Authors:
  Manuscript_JHydrol.docx     single column, numbered sections, display equations
                              numbered on the right, Table 1, CRediT and all
                              declarations, author-year references.
  Highlights.docx             3 to 5 points, each within 85 characters.
  Supplementary_Information.docx   supporting tables and a note on the figures.

House style: navy (#12355B) headings on black body text, no em dashes.
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
NAVY = RGBColor(0x12, 0x35, 0x5B)
BLACK = RGBColor(0x11, 0x11, 0x11)
BODY_FONT = "Times New Roman"
BODY_PT = 11

def _load(name):
    for p in (ROOT / name, ROOT / "06_outputs" / "phase5" / name):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(name)


allion = _load("allion_summary.json")
scen = _load("scenario_results.json")
robust = _load("robustness_results.json")
ph5 = _load("phase5_results.json")
typo = _load("typology_summary.json")
seas = _load("seasonality_summary.json")
mgmt = _load("management_summary.json")


# ----------------------------------------------------------------------------- base document
def new_doc(double_spaced=True, line_numbers=False):
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(BODY_PT)
    normal.font.color.rgb = BLACK
    pf = normal.paragraph_format
    pf.line_spacing = 2.0 if double_spaced else 1.15
    pf.space_after = Pt(0)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(1)
        s.left_margin = s.right_margin = Inches(1)
    if line_numbers:
        sectPr = doc.sections[0]._sectPr
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        ln.set(qn("w:distance"), "360")
        sectPr.append(ln)
    return doc


def _no_dash(t):
    return t.replace("—", ", ").replace("–", "-")


def para(doc, text="", *, size=BODY_PT, bold=False, italic=False, color=BLACK,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=0,
         line=None, indent=None):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if line is not None:
        pf.line_spacing = line
    if indent is not None:
        pf.first_line_indent = Inches(indent)
    if text:
        r = p.add_run(_no_dash(text))
        r.bold = bold
        r.italic = italic
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = BODY_FONT
    return p


def heading(doc, text, level=1):
    sizes = {1: 13, 2: 11.5, 3: 11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(_no_dash(text))
    r.bold = True
    r.font.size = Pt(sizes[level])
    r.font.color.rgb = NAVY
    r.font.name = BODY_FONT
    return p


def add_eq(doc, segments, num):
    """One display equation, centred, with its number pushed to the right margin.

    segments: list of (text, kind); kind in {'t','v','sub','sup'}:
      t normal text, v italic variable, sub subscript, sup superscript.
    """
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.15
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.tab_stops.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    for text, kind in segments:
        r = p.add_run(text)
        r.font.name = BODY_FONT
        r.font.size = Pt(BODY_PT)
        r.font.color.rgb = BLACK
        if kind == "v":
            r.italic = True
        elif kind == "sub":
            r.font.subscript = True
        elif kind == "sup":
            r.font.superscript = True
    rn = p.add_run("\t(%d)" % num)
    rn.font.name = BODY_FONT
    rn.font.size = Pt(BODY_PT)
    return p


def set_cell(cell, text, bold=False, size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(_no_dash(str(text)))
    r.bold = bold
    r.font.size = Pt(size)
    r.font.name = BODY_FONT
    r.font.color.rgb = BLACK


def horizontal_only_borders(table):
    """Elsevier tables: no vertical rules, no shading. Keep only top/bottom/header lines."""
    tbl = table._tbl
    for cell in table._cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "bottom"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "111111")
            borders.append(el)
        for edge in ("left", "right"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "nil")
            borders.append(el)
        tcPr.append(borders)


# ----------------------------------------------------------------------------- manuscript
def build_manuscript():
    doc = new_doc(double_spaced=True, line_numbers=True)

    # Title page
    title = ("Continental patterns, drivers, and mid-century projection of freshwater "
             "salinization across the United States and Canada")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = NAVY
    r.font.name = BODY_FONT

    para(doc, "Naimul Islam", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
         line=1.15, space_after=2)
    para(doc, "Independent Researcher, Dhaka, Bangladesh", size=10.5, italic=True,
         align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15, space_after=2)
    para(doc, "ORCID: https://orcid.org/0009-0002-3442-8980", size=10.5,
         align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15, space_after=8)
    para(doc, "Corresponding author: Naimul Islam, naimul.islam.bangladesh@gmail.com, "
              "Dhaka, Bangladesh", size=10.5, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15,
         space_after=2)

    doc.add_page_break()

    # Abstract
    heading(doc, "Abstract", 1)
    abstract = (
        "Freshwater salinization threatens drinking water, aquatic life, and "
        "infrastructure across North America, yet most assessments are regional and no "
        "harmonized forecast spans both the United States and Canada. We assembled 2.51 "
        "million major-ion records from five long-term monitoring networks into a common "
        "continental record for chloride, sodium, calcium, magnesium, sulfate, and "
        "alkalinity, covering 3,944 stations from 1990 to 2026. We quantified multi-decadal "
        "trends, attributed the catchment and climate drivers of salinity with interpretable "
        "machine learning validated on unseen regions, fingerprinted salt sources through "
        "ion ratios, classified stations into salinization regimes, and projected each "
        "station to 2050 under trend continuation, CMIP6 climate scenarios, and "
        "salt-management counterfactuals. Chloride rises significantly at 25.9 percent of "
        "stations and falls at 11.0 percent; sodium mirrors it, and sulfate declines with "
        "reduced atmospheric deposition. Salinity level is well predicted from drivers, but "
        "its rate is not, which points to local de-icing management. The sodium-to-chloride "
        "ratio marks a halite signature across the cold, developed northeast, and winter "
        "chloride exceeds summer chloride at most northern stations. Five regimes emerge, "
        "led by a small road-salt urban group with a median chloride of 188 mg/L. By 2050 "
        "the median chloride station gains about 11 percent and the share exceeding the "
        "United States chronic aquatic-life criterion rises from 8.1 to 12.1 percent; "
        "climate scenarios move this little, but stabilizing the rising stations would avoid "
        "three quarters of the increase. Management, not climate, governs the mid-century "
        "outlook.")
    para(doc, abstract, line=1.5, space_after=6)

    kw = ("Keywords: Salinization; Chloride; Road salt; Trend analysis; Machine learning; "
          "CMIP6; North America")
    para(doc, kw, size=10.5, line=1.5, space_after=6)

    doc.add_page_break()

    # 1. Introduction
    heading(doc, "1. Introduction", 1)
    para(doc,
        "Rising concentrations of dissolved ions in fresh water, described collectively as "
        "the freshwater salinization syndrome, degrade drinking-water sources, harm aquatic "
        "communities, mobilize metals and nutrients, and corrode infrastructure (Kaushal et "
        "al., 2018). The mechanisms are established in principle: de-icing salt on roads, "
        "weathering accelerated by acid and salt inputs, agriculture, wastewater, mining, "
        "and evapoconcentration under a warming climate (Kaushal et al., 2023). Early work "
        "in the northeastern United States linked long-term chloride increases directly to "
        "road salt and watershed development (Kaushal et al., 2005), and later syntheses "
        "showed the same signal in rivers of snow-affected urban watersheds (Corsi et al., "
        "2015) and in lakes across North America (Dugan et al., 2017). Chloride from road "
        "salt is the most visible signal in cold regions, but the syndrome spans many ions "
        "and both natural and human sources.", indent=0.25)
    para(doc,
        "Most salinization assessments are regional, focused on a single network, state, or "
        "basin. Continental frameworks for stream chemistry exist, such as the "
        "Stream-Catchment dataset for the conterminous United States (Hill et al., 2016) and "
        "the global HydroATLAS attributes (Linke et al., 2019), but they stop at national "
        "borders or do not couple observed trends with drivers and a forward projection, "
        "even though the drainage systems, climate gradients, and salt-management practices "
        "of the United States and Canada form one connected system. No harmonized assessment "
        "couples the long-term trend, the drivers, and a forward projection across the two "
        "countries. This study fills that gap. We ask three questions. What are the "
        "multi-decadal trends in the six major salinization ions across the United States "
        "and southern Canada? What catchment and climate factors control salinity, and how "
        "well do they generalize to unseen regions? And where is salinization heading by "
        "2050 under a continuation of current trends and under climate scenarios?", indent=0.25)

    # 2. Data
    heading(doc, "2. Data", 1)
    para(doc,
        "We combined records from five long-term water-quality networks: the United States "
        "Water Quality Portal, the United States Geological Survey, Environment and Climate "
        "Change Canada, the Ontario Provincial Water Quality Monitoring Network, and the "
        "British Columbia Environmental Monitoring System. Records were harmonized to a "
        "single long-format schema with consistent ion identities, units converted to mg/L "
        "with alkalinity expressed as calcium carbonate, below-detection handling, and "
        "cross-border duplicate removal. The harmonized record holds 2,512,644 observations "
        "of the six ions across 3,944 stations from 1990 to 2026 (Water Quality Portal "
        "1,336,965; British Columbia 672,350; Ontario 272,477; Environment and Climate "
        "Change Canada 200,818; United States Geological Survey 30,034; Table S1). Analysis "
        "was restricted to the study domain, the lower 48 states, the District of Columbia, "
        "and southern Canada between 24 and 60 degrees north. Stations were retained for "
        "trend analysis when they had at least ten years of record with four or more "
        "observations per year, giving 14,166 station-ion series.", indent=0.25)

    # 3. Methods
    heading(doc, "3. Methods", 1)

    heading(doc, "3.1 Trend estimation", 2)
    para(doc,
        "For each station and ion, sub-annual observations were reduced to annual means and "
        "log-transformed as y = ln(1 + C), where C is the annual mean concentration in "
        "mg/L. The long-term trend was estimated with the Theil-Sen slope, the median of "
        "all pairwise slopes between annual values (Sen, 1968; Theil, 1950), which is robust "
        "to outliers and to non-normal residuals (Eq. 1). Significance was assessed with the "
        "nonparametric Mann-Kendall test through its S statistic (Eq. 2), a pairing standard "
        "for water-quality trend detection (Mann, 1945; Kendall, 1975).", indent=0.25)
    add_eq(doc, [("β = median", "t"), (" [ (", "t"), ("y", "v"), ("j", "sub"),
                 (" − ", "t"), ("y", "v"), ("i", "sub"), (") / (", "t"),
                 ("t", "v"), ("j", "sub"), (" − ", "t"), ("t", "v"), ("i", "sub"),
                 (") ],  ", "t"), ("i", "v"), (" < ", "t"), ("j", "v")], 1)
    add_eq(doc, [("S = ", "t"), ("Σ", "t"), ("i", "sub"), (" < ", "t"), ("j", "sub"),
                 (" sgn(", "t"), ("y", "v"), ("j", "sub"), (" − ", "t"),
                 ("y", "v"), ("i", "sub"), (")", "t")], 2)
    para(doc,
        "The slope in log units per year was converted to a percentage change per year "
        "(Eq. 3), which is the quantity mapped in Figure 1.", indent=0.25)
    add_eq(doc, [("p = ( ", "t"), ("e", "v"), ("β", "sup"), (" − 1 ) × 100", "t")], 3)

    heading(doc, "3.2 Driver covariates", 2)
    para(doc,
        "Catchment and climate covariates were sampled at every station through Google Earth "
        "Engine. Land cover and human-pressure covariates were taken at two scales, a 500 m "
        "near-stream buffer and a 5 km landscape buffer, because land use influences stream "
        "chemistry differently at the riparian and the wider scale. Elevation and slope came "
        "from the MERIT digital elevation model (Yamazaki et al., 2017); land cover from ESA "
        "WorldCover (Zanaga et al., 2022); climate normals from WorldClim (Fick and Hijmans, "
        "2017); potential evapotranspiration and an aridity index from TerraClimate "
        "(Abatzoglou et al., 2018); a 1990 to 2023 temperature and precipitation trend from "
        "TerraClimate; population density from the Gridded Population of the World (CIESIN, "
        "2018); night-time lights from VIIRS (Elvidge et al., 2017); and surface soil clay, "
        "sand, organic carbon, and pH from OpenLandMap (Hengl et al., 2017). The full "
        "covariate list is given in Table S4.", indent=0.25)

    heading(doc, "3.3 Models and validation", 2)
    para(doc,
        "Two multitask gradient-boosted models (LightGBM; Ke et al., 2017) were fit with ion "
        "identity as a categorical feature, one predicting the recent salinity level, the "
        "mean of the last five years in log units, and one predicting the trend. Both were "
        "evaluated with spatial-block cross validation: stations were grouped into two degree "
        "cells and folds were split by cell, so a fold is always tested on regions it never "
        "trained on. This follows Roberts et al. (2017) and avoids the optimistic skill that "
        "random cross validation gives on spatially autocorrelated data. Model skill was "
        "measured by the coefficient of determination on the held-out folds (Eq. 4). Driver "
        "importance was computed with SHAP values (Lundberg and Lee, 2017).", indent=0.25)
    add_eq(doc, [("R", "v"), ("2", "sup"), (" = 1 − ", "t"),
                 ("Σ (", "t"), ("y", "v"), ("i", "sub"), (" − ", "t"),
                 ("ŷ", "v"), ("i", "sub"), (")", "t"), ("2", "sup"),
                 (" / Σ (", "t"), ("y", "v"), ("i", "sub"), (" − ", "t"),
                 ("ȳ", "v"), (")", "t"), ("2", "sup")], 4)

    heading(doc, "3.4 Projection and climate scenarios", 2)
    para(doc,
        "Each station was projected to 2050 by continuing its observed Theil-Sen trend from "
        "the recent base level (Eq. 5), with a 95 percent interval propagated from the "
        "Theil-Sen slope confidence bounds (Eq. 6), where t_base is the last year of record "
        "and the low and high slopes are the interval on the estimated trend.", indent=0.25)
    add_eq(doc, [("Ĉ", "v"), ("2050", "sub"), (" = ", "t"), ("C", "v"), ("base", "sub"),
                 (" · exp[ β ( 2050 − ", "t"), ("t", "v"), ("base", "sub"),
                 (" ) ]", "t")], 5)
    add_eq(doc, [("[ ", "t"), ("C", "v"), ("base", "sub"), (" e", "t"),
                 ("β", "sup"), ("lo", "sup"), ("Δt", "sup"), (" ,  ", "t"),
                 ("C", "v"), ("base", "sub"), (" e", "t"), ("β", "sup"),
                 ("hi", "sup"), ("Δt", "sup"), (" ],   Δt = 2050 − ", "t"),
                 ("t", "v"), ("base", "sub")], 6)
    para(doc,
        "A climate overlay was added from the NASA NEX-GDDP-CMIP6 downscaled ensemble "
        "(Thrasher et al., 2022): the change in mean annual temperature and precipitation "
        "between 2005 to 2014 and 2046 to 2055 was computed as a five-model mean under "
        "SSP2-4.5 and SSP5-8.5, sampled on a coarse grid and interpolated to each station. "
        "The climate covariates were shifted to their 2050 values, with potential "
        "evapotranspiration scaled by a standard sensitivity of about two percent per degree "
        "of warming (Eq. 7), and the level model was re-evaluated. The ratio of the shifted "
        "to the baseline prediction gives a climate response factor (Eq. 8) that was applied "
        "to the trend projection while management was held constant (Eq. 9), which isolates "
        "the climate signal and gives the forecast a scenario range.", indent=0.25)
    add_eq(doc, [("PET′ = PET · ( 1 + 0.02 ΔT", "t"), ("s", "sub"), (" )", "t")], 7)
    add_eq(doc, [("f", "v"), ("s", "sub"), (" = exp[ ", "t"), ("g", "v"),
                 ("(", "t"), ("x", "v"), ("′", "t"), (") − ", "t"),
                 ("g", "v"), ("(", "t"), ("x", "v"), (") ]", "t")], 8)
    add_eq(doc, [("Ĉ", "v"), ("2050", "sub"), ("s", "sup"), (" = ", "t"),
                 ("Ĉ", "v"), ("2050", "sub"), (" · ", "t"), ("f", "v"),
                 ("s", "sub")], 9)
    para(doc,
        "Here g is the trained level model, x the baseline covariates, x-prime the "
        "covariates shifted to the 2050 climate under scenario s, and f the resulting "
        "response factor.", indent=0.25)

    heading(doc, "3.5 Source fingerprinting and regime typology", 2)
    para(doc,
        "At every station with a co-located multi-ion record we computed diagnostic mass "
        "ratios from the recent levels. The sodium-to-chloride ratio (Eq. 10) fingerprints "
        "the salt source: pure halite, the mineral of road salt and natural rock salt, sits "
        "near 0.65, a chloride excess below that value still marks halite-type inputs with "
        "sodium retention in soils, and a sodium excess above about one marks silicate "
        "weathering, wastewater, or ion-exchange softeners. The calcium-to-magnesium ratio "
        "separates limestone from dolomite weathering, and the chloride-to-sulfate ratio "
        "separates the road-salt signal from the mineral and deposition signal. To "
        "synthesize these signatures we clustered the stations that carried at least four of "
        "the six ions on their log concentrations, their sodium-to-chloride and "
        "calcium-to-magnesium ratios, their chloride and sulfate trends, and a few landscape "
        "drivers, using k-means on standardized features. The number of regimes was fixed at "
        "five for interpretability, and each cluster was named from its centre. The regimes "
        "form a soft continuum rather than sharp groups, so the typology is descriptive.",
        indent=0.25)
    add_eq(doc, [("r", "v"), ("Na:Cl", "sub"), (" = ", "t"), ("C", "v"), ("Na", "sub"),
                 (" / ", "t"), ("C", "v"), ("Cl", "sub"), ("     (halite ≈ 0.65)", "t")], 10)

    heading(doc, "3.6 Seasonal chloride signal", 2)
    para(doc,
        "Road salt is applied in winter, so a road-salt-driven stream carries more chloride "
        "in winter than in summer. From the observation-level record we computed, for every "
        "station with at least eight winter (December to February) and eight summer (June to "
        "August) chloride samples, the ratio of mean winter to mean summer chloride (Eq. 11). "
        "We tested how this ratio varies with latitude and urban intensity, and compared the "
        "winter and summer chloride trends to see whether de-icing intensity is changing.",
        indent=0.25)
    add_eq(doc, [("R", "v"), ("w:s", "sub"), (" = ", "t"),
                 ("mean C", "t"), ("Cl,DJF", "sub"), (" / ", "t"),
                 ("mean C", "t"), ("Cl,JJA", "sub")], 11)

    heading(doc, "3.7 Management counterfactual", 2)
    para(doc,
        "To separate the avoidable part of the 2050 outlook from the committed part, we "
        "reprojected chloride under two management counterfactuals applied from a policy "
        "start year of 2025 to the stations with a significant rising trend, holding all "
        "other stations on their measured trajectory (Eq. 12). A policy factor of one is the "
        "no-action trend, one half is a halving of the application-driven slope, and zero "
        "holds the rising stations at their policy-year level. This is an illustrative "
        "counterfactual, not a prediction of policy.", indent=0.25)
    add_eq(doc, [("Ĉ", "v"), ("2050", "sub"), (" = ", "t"), ("C", "v"), ("base", "sub"),
                 (" exp[ β ( ", "t"), ("t", "v"), ("p", "sub"), (" − ", "t"),
                 ("t", "v"), ("base", "sub"), (" ) + φ β ( 2050 − ", "t"),
                 ("t", "v"), ("p", "sub"), (" ) ]", "t")], 12)
    para(doc,
        "Here t-p is the policy start year and the policy factor phi takes the values one, "
        "one half, and zero for the no-action, halved, and stabilized cases.", indent=0.25)

    # 4. Results
    heading(doc, "4. Results", 1)

    heading(doc, "4.1 Observed trends", 2)
    para(doc,
        "Across the 14,166 long-term series, chloride rises significantly at 25.9 percent of "
        "stations and falls at 11.0 percent, and sodium shows the same asymmetry (25.4 "
        "versus 9.1 percent). Calcium and magnesium rise at about 20 percent of stations "
        "against 13 percent falling. Sulfate is the exception, falling at 28.1 percent of "
        "stations and rising at 15.8 percent, consistent with the decline in atmospheric "
        "sulfur deposition since the 1990s. Alkalinity is close to balanced. The spatial "
        "pattern (Figure 1) shows the strongest chloride increases across the northeast, the "
        "upper Midwest, and other cold, developed regions where road salt is heavily "
        "applied.", indent=0.25)

    heading(doc, "4.2 Drivers of salinity", 2)
    para(doc,
        "Salinity level is well predicted from catchment and climate drivers under spatial "
        "cross validation, with an overall R2 of 0.58 and per-ion values from 0.47 for "
        "sodium and sulfate to 0.63 for alkalinity (Table S2). The strongest drivers by SHAP "
        "value are ion identity, forest cover, precipitation, an east-west gradient, bare "
        "ground, potential evapotranspiration, night-time lights, elevation, the warming "
        "rate, and soil pH (Figure 2). The set is physically coherent: climate sets dilution "
        "and evapoconcentration, forest and bare cover mark low and high salinity "
        "landscapes, night-time lights carry the urban and road-salt signal, and soil pH "
        "carries the carbonate weathering that raises calcium, magnesium, and alkalinity.",
        indent=0.25)
    para(doc,
        "The trend is far harder to predict, with a spatial cross-validation R2 of about "
        "0.05. This is a result rather than a shortcoming. The rate of salinization is "
        "spatially heterogeneous and is not set by fixed catchment properties, which is what "
        "would be expected if the dominant control is the intensity of local de-icing salt "
        "application, a management variable that static maps do not capture.", indent=0.25)

    heading(doc, "4.3 Projection to 2050", 2)
    para(doc,
        "Continuing observed trends, the median chloride station gains about 11 percent by "
        "2050, and the share of stations exceeding the United States chronic aquatic-life "
        "criterion of 230 mg/L rises from 8.1 to 12.1 percent, with the aquatic-life "
        "threshold of 120 mg/L crossed by 19.3 percent of stations, up from 13.6 percent "
        "(Figure 5; Table 1). Sodium rises by a similar 10.8 percent at the median, and the "
        "share exceeding the 200 mg/L aesthetic guidance rises from 6.7 to 11.0 percent. "
        "Calcium and magnesium rise modestly, near 3 percent. Sulfate declines by about 10 "
        "percent at the median. The projected change is mapped for chloride in Figure 7. "
        "Every projected value carries a Theil-Sen interval; for the median chloride station "
        "the projected 2050 concentration is 23.8 mg/L with a 95 percent interval of 15.1 to "
        "38.1 mg/L.", indent=0.25)

    heading(doc, "4.4 Climate scenarios", 2)
    para(doc,
        "Mid-century warming reaches about 1.7 degrees C under SSP2-4.5 and 2.2 degrees C "
        "under SSP5-8.5 across the domain, with precipitation generally increasing. Across "
        "all six ions the median climate response factor is close to one (0.99 under both "
        "scenarios), because warming and higher precipitation nearly offset in their effect "
        "on major-ion concentration. For chloride the response is slightly amplifying, with "
        "a median factor of 1.01 under SSP2-4.5 and 1.02 under SSP5-8.5, which raises the "
        "median 2050 chloride change from 11 percent under trend alone to 14 percent under "
        "SSP2-4.5 and 16 percent under SSP5-8.5. Threshold exceedance nonetheless stays "
        "within about one percentage point of the trend projection under both scenarios "
        "(Figure 5), because the climate response acts mainly on the central tendency rather "
        "than the extreme tail. The management-driven trajectory therefore dominates the "
        "2050 outlook.", indent=0.25)

    heading(doc, "4.5 Source signatures and salinization regimes", 2)
    na_cl = typo["na_cl"]["median"]
    halite_share = typo["halite_like_share_pct"]
    para(doc,
        f"The sodium-to-chloride mass ratio has a continental median of {na_cl:.2f}, close to "
        f"the halite value of 0.65, and {halite_share:.0f} percent of stations fall in the "
        "halite-like band between 0.4 and 0.9. The ratio is spatially organized (Figure 3): a "
        "chloride excess relative to sodium dominates the cold, developed northeast and the "
        "Great Lakes, the halite signature expected where road salt is the source, while a "
        "sodium excess dominates the arid and agricultural interior, where silicate "
        "weathering, evaporite dissolution, and softeners raise sodium. The five-regime "
        "typology (Figure 4) summarizes these signatures. A small road-salt urban regime of "
        f"{typo['typology']['regimes']['Road-salt urban (rising Cl)']['n']} stations carries "
        "a median chloride of 188 mg/L, close to the chronic criterion, with a "
        "sodium-to-chloride ratio of 0.57 and a rising trend, and sits in the northeast. A "
        "carbonate-weathering regime is the largest and covers the limestone interior; a "
        "sulfate-declining regime tracks the deposition signal; a dilute forested regime "
        "covers the eastern Canadian and Pacific coasts; and an arid evaporative regime, with "
        "a sodium excess, covers the dry interior. The regimes form a continuum rather than "
        "sharp groups, so they describe rather than partition the continent.", indent=0.25)

    heading(doc, "4.6 Seasonal chloride signal", 2)
    ws_all = seas["ws_ratio_median_all"]; ws_cold = seas["ws_ratio_median_cold_lat_ge40"]
    we_cold = seas["share_winter_enriched_cold_pct"]
    para(doc,
        f"Across {seas['n_stations']:,} stations with adequate seasonal sampling, winter "
        f"chloride exceeds summer chloride at {seas['share_winter_enriched_pct']:.0f} percent "
        f"of stations, and at {we_cold:.0f} percent of the colder stations north of 40 "
        f"degrees. The median winter-to-summer ratio rises from {seas['ws_ratio_median_warm_lat_lt40']:.2f} "
        f"in the warm south to {ws_cold:.2f} in the cold north, and climbs further at the "
        "highest latitudes (Figure S4), the direct seasonal fingerprint of winter de-icing. "
        "The amplitude is modest, however, and the winter and summer chloride trends in cold "
        f"regions are almost equal ({seas['winter_slope_median_cold']:.4f} against "
        f"{seas['summer_slope_median_cold']:.4f} in log units per year), which indicates that "
        "a large part of the road-salt load persists year-round in groundwater and sustains "
        "baseflow chloride rather than appearing only as a winter pulse (Kelly et al., 2008). "
        "The winter-to-summer ratio is mapped in Figure S3.", indent=0.25)

    heading(doc, "4.7 Avoidable exceedance under management", 2)
    ex = mgmt["exceedance"]; av = mgmt["avoided_at_230_pp"]
    para(doc,
        f"Of the {mgmt['n_chloride_stations']:,} chloride stations, "
        f"{mgmt['n_rising_significant']:,} ({mgmt['share_rising_pct']:.1f} percent) have a "
        "significant rising trend and drive the projected increase. Acting on these stations "
        f"from 2025 changes the 2050 outlook markedly (Figure 6). At the chronic criterion of "
        f"230 mg/L, exceedance rises from {ex['recent']['230']} percent now to "
        f"{ex['2050_trend']['230']} percent under no action, but only to {ex['2050_halved']['230']} "
        f"percent if their application-driven slope is halved and to {ex['2050_stabilized']['230']} "
        f"percent if it is stabilized. Stabilization therefore avoids {av['stabilized_vs_trend']:.1f} "
        f"of the {ex['2050_trend']['230'] - ex['recent']['230']:.1f} percentage-point increase, "
        "about three quarters of the committed rise, which quantifies the leverage of salt "
        "management identified by the climate scenarios.", indent=0.25)

    # 5. Discussion
    heading(doc, "5. Discussion", 1)
    para(doc,
        "The continental picture confirms and extends the freshwater salinization syndrome "
        "across both countries. Chloride and sodium, the road-salt pair, rise in about two "
        "and a half times as many stations as they fall, matching the river and lake trends "
        "reported for parts of the continent (Corsi et al., 2015; Dugan et al., 2017), while "
        "sulfate declines with the well-documented reduction in atmospheric deposition. The "
        "driver analysis shows that where salinity is high can be explained well from "
        "climate, land cover, urban intensity, and soil chemistry, but how fast it is "
        "changing cannot, which places local salt management at the center of the problem "
        "and of any solution. This separation of level from rate is the main methodological "
        "result and is consistent with the state-factor view of progressive salinization "
        "stages (Kaushal et al., 2023). The ion-ratio fingerprint and the regime typology add "
        "the mechanism behind the pattern: the halite sodium-to-chloride signature and the "
        "road-salt urban regime concentrate in the cold, developed northeast, while sodium "
        "excess and carbonate weathering dominate the interior, which is the multi-ion "
        "chemical-cocktail structure that the salinization syndrome predicts (Kaushal et al., "
        "2019). The seasonal signal supports the same reading but with a caveat: winter "
        "chloride exceeds summer chloride at most northern stations, yet the modest amplitude "
        "and the near-equal winter and summer trends show that much of the road-salt load "
        "persists year-round in groundwater, a legacy store that sustains baseflow chloride "
        "and will delay any recovery (Kelly et al., 2008). The projection makes the stakes "
        "concrete: without a change in trajectory, one in eight chloride stations will exceed "
        "the chronic aquatic-life criterion by 2050, the climate scenarios show that this "
        "outcome is driven by management rather than by climate change itself, and the "
        "management counterfactual shows that stabilizing the rising stations would avoid "
        "about three quarters of the increase. The lever is real and local.", indent=0.25)
    para(doc,
        "Several limitations frame these results. Trend continuation assumes each station "
        "holds its historical rate, which is transparent but will misstate stations where "
        "management changes. The driver covariates are point and buffer samples rather than "
        "full upstream catchments, and the low skill of the trend model means the projection "
        "relies on measured trends rather than a fitted surface, which is appropriate here "
        "but limits extrapolation to unmonitored areas. The climate overlay uses a "
        "five-model ensemble and a level-model response rather than a process model of "
        "solute transport. These choices keep the forecast interpretable and reproducible, "
        "and each can be relaxed in later work.", indent=0.25)

    # 6. Conclusion
    heading(doc, "6. Conclusions", 1)
    para(doc,
        "A harmonized, bi-national record of 2.51 million major-ion observations shows "
        "widespread freshwater salinization across the United States and Canada, driven at "
        "the landscape scale by climate, land cover, urban intensity, and soil chemistry, "
        "and at the rate scale by local salt management. Ion ratios and a regime typology "
        "trace the increase to a halite road-salt signature in the cold, developed northeast, "
        "and a seasonal analysis confirms winter enrichment while showing that much of the "
        "load persists year-round. Under a continuation of current trends, threshold "
        "exceedance rises materially by 2050, climate scenarios change this only at the "
        "margin, and a management counterfactual shows that stabilizing the rising stations "
        "would avoid about three quarters of the increase. The result argues for management "
        "of de-icing salt as the primary lever for protecting fresh water across the "
        "continent.", indent=0.25)

    # Back matter
    heading(doc, "Acknowledgements", 1)
    para(doc,
        "This work used open data from the United States Water Quality Portal, the United "
        "States Geological Survey, Environment and Climate Change Canada, the Ontario "
        "Provincial Water Quality Monitoring Network, and the British Columbia Environmental "
        "Monitoring System, and open geospatial products served through Google Earth Engine. "
        "The author thanks the agencies and data providers that maintain these networks.",
        indent=0.25, line=1.5)

    heading(doc, "CRediT author contribution statement", 1)
    para(doc,
        "Naimul Islam: Conceptualization, Data curation, Formal analysis, Methodology, "
        "Software, Validation, Visualization, Writing - original draft, Writing - review and "
        "editing.", indent=0.25, line=1.5)

    heading(doc, "Declaration of competing interest", 1)
    para(doc, "The author declares no competing interests.", indent=0.25, line=1.5)

    heading(doc, "Funding", 1)
    para(doc,
        "This research did not receive any specific grant from funding agencies in the "
        "public, commercial, or not-for-profit sectors.", indent=0.25, line=1.5)

    heading(doc, "Declaration of generative AI and AI-assisted technologies in the "
                 "manuscript preparation process", 1)
    para(doc,
        "During the preparation of this work the author used Claude (Anthropic) to assist "
        "with code development, data-processing scripts, and language editing. After using "
        "this tool the author reviewed and edited the content as needed and takes full "
        "responsibility for the content of the published article.", indent=0.25, line=1.5)

    heading(doc, "Data availability", 1)
    para(doc,
        "The harmonized station-level dataset and all analysis code are openly available in "
        "a public repository with a citable DOI, to be inserted at acceptance, and are "
        "mirrored at github.com/Naimul-islam-bd. The underlying raw records are public and "
        "available from the Water Quality Portal, the United States Geological Survey, "
        "Environment and Climate Change Canada, the Ontario Provincial Water Quality "
        "Monitoring Network, and the British Columbia Environmental Monitoring System.",
        indent=0.25, line=1.5)

    # References
    heading(doc, "References", 1)
    refs = [
        "Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C., 2018. TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958 to 2015. Scientific Data 5, 170191.",
        "CIESIN (Center for International Earth Science Information Network), Columbia University, 2018. Gridded Population of the World, Version 4 (GPWv4): Population Density. NASA SEDAC, Palisades, NY.",
        "Corsi, S.R., De Cicco, L.A., Lutz, M.A., Hirsch, R.M., 2015. River chloride trends in snow-affected urban watersheds: increasing concentrations outpace urban growth rate and are common among all seasons. Science of the Total Environment 508, 488-497.",
        "Dugan, H.A., Bartlett, S.L., Burke, S.M., Doubek, J.P., Krivak-Tetley, F.E., Skaff, N.K., Summers, J.C., Farrell, K.J., McCullough, I.M., Morales-Williams, A.M., Roberts, D.C., Ouyang, Z., Scordo, F., Hanson, P.C., Weathers, K.C., 2017. Salting our freshwater lakes. Proceedings of the National Academy of Sciences 114, 4453-4458.",
        "Elvidge, C.D., Baugh, K., Zhizhin, M., Hsu, F.C., Ghosh, T., 2017. VIIRS night-time lights. International Journal of Remote Sensing 38, 5860-5879.",
        "Fick, S.E., Hijmans, R.J., 2017. WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. International Journal of Climatology 37, 4302-4315.",
        "Hengl, T., Mendes de Jesus, J., Heuvelink, G.B.M., Ruiperez Gonzalez, M., Kilibarda, M., Blagotic, A., Shangguan, W., Wright, M.N., Geng, X., Bauer-Marschallinger, B., Guevara, M.A., Vargas, R., MacMillan, R.A., Batjes, N.H., Leenaars, J.G.B., Ribeiro, E., Wheeler, I., Mantel, S., Kempen, B., 2017. SoilGrids250m: global gridded soil information based on machine learning. PLoS ONE 12, e0169748.",
        "Hill, R.A., Weber, M.H., Leibowitz, S.G., Olsen, A.R., Thornbrugh, D.J., 2016. The Stream-Catchment (StreamCat) dataset: a database of watershed metrics for the conterminous United States. Journal of the American Water Resources Association 52, 120-128.",
        "Kaushal, S.S., Groffman, P.M., Likens, G.E., Belt, K.T., Stack, W.P., Kelly, V.R., Band, L.E., Fisher, G.T., 2005. Increased salinization of fresh water in the northeastern United States. Proceedings of the National Academy of Sciences 102, 13517-13520.",
        "Kaushal, S.S., Likens, G.E., Pace, M.L., Utz, R.M., Haq, S., Gorman, J., Grese, M., 2018. Freshwater salinization syndrome on a continental scale. Proceedings of the National Academy of Sciences 115, E574-E583.",
        "Kaushal, S.S., Likens, G.E., Pace, M.L., Utz, R.M., Haq, S., Gorman, J., Grese, M., et al., 2019. Novel 'chemical cocktails' in inland waters are a consequence of the freshwater salinization syndrome. Philosophical Transactions of the Royal Society B 374, 20180017.",
        "Kaushal, S.S., Likens, G.E., Pace, M.L., Reimer, J.E., Maas, C.M., Galella, J.G., Utz, R.M., Duan, S., Kryger, J.R., Yaculak, A.M., et al., 2023. Five state factors control progressive stages of freshwater salinization syndrome. Limnology and Oceanography Letters 8, 190-211.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017. LightGBM: a highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30, 3146-3154.",
        "Kelly, V.R., Lovett, G.M., Weathers, K.C., Findlay, S.E.G., Strayer, D.L., Burns, D.J., Likens, G.E., 2008. Long-term sodium chloride retention in a rural watershed: legacy effects of road salt on streamwater concentration. Environmental Science and Technology 42, 410-415.",
        "Kendall, M.G., 1975. Rank Correlation Methods, fourth ed. Charles Griffin, London.",
        "Linke, S., Lehner, B., Ouellet Dallaire, C., Ariwi, J., Grill, G., Anand, M., Beames, P., Burchard-Levine, V., Maxwell, S., Moidu, H., Tan, F., Thieme, M., 2019. Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6, 283.",
        "Lundberg, S.M., Lee, S.-I., 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30, 4765-4774.",
        "Mann, H.B., 1945. Nonparametric tests against trend. Econometrica 13, 245-259.",
        "Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schroder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F., Dormann, C.F., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40, 913-929.",
        "Sen, P.K., 1968. Estimates of the regression coefficient based on Kendall's tau. Journal of the American Statistical Association 63, 1379-1389.",
        "Theil, H., 1950. A rank-invariant method of linear and polynomial regression analysis. Proceedings of the Royal Netherlands Academy of Sciences 53, 386-392, 521-525, 1397-1412.",
        "Thrasher, B., Wang, W., Michaelis, A., Melton, F., Lee, T., Nemani, R., 2022. NASA Global Daily Downscaled Projections, CMIP6. Scientific Data 9, 262.",
        "Yamazaki, D., Ikeshima, D., Tawatari, R., Yamaguchi, T., O'Loughlin, F., Neal, J.C., Sampson, C.C., Kanae, S., Bates, P.D., 2017. A high-accuracy map of global terrain elevations. Geophysical Research Letters 44, 5844-5853.",
        "Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., et al., 2022. ESA WorldCover 10 m 2021 v200. Zenodo.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        run = p.add_run(_no_dash(r))
        run.font.size = Pt(10)
        run.font.name = BODY_FONT
        run.font.color.rgb = BLACK

    # Figure captions
    doc.add_page_break()
    heading(doc, "Figure captions", 1)
    figcaps = [
        ("Figure 1.", " Observed chloride trend, 1990 to 2026, in per cent per year, at the "
         "3,231 long-term chloride stations. Navy marks increases and orange marks decreases."),
        ("Figure 2.", " Drivers of salinity level, as the mean absolute SHAP value of each "
         "covariate in the spatially cross-validated level model. Larger values indicate a "
         "stronger control on salinity level."),
        ("Figure 3.", " Sodium-to-chloride mass ratio at co-located stations. The reference "
         "line at 0.65 is the ratio of pure halite; values below it mark a chloride excess "
         "typical of road salt, and values above about one mark a sodium excess from "
         "weathering, wastewater, or softeners."),
        ("Figure 4.", " Salinization regime typology. Five regimes from k-means clustering on "
         "the multi-ion signature, chloride and sulfate trends, and landscape drivers. The "
         "regimes form a continuum and are named from their cluster centres."),
        ("Figure 5.", " Chloride threshold exceedance, the share of chloride stations above "
         "each guideline, for recent conditions and 2050 under the trend and the two climate "
         "scenarios."),
        ("Figure 6.", " Chloride threshold exceedance in 2050 under the management "
         "counterfactuals: no action, halved application, and stabilization of the "
         "significantly rising stations from 2025."),
        ("Figure 7.", " Projected chloride change by 2050, in per cent, under trend "
         "continuation. Navy marks increases and orange marks decreases."),
    ]
    for lab, txt in figcaps:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        rb = p.add_run(lab); rb.bold = True; rb.font.size = Pt(10.5); rb.font.name = BODY_FONT
        rb.font.color.rgb = BLACK
        rt = p.add_run(_no_dash(txt)); rt.font.size = Pt(10.5); rt.font.name = BODY_FONT
        rt.font.color.rgb = BLACK

    # Table 1 on its own page
    doc.add_page_break()
    cap = doc.add_paragraph()
    cap.paragraph_format.line_spacing = 1.15
    cap.paragraph_format.space_after = Pt(6)
    cr = cap.add_run("Table 1. ")
    cr.bold = True
    cr.font.size = Pt(10.5)
    cr.font.name = BODY_FONT
    cr.font.color.rgb = BLACK
    ct = cap.add_run(_no_dash(
        "Projected 2050 change and threshold exceedance by ion. Median change is the "
        "median across stations of the percentage change from the recent base level; "
        "increasing is the share of stations with a positive trend; exceedance is the "
        "share of stations above the stated guideline now and in 2050 under the trend "
        "projection. A dash means no single-value guideline applies."))
    ct.font.size = Pt(10.5)
    ct.font.name = BODY_FONT
    ct.font.color.rgb = BLACK

    header = ["Ion", "Stations", "Median change to 2050 (%)", "Stations increasing (%)",
              "Guideline (mg/L)", "Exceedance now (%)", "Exceedance 2050 (%)"]
    rows = [
        ["Chloride (Cl)", "3,231", "+11.1", "60.3", "EPA chronic, 230", "8.1", "12.1"],
        ["Sodium (Na)", "2,383", "+10.8", "62.2", "Aesthetic, 200", "6.7", "11.0"],
        ["Calcium (Ca)", "2,520", "+3.0", "56.1", "-", "-", "-"],
        ["Magnesium (Mg)", "2,557", "+3.0", "55.9", "-", "-", "-"],
        ["Sulfate (SO4)", "2,505", "-9.9", "40.5", "Secondary MCL, 250", "15.5", "16.7"],
        ["Alkalinity", "970", "-2.8", "44.5", "-", "-", "-"],
    ]
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.autofit = True
    for j, h in enumerate(header):
        set_cell(table.rows[0].cells[j], h, bold=True, size=9,
                 align=WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            set_cell(table.rows[i].cells[j], val, size=9,
                     align=WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT)
    horizontal_only_borders(table)
    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    nr = note.add_run(_no_dash(
        "Note: guideline values are the United States EPA chronic (230 mg/L) and acute "
        "aquatic-life criteria for chloride, the secondary maximum contaminant level for "
        "sulfate (250 mg/L), and the aesthetic guidance for sodium (200 mg/L)."))
    nr.font.size = Pt(8.5)
    nr.italic = True
    nr.font.name = BODY_FONT
    nr.font.color.rgb = BLACK

    out = ROOT / "Manuscript_JHydrol.docx"
    doc.save(out)
    print("wrote", out.name)


# ----------------------------------------------------------------------------- highlights
def build_highlights():
    doc = new_doc(double_spaced=False)
    heading(doc, "Highlights", 1)
    pts = [
        "First harmonized salinization record spanning the United States and Canada",
        "Chloride rises at 26 percent of stations and falls at 11 percent",
        "Salinity level is predictable from drivers; its rate is set by local salt use",
        "Ion ratios reveal a road-salt halite signature across the cold northeast",
        "Managing rising stations would avoid most of the 2050 chloride increase",
    ]
    for t in pts:
        p = doc.add_paragraph(style=None)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run("• " + _no_dash(t))
        r.font.size = Pt(11)
        r.font.name = BODY_FONT
        r.font.color.rgb = BLACK
    # length check note (each within 85 characters, per the guide)
    assert all(len(t) <= 85 for t in pts), [len(t) for t in pts]
    out = ROOT / "Highlights.docx"
    doc.save(out)
    print("wrote", out.name, "max len", max(len(t) for t in pts))


# ----------------------------------------------------------------------------- supplement
def build_supplement():
    doc = new_doc(double_spaced=False)
    p = doc.add_paragraph()
    r = p.add_run("Supplementary Information")
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = NAVY
    r.font.name = BODY_FONT
    para(doc, "Continental patterns, drivers, and mid-century projection of freshwater "
              "salinization across the United States and Canada", italic=True, size=10.5,
         align=WD_ALIGN_PARAGRAPH.LEFT, space_after=8)

    def stable(caption_bold, caption_rest, header, rows, note=None, colsize=9):
        cap = doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(4)
        a = cap.add_run(caption_bold)
        a.bold = True; a.font.size = Pt(10.5); a.font.name = BODY_FONT; a.font.color.rgb = BLACK
        b = cap.add_run(_no_dash(caption_rest))
        b.font.size = Pt(10.5); b.font.name = BODY_FONT; b.font.color.rgb = BLACK
        t = doc.add_table(rows=1 + len(rows), cols=len(header))
        for j, h in enumerate(header):
            set_cell(t.rows[0].cells[j], h, bold=True, size=colsize,
                     align=WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT)
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                set_cell(t.rows[i].cells[j], val, size=colsize,
                         align=WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT)
        horizontal_only_borders(t)
        if note:
            np_ = doc.add_paragraph()
            np_.paragraph_format.space_before = Pt(3)
            nr = np_.add_run(_no_dash(note))
            nr.italic = True; nr.font.size = Pt(8.5); nr.font.name = BODY_FONT

    # S1 data sources
    stable("Table S1. ", "Data sources and record counts in the harmonized dataset.",
           ["Network", "Country", "Records"],
           [["United States Water Quality Portal", "US", "1,336,965"],
            ["British Columbia Environmental Monitoring System", "Canada", "672,350"],
            ["Ontario Provincial Water Quality Monitoring Network", "Canada", "272,477"],
            ["Environment and Climate Change Canada", "Canada", "200,818"],
            ["United States Geological Survey", "US", "30,034"],
            ["Total", "", "2,512,644"]])

    # S2 per-ion R2
    r2 = ph5["level_R2_by_ion"]
    stable("Table S2. ", "Level-model skill by ion under spatial-block cross validation, "
           "and the trend-model skill. R2 is on held-out spatial folds.",
           ["Ion", "Level R2", "Stations"],
           [["Chloride (Cl)", f"{r2['cl']:.3f}", "3,231"],
            ["Sodium (Na)", f"{r2['na']:.3f}", "2,383"],
            ["Calcium (Ca)", f"{r2['ca']:.3f}", "2,520"],
            ["Magnesium (Mg)", f"{r2['mg']:.3f}", "2,557"],
            ["Sulfate (SO4)", f"{r2['so4']:.3f}", "2,505"],
            ["Alkalinity", f"{r2['alk']:.3f}", "970"],
            ["All ions (level)", f"{ph5['level_model_R2_spatialCV']:.2f}", "14,166"],
            ["All ions (trend)", f"{ph5['trend_model_R2_spatialCV']:.2f}", "14,166"]],
           note="The low trend R2 is a finding: the rate of change is not set by static "
                "catchment properties, which points to local de-icing management.")

    # S3 robustness
    stable("Table S3. ", "Robustness of the level model to algorithm and to land-cover "
           "buffer scale, all under the same spatial-block cross validation (R2).",
           ["Configuration", "R2"],
           [["LightGBM, both buffers", f"{robust['LightGBM_both']:.3f}"],
            ["Random forest, both buffers", f"{robust['RandomForest_both']:.3f}"],
            ["LightGBM, 500 m buffer only", f"{robust['LightGBM_500m_only']:.3f}"],
            ["LightGBM, 5 km buffer only", f"{robust['LightGBM_5km_only']:.3f}"]],
           note="Skill is stable across the algorithm and buffer choices, so the level "
                "result does not depend on these settings.")

    # S4 covariates
    stable("Table S4. ", "Driver covariates, their source products, and the scale at which "
           "each was sampled.",
           ["Covariate group", "Variables", "Source", "Scale"],
           [["Terrain", "Elevation, slope", "MERIT DEM (Yamazaki et al., 2017)", "Point"],
            ["Land cover", "Forest, bare, cropland, grassland, shrub, built-up, wetland, water, snow/ice",
             "ESA WorldCover (Zanaga et al., 2022)", "500 m and 5 km"],
            ["Climate normals", "Air temperature, precipitation", "WorldClim (Fick and Hijmans, 2017)", "Point"],
            ["Water balance", "Potential evapotranspiration, aridity, precipitation",
             "TerraClimate (Abatzoglou et al., 2018)", "Point"],
            ["Climate trend", "Warming rate, precipitation trend (1990 to 2023)", "TerraClimate", "Point"],
            ["Population", "Population density", "GPWv4 (CIESIN, 2018)", "500 m and 5 km"],
            ["Night lights", "Night-time radiance", "VIIRS (Elvidge et al., 2017)", "500 m and 5 km"],
            ["Soil", "Clay, sand, organic carbon, pH", "OpenLandMap (Hengl et al., 2017)", "Point"],
            ["Location", "Latitude, longitude", "Station coordinates", "Point"]],
           colsize=8.5)

    # S5 chloride exceedance full
    ex = scen["chloride_exceedance"]
    stable("Table S5. ", "Chloride threshold exceedance (percent of chloride stations) for "
           "recent conditions and 2050 under the trend and climate scenarios.",
           ["Threshold (mg/L)", "Recent", "2050 trend", "2050 SSP2-4.5", "2050 SSP5-8.5"],
           [["120 (aquatic-life chronic)", f"{ex['recent']['120']}", f"{ex['2050_trend']['120']}",
             f"{ex['2050_ssp245']['120']}", f"{ex['2050_ssp585']['120']}"],
            ["230 (EPA chronic)", f"{ex['recent']['230']}", f"{ex['2050_trend']['230']}",
             f"{ex['2050_ssp245']['230']}", f"{ex['2050_ssp585']['230']}"],
            ["860 (EPA acute)", f"{ex['recent']['860']}", f"{ex['2050_trend']['860']}",
             f"{ex['2050_ssp245']['860']}", f"{ex['2050_ssp585']['860']}"]],
           note=f"Median chloride climate response factor: {scen['climate_factor_median']['ssp245']} "
                f"(SSP2-4.5) and {scen['climate_factor_median']['ssp585']} (SSP5-8.5) across all ions.")

    # S6 ion ratios
    rr = typo
    stable("Table S6. ", "Diagnostic ion mass ratios across co-located stations "
           "(median and interquartile range). The halite sodium-to-chloride ratio is 0.65.",
           ["Ratio", "Median", "P25", "P75", "Stations"],
           [["Sodium-to-chloride", f"{rr['na_cl']['median']}", f"{rr['na_cl']['p25']}",
             f"{rr['na_cl']['p75']}", f"{rr['na_cl']['n']:,}"],
            ["Calcium-to-magnesium", f"{rr['ca_mg']['median']}", f"{rr['ca_mg']['p25']}",
             f"{rr['ca_mg']['p75']}", f"{rr['ca_mg']['n']:,}"],
            ["Chloride-to-sulfate", f"{rr['cl_so4']['median']}", f"{rr['cl_so4']['p25']}",
             f"{rr['cl_so4']['p75']}", f"{rr['cl_so4']['n']:,}"]],
           note=f"{rr['halite_like_share_pct']}% of stations fall in the halite-like band "
                "(sodium-to-chloride 0.4 to 0.9).")

    # S7 typology regimes
    reg = typo["typology"]["regimes"]
    order = ["Road-salt urban (rising Cl)", "Carbonate weathering", "Sulfate-declining",
             "Dilute forested", "Arid evaporative"]
    stable("Table S7. ", "Salinization regimes from k-means clustering (k = 5, silhouette "
           f"{typo['typology']['silhouette']}). Values are cluster-median signatures.",
           ["Regime", "Stations", "Cl (mg/L)", "Na:Cl", "Ca:Mg", "Cl trend", "SO4 trend"],
           [[k, f"{reg[k]['n']:,}", f"{reg[k]['cl_level']}", f"{reg[k]['na_cl']}",
             f"{reg[k]['ca_mg']}", f"{reg[k]['slope_cl']:+.4f}", f"{reg[k]['slope_so4']:+.4f}"]
            for k in order if k in reg],
           note="Trends are Theil-Sen slopes in log units per year. The regimes form a soft "
                "continuum rather than sharp groups.", colsize=8.5)

    # S8 seasonality
    stable("Table S8. ", "Seasonal chloride signal: winter (Dec to Feb) to summer (Jun to "
           "Aug) ratio, and winter and summer trends in cold regions.",
           ["Quantity", "Value"],
           [["Stations with adequate seasonal sampling", f"{seas['n_stations']:,}"],
            ["Median winter-to-summer ratio, all", f"{seas['ws_ratio_median_all']}"],
            ["Median ratio, cold (latitude >= 40)", f"{seas['ws_ratio_median_cold_lat_ge40']}"],
            ["Median ratio, warm (latitude < 40)", f"{seas['ws_ratio_median_warm_lat_lt40']}"],
            ["Stations winter-enriched (%)", f"{seas['share_winter_enriched_pct']}"],
            ["Winter-enriched in cold regions (%)", f"{seas['share_winter_enriched_cold_pct']}"],
            ["Winter chloride trend, cold (log/yr)", f"{seas['winter_slope_median_cold']}"],
            ["Summer chloride trend, cold (log/yr)", f"{seas['summer_slope_median_cold']}"]])

    # S9 management
    mex = mgmt["exceedance"]
    stable("Table S9. ", "Management counterfactual: chloride exceedance (percent of stations) "
           "in 2050 under no action, halved application, and stabilization of the rising "
           "stations from 2025.",
           ["Threshold (mg/L)", "Recent", "No action", "Halved", "Stabilized"],
           [["120", f"{mex['recent']['120']}", f"{mex['2050_trend']['120']}",
             f"{mex['2050_halved']['120']}", f"{mex['2050_stabilized']['120']}"],
            ["230", f"{mex['recent']['230']}", f"{mex['2050_trend']['230']}",
             f"{mex['2050_halved']['230']}", f"{mex['2050_stabilized']['230']}"],
            ["860", f"{mex['recent']['860']}", f"{mex['2050_trend']['860']}",
             f"{mex['2050_halved']['860']}", f"{mex['2050_stabilized']['860']}"]],
           note=f"{mgmt['n_rising_significant']:,} of {mgmt['n_chloride_stations']:,} chloride "
                f"stations ({mgmt['share_rising_pct']}%) are significantly rising and were "
                "treated as manageable.")

    # Supplementary figures note
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(12)
    a = cap.add_run("Supplementary figures. ")
    a.bold = True; a.font.size = Pt(10.5); a.font.name = BODY_FONT; a.font.color.rgb = BLACK
    b = cap.add_run(_no_dash(
        "Figure S1, observed trend maps for all six ions. Figure S2, projected 2050 change "
        "maps for all six ions. Figure S3, winter-to-summer chloride ratio map. Figure S4, "
        "winter-to-summer ratio against latitude. All figures are supplied as separate "
        "high-resolution TIFF and PNG files, along with the seven main-text figures and a "
        "graphical abstract."))
    b.font.size = Pt(10.5); b.font.name = BODY_FONT; b.font.color.rgb = BLACK

    out = ROOT / "Supplementary_Information.docx"
    doc.save(out)
    print("wrote", out.name)


if __name__ == "__main__":
    build_manuscript()
    build_highlights()
    build_supplement()
    print("done")
