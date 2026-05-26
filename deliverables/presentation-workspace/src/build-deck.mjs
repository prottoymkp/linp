import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {
  Presentation,
  PresentationFile,
  column,
  row,
  grid,
  layers,
  panel,
  text,
  image,
  shape,
  chart,
  table,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} from "@oai/artifact-tool";
import { paint, stroke } from "@oai/artifact-tool/presentation-jsx";

const WORKSPACE = path.resolve(process.cwd());
const REPO_ROOT = path.resolve(WORKSPACE, "..", "..");
const DELIVERABLES = path.join(REPO_ROOT, "deliverables");
const ASSETS = path.join(DELIVERABLES, "assets");
const OUT = path.join(WORKSPACE, "output");
const SCRATCH = path.join(WORKSPACE, "scratch");
const PREVIEWS = path.join(SCRATCH, "previews");
const LAYOUTS = path.join(SCRATCH, "layouts");
const SCRATCH_ASSETS = path.join(SCRATCH, "assets");

const C = {
  bg: "#F6F1EA",
  paper: "#FFFDF8",
  ink: "#18201D",
  muted: "#51645E",
  quiet: "#77867F",
  teal: "#1F5B63",
  tealDark: "#123B42",
  tealSoft: "#DDEBE8",
  leather: "#A96533",
  amber: "#D58A3A",
  amberSoft: "#F4DEC3",
  clay: "#C86F4A",
  line: "#D7C8B8",
  charcoal: "#101312",
  white: "#FFFFFF",
};

const FONT = "Aptos";
const DISPLAY = "Aptos Display";
const W = 1920;
const H = 1080;

function repoPython() {
  const code = `
import json
from app.assets import load_demo_tables
from app.orchestrator import run_optimization
from app.config import FG_DATASET, BOM_DATASET, RM_DATASET, CAP_DATASET
from app.validate import validate_inputs

tables = load_demo_tables()
validate_inputs(tables)
fg, rm, meta, ps, purchase_detail = run_optimization(
    tables,
    run_purchase_planner=True,
    purchase_target_fill_pcts="25,50,75,100",
)
meta_map = dict(zip(meta["Key"], meta["Value"]))
purchase_rows = []
if ps is not None and not ps.empty:
    for _, row in ps.iterrows():
        purchase_rows.append({
            "targetFillPct": float(row["TargetFillPct"]),
            "targetPairs": int(row["TargetPairs"]),
            "achievedPairs": int(row["AchievedPairs"]),
            "achievedMargin": float(row["AchievedMargin"]),
            "totalBuyCost": float(row["TotalBuyCost"]),
            "status": str(row["Status"]),
            "method": str(row["Method"]),
        })

binding = rm.sort_values(["is_binding", "availability_utilization_pct"], ascending=[False, False]).head(3)
unmet = fg.sort_values(["Unmet Plan Qty", "Plan Cap"], ascending=[False, False]).head(3)
payload = {
    "shape": {
        "fg": len(tables[FG_DATASET]),
        "rm": len(tables[RM_DATASET]),
        "bom": len(tables[BOM_DATASET]),
        "capRows": len(tables[CAP_DATASET]),
    },
    "kpis": {
        "totalCapPairs": float(meta_map["TotalCapPairs"]),
        "achievedPairs": float(meta_map["AchievedPairs"]),
        "overallFillPairs": float(meta_map["OverallFillPairs"]),
        "planMarginMax": float(meta_map["PlanMarginMax"]),
        "achievedMargin": float(meta_map["AchievedMargin"]),
        "marginFillAtPairFill": float(meta_map["MarginFillAtPairFill"]),
        "phaseAStatus": str(meta_map["phase_a_status"]),
        "phaseAMethod": str(meta_map["phase_a_method"]),
        "purchaseSolver": str(meta_map["purchase_solver"]),
    },
    "purchaseRows": purchase_rows,
    "topRm": binding[["RM Code", "availability_total", "availability_used", "availability_remaining", "availability_utilization_pct"]].to_dict("records"),
    "topUnmet": unmet[["FG Code", "Plan Cap", "Opt Qty Total", "Fill_FG", "Unmet Plan Qty", "Likely Limiting RM"]].to_dict("records"),
}
print(json.dumps(payload))
`;
  const result = spawnSync("python", ["-c", code], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || "Python KPI extraction failed");
  }
  return JSON.parse(result.stdout);
}

const data = repoPython();
await fs.mkdir(OUT, { recursive: true });
await fs.mkdir(PREVIEWS, { recursive: true });
await fs.mkdir(LAYOUTS, { recursive: true });
await fs.mkdir(SCRATCH_ASSETS, { recursive: true });
await fs.writeFile(path.join(SCRATCH, "computed-kpis.json"), JSON.stringify(data, null, 2));

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const money = (value) => fmt.format(value);

function bg(color = C.bg) {
  return shape({ name: "slide-bg", width: fill, height: fill, fill: paint(color) });
}

function footer(textValue = "Sanitized repository demo data; amounts are sample currency units.", columnSpan = 1) {
  return text(textValue, {
    name: "footer",
    width: fill,
    height: hug,
    columnSpan,
    style: { fontFamily: FONT, fontSize: 14, color: C.quiet },
  });
}

function titleBlock(title, subtitle, options = {}) {
  return column(
    {
      name: options.name ?? "title-stack",
      width: fill,
      height: hug,
      gap: 14,
      columnSpan: options.columnSpan,
      rowSpan: options.rowSpan,
    },
    [
      text(title, {
        name: "slide-title",
        width: options.titleWidth ?? fill,
        height: hug,
        style: {
          fontFamily: DISPLAY,
          fontSize: options.size ?? 54,
          bold: true,
          color: options.color ?? C.ink,
        },
      }),
      subtitle
        ? text(subtitle, {
            name: "slide-subtitle",
            width: options.subtitleWidth ?? wrap(1180),
            height: hug,
            style: {
              fontFamily: FONT,
              fontSize: options.subtitleSize ?? 25,
              color: options.subtitleColor ?? C.muted,
            },
          })
        : rule({ name: "title-rule", width: fixed(190), stroke: C.amber, weight: 5 }),
    ].filter(Boolean),
  );
}

function label(value, color = C.teal) {
  return text(value, {
    name: "section-label",
    width: fill,
    height: hug,
    style: {
      fontFamily: FONT,
      fontSize: 16,
      bold: true,
      color,
      allCaps: true,
    },
  });
}

function metric(value, caption, options = {}) {
  return column(
    { name: options.name ?? "metric", width: fill, height: hug, gap: 4 },
    [
      text(value, {
        name: `${options.name ?? "metric"}-value`,
        width: fill,
        height: hug,
        style: {
          fontFamily: DISPLAY,
          fontSize: options.size ?? 58,
          bold: true,
          color: options.color ?? C.ink,
        },
      }),
      text(caption, {
        name: `${options.name ?? "metric"}-caption`,
        width: fill,
        height: hug,
        style: {
          fontFamily: FONT,
          fontSize: options.captionSize ?? 19,
          color: options.captionColor ?? C.muted,
        },
      }),
    ],
  );
}

function openBullets(items, color = C.ink) {
  return column(
    { name: "open-bullets", width: fill, height: hug, gap: 18 },
    items.map((item, i) =>
      row(
        { name: `bullet-row-${i + 1}`, width: fill, height: hug, gap: 14, align: "center" },
        [
          shape({
            name: `bullet-mark-${i + 1}`,
            width: fixed(12),
            height: fixed(12),
            fill: paint(i % 2 ? C.amber : C.teal),
            borderRadius: "rounded-full",
          }),
          text(item, {
            name: `bullet-text-${i + 1}`,
            width: fill,
            height: hug,
            style: { fontFamily: FONT, fontSize: 26, color },
          }),
        ],
      ),
    ),
  );
}

function smallPanel(name, child, options = {}) {
  return panel(
    {
      name,
      width: options.width ?? fill,
      height: options.height ?? hug,
      padding: options.padding ?? { x: 26, y: 22 },
      fill: paint(options.fill ?? C.paper),
      line: stroke(options.line ?? C.line),
      borderRadius: options.radius ?? "rounded-lg",
    },
    child,
  );
}

function simpleTable(name, columns, rows, options = {}) {
  const header = row(
    { name: `${name}-header`, width: fill, height: hug, gap: 12 },
    columns.map((col, i) =>
      text(col, {
        name: `${name}-header-${i}`,
        width: grow(options.widths?.[i] ?? 1),
        height: hug,
        style: { fontFamily: FONT, fontSize: 16, bold: true, color: C.tealDark },
      }),
    ),
  );
  const bodyRows = rows.map((r, ri) =>
    column(
      { name: `${name}-row-wrap-${ri}`, width: fill, height: hug, gap: 10 },
      [
        rule({ name: `${name}-row-rule-${ri}`, width: fill, stroke: C.line, weight: 1 }),
        row(
          { name: `${name}-row-${ri}`, width: fill, height: hug, gap: 12, align: "start" },
          r.map((cell, ci) =>
            text(cell, {
              name: `${name}-cell-${ri}-${ci}`,
              width: grow(options.widths?.[ci] ?? 1),
              height: hug,
              style: {
                fontFamily: FONT,
                fontSize: options.fontSize ?? 20,
                bold: ci === 0,
                color: ci === 0 ? C.ink : C.muted,
              },
            }),
          ),
        ),
      ],
    ),
  );
  return column({ name, width: fill, height: hug, gap: 14 }, [header, ...bodyRows]);
}

function addSlide(presentation, node) {
  const slide = presentation.slides.add();
  slide.compose(node, { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 });
  return slide;
}

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

async function stageAsset(fileName) {
  const src = path.join(ASSETS, fileName);
  const dst = path.join(SCRATCH_ASSETS, fileName);
  await fs.copyFile(src, dst);
  const bytes = await fs.readFile(dst);
  return `data:image/png;base64,${bytes.toString("base64")}`;
}

const coverImage = await stageAsset("cover-footwear-planning.png");
const complexityImage = await stageAsset("complexity-materials-planning.png");
const pilotImage = await stageAsset("pilot-workshop.png");

addSlide(
  presentation,
  layers({ name: "cover-root", width: fill, height: fill }, [
    image({
      name: "cover-photo",
      dataUrl: coverImage,
      contentType: "image/png",
      width: fill,
      height: fill,
      fit: "cover",
      alt: "Footwear production planning room beside a factory floor",
    }),
    grid(
      {
        name: "cover-grid",
        width: fill,
        height: fill,
        columns: [fr(0.92), fr(1.08)],
        padding: { x: 96, y: 82 },
      },
      [
        column(
          { name: "cover-copy", width: fill, height: fill, gap: 24, justify: "center" },
          [
            label("Client presentation", C.amberSoft),
            text("RM-Constrained FG Planning Optimizer", {
              name: "cover-title",
              width: wrap(760),
              height: hug,
              style: {
                fontFamily: DISPLAY,
                fontSize: 78,
                bold: true,
                color: C.white,
              },
            }),
            rule({ name: "cover-rule", width: fixed(260), stroke: C.amber, weight: 5 }),
            text("A practical algorithmic planning layer for footwear manufacturers facing raw-material scarcity, SKU proliferation, and time-sensitive production commitments.", {
              name: "cover-subtitle",
              width: wrap(720),
              height: hug,
              style: { fontFamily: FONT, fontSize: 28, color: "#EDE5DA" },
            }),
          ],
        ),
        text("", { name: "cover-spacer", width: fill, height: hug }),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s2-root", width: fill, height: fill }, [
    bg(),
    grid(
      {
        name: "s2-grid",
        width: fill,
        height: fill,
        columns: [fr(0.96), fr(1.04)],
        rows: [auto, fr(1), auto],
        padding: { x: 86, y: 64 },
        columnGap: 56,
        rowGap: 34,
      },
      [
        titleBlock(
          "The client problem starts on the factory floor",
          "A monthly footwear plan looks simple until the same leather, outsole, trim, and labor-adjacent materials are shared across dozens of styles.",
          { subtitleWidth: wrap(1320), columnSpan: 2 },
        ),
        column(
          { name: "s2-story", width: fill, height: fill, gap: 30 },
          [
            text("A practical use case", {
              name: "s2-usecase-head",
              width: fill,
              height: hug,
              style: { fontFamily: DISPLAY, fontSize: 42, bold: true, color: C.ink },
            }),
            openBullets([
              "Production wants the highest possible plan fill before the month closes.",
              "Procurement needs to know the smallest RM buy needed for each service-level target.",
              "Operations leaders need to see which finished goods are blocked and why.",
            ]),
          ],
        ),
        smallPanel(
          "s2-flow-panel",
          column(
            { name: "s2-flow", width: fill, height: fill, gap: 22 },
            [
              metric("50 FG", "styles in the sanitized demo", { name: "s2-m1", color: C.teal }),
              rule({ name: "s2-r1", width: fill, stroke: C.line, weight: 1 }),
              metric("200 RM", "raw materials competing across BOMs", { name: "s2-m2", color: C.leather }),
              rule({ name: "s2-r2", width: fill, stroke: C.line, weight: 1 }),
              metric("400 BOM rows", "the mapping that makes manual tradeoffs fragile", { name: "s2-m3", color: C.ink, size: 52 }),
            ],
          ),
          { height: fill, padding: { x: 36, y: 34 } },
        ),
        footer(undefined, 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s3-root", width: fill, height: fill }, [
    bg(C.paper),
    grid(
      {
        name: "s3-grid",
        width: fill,
        height: fill,
        columns: [fr(0.88), fr(1.12)],
        rows: [auto, fr(1), auto],
        padding: { x: 82, y: 60 },
        columnGap: 50,
        rowGap: 26,
      },
      [
        titleBlock(
          "Complexity grows multiplicatively, not linearly",
          "Every new SKU adds BOM consumption rows; every shared RM creates tradeoffs; every target creates another scenario to solve.",
          { columnSpan: 2, subtitleWidth: wrap(1350) },
        ),
        column(
          { name: "s3-left", width: fill, height: fill, gap: 28, justify: "center" },
          [
            metric("SKU x RM x BOM", "is the real planning surface", { name: "s3-metric", size: 62, color: C.tealDark }),
            smallPanel(
              "s3-scale-panel",
              simpleTable(
                "s3-scale-table",
                ["Planning layer", "Why it scales"],
                [
                  ["Styles", "more product promises to protect"],
                  ["Materials", "more shared constraints and substitutes"],
                  ["Availability mode", "stock vs stock plus PO changes feasibility"],
                  ["Targets", "25, 50, 75, 100 percent create distinct buy plans"],
                ],
                { widths: [0.78, 1.22], fontSize: 19 },
              ),
              { padding: { x: 28, y: 24 }, fill: "#FBF7F0" },
            ),
          ],
        ),
        image({
          name: "complexity-photo",
          dataUrl: complexityImage,
          contentType: "image/png",
          width: fill,
          height: fill,
          fit: "cover",
          alt: "Footwear components and planning materials arranged for a complexity discussion",
        }),
        footer("Generated image is illustrative; scale metrics are from the sanitized repository demo.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s4-root", width: fill, height: fill }, [
    bg(),
    grid(
      {
        name: "s4-grid",
        width: fill,
        height: fill,
        columns: [fr(1), fr(1.1)],
        rows: [auto, fr(1), auto],
        padding: { x: 88, y: 66 },
        columnGap: 58,
        rowGap: 36,
      },
      [
        titleBlock(
          "Spreadsheet planning leaks value in four places",
          "The issue is not spreadsheet skill. It is that the combinatorial tradeoff gets too large to reason through manually under deadline pressure.",
          { columnSpan: 2, subtitleWidth: wrap(1340) },
        ),
        column(
          { name: "s4-left", width: fill, height: fill, gap: 36, justify: "center" },
          [
            text("When planners work by hand, the first feasible answer can crowd out the best answer.", {
              name: "s4-claim",
              width: wrap(760),
              height: hug,
              style: { fontFamily: DISPLAY, fontSize: 52, bold: true, color: C.ink },
            }),
            rule({ name: "s4-rule", width: fixed(250), stroke: C.amber, weight: 5 }),
          ],
        ),
        grid(
          {
            name: "s4-pain-grid",
            width: fill,
            height: fill,
            columns: [fr(1), fr(1)],
            rows: [fr(1), fr(1)],
            columnGap: 22,
            rowGap: 22,
          },
          [
            smallPanel("s4-p1", metric("Time", "hours spent reconciling BOM, caps, and stock tables", { name: "s4-time", size: 44, color: C.teal }), { height: fill }),
            smallPanel("s4-p2", metric("Money", "unfocused buys can chase full shortages instead of target fills", { name: "s4-money", size: 44, color: C.leather }), { height: fill }),
            smallPanel("s4-p3", metric("Margin", "scarce RM may be consumed by lower-value styles first", { name: "s4-margin", size: 44, color: C.clay }), { height: fill }),
            smallPanel("s4-p4", metric("Opportunity", "missed production windows become missed customer commitments", { name: "s4-opp", size: 44, color: C.ink }), { height: fill }),
          ],
        ),
        footer(undefined, 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s5-root", width: fill, height: fill }, [
    bg(C.charcoal),
    grid(
      {
        name: "s5-grid",
        width: fill,
        height: fill,
        columns: [fr(1.06), fr(0.94)],
        rows: [auto, fr(1), auto],
        padding: { x: 90, y: 70 },
        columnGap: 60,
        rowGap: 34,
      },
      [
        titleBlock(
          "The optimization question is concrete",
          "Given scarce raw material, what should we build now, and what would it cost to buy our way to a higher fill target?",
          { columnSpan: 2, color: C.white, subtitleColor: "#C9D6D2", subtitleWidth: wrap(1320) },
        ),
        column(
          { name: "s5-question", width: fill, height: fill, gap: 32, justify: "center" },
          [
            text("Choose integer production quantities for every FG style.", {
              name: "s5-main",
              width: wrap(820),
              height: hug,
              style: { fontFamily: DISPLAY, fontSize: 56, bold: true, color: C.white },
            }),
            text("The answer must respect every raw-material constraint and every plan cap.", {
              name: "s5-support",
              width: wrap(740),
              height: hug,
              style: { fontFamily: FONT, fontSize: 28, color: "#DDE8E4" },
            }),
          ],
        ),
        column(
          { name: "s5-constraint-list", width: fill, height: fill, gap: 24, justify: "center" },
          [
            smallPanel("s5-c1", metric("x_fg", "decision variable: pairs to produce", { name: "s5-x", color: C.teal, size: 48 }), { fill: "#F6F1EA", height: hug }),
            smallPanel("s5-c2", metric("BOM x quantity <= RM available", "constraint: material usage cannot exceed stock or stock plus PO", { name: "s5-bom", color: C.leather, size: 42 }), { fill: "#F6F1EA", height: hug }),
            smallPanel("s5-c3", metric("0 <= x_fg <= plan cap", "constraint: production recommendation stays inside the plan envelope", { name: "s5-cap", color: C.ink, size: 42 }), { fill: "#F6F1EA", height: hug }),
          ],
        ),
        footer("Algorithm summary based on app/model.py and app/orchestrator.py.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s6-root", width: fill, height: fill }, [
    bg(),
    grid(
      {
        name: "s6-grid",
        width: fill,
        height: fill,
        columns: [fr(1), fr(0.56), fr(1)],
        rows: [auto, fr(1), auto],
        padding: { x: 82, y: 62 },
        columnGap: 34,
        rowGap: 32,
      },
      [
        titleBlock(
          "The project keeps the input model familiar",
          "Planners bring named Excel tables. The app validates the workbook, runs the solver, and returns an output workbook designed for review.",
          { columnSpan: 3, subtitleWidth: wrap(1380) },
        ),
        column(
          { name: "s6-inputs", width: fill, height: fill, gap: 18, justify: "center" },
          [
            label("Input workbook"),
            ...[
              ["fg_master", "FG code and unit margin"],
              ["bom_master", "FG to RM usage per pair"],
              ["tblFGPlanCap", "monthly plan cap by FG"],
              ["tblRMAvail", "stock, stock plus PO, RM rate"],
              ["tblControl_2", "objective and solver controls"],
            ].map(([a, b], i) =>
              smallPanel(
                `s6-in-${i}`,
                column({ width: fill, height: hug, gap: 3 }, [
                  text(a, { name: `s6-in-name-${i}`, width: fill, height: hug, style: { fontFamily: FONT, fontSize: 24, bold: true, color: C.ink } }),
                  text(b, { name: `s6-in-desc-${i}`, width: fill, height: hug, style: { fontFamily: FONT, fontSize: 17, color: C.muted } }),
                ]),
                { padding: { x: 22, y: 16 }, fill: C.paper },
              ),
            ),
          ],
        ),
        column(
          { name: "s6-middle", width: fill, height: fill, gap: 24, justify: "center", align: "center" },
          [
            shape({ name: "s6-core", width: fixed(210), height: fixed(210), fill: paint(C.teal), borderRadius: "rounded-full" }),
            text("HiGHS-backed LP/MIP optimizer", {
              name: "s6-core-label",
              width: wrap(260),
              height: hug,
              style: { fontFamily: DISPLAY, fontSize: 30, bold: true, color: C.tealDark },
            }),
          ],
        ),
        column(
          { name: "s6-outputs", width: fill, height: fill, gap: 18, justify: "center" },
          [
            label("Output workbook", C.leather),
            ...[
              ["FG_Result", "what to build and what is short"],
              ["RM_Diagnostic", "which materials are tight"],
              ["Purchase_Summary", "target fill vs buy cost"],
              ["Purchase_Detail", "RM-level buy quantities"],
              ["Run_Metadata", "status, timings, audit flags"],
            ].map(([a, b], i) =>
              smallPanel(
                `s6-out-${i}`,
                column({ width: fill, height: hug, gap: 3 }, [
                  text(a, { name: `s6-out-name-${i}`, width: fill, height: hug, style: { fontFamily: FONT, fontSize: 24, bold: true, color: C.ink } }),
                  text(b, { name: `s6-out-desc-${i}`, width: fill, height: hug, style: { fontFamily: FONT, fontSize: 17, color: C.muted } }),
                ]),
                { padding: { x: 22, y: 16 }, fill: "#FBF7F0" },
              ),
            ),
          ],
        ),
        footer("Workbook interface from README.md and app/config.py.", 3),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s7-root", width: fill, height: fill }, [
    bg(C.paper),
    grid(
      {
        name: "s7-grid",
        width: fill,
        height: fill,
        columns: [fr(1.1), fr(0.9)],
        rows: [auto, fr(1), auto],
        padding: { x: 86, y: 64 },
        columnGap: 54,
        rowGap: 30,
      },
      [
        titleBlock(
          "The solver turns planning into a constrained optimization model",
          "The code builds a sparse coefficient matrix from the BOM, then solves integer quantities against raw-material rows and plan caps.",
          { columnSpan: 2, subtitleWidth: wrap(1360) },
        ),
        column(
          { name: "s7-equation", width: fill, height: fill, gap: 26, justify: "center" },
          [
            text("Maximize objective", {
              name: "s7-eq-1",
              width: fill,
              height: hug,
              style: { fontFamily: DISPLAY, fontSize: 56, bold: true, color: C.ink },
            }),
            text("subject to", {
              name: "s7-eq-2",
              width: fill,
              height: hug,
              style: { fontFamily: FONT, fontSize: 26, color: C.quiet },
            }),
            text("RM usage <= availability\nFG quantity <= plan cap\nFG quantity is integer", {
              name: "s7-eq-3",
              width: wrap(780),
              height: hug,
              style: { fontFamily: FONT, fontSize: 38, bold: true, color: C.tealDark },
            }),
          ],
        ),
        smallPanel(
          "s7-routing",
          column(
            { name: "s7-routing-col", width: fill, height: fill, gap: 20 },
            [
              label("Objective routing"),
              simpleTable(
                "s7-routing-table",
                ["Mode", "What it optimizes"],
                [
                  ["PAIRS", "maximum total production pairs"],
                  ["MARGIN", "maximum total margin"],
                  ["PLAN", "pair fill first, then margin"],
                ],
                { widths: [0.5, 1.5], fontSize: 22 },
              ),
            ],
          ),
          { height: fill, padding: { x: 34, y: 30 }, fill: C.bg },
        ),
        footer("Implementation source: app/model.py and app/orchestrator.py.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s8-root", width: fill, height: fill }, [
    bg(),
    grid(
      {
        name: "s8-grid",
        width: fill,
        height: fill,
        columns: [fr(1), fr(1)],
        rows: [auto, fr(1), auto],
        padding: { x: 88, y: 64 },
        columnGap: 42,
        rowGap: 34,
      },
      [
        titleBlock(
          "PLAN objective uses a lexicographic solve",
          "For planners, service fill usually comes first. Once the pair count is locked, the optimizer chooses the best-margin mix at that same output level.",
          { columnSpan: 2, subtitleWidth: wrap(1340) },
        ),
        smallPanel(
          "s8-stage1",
          column(
            { name: "s8-stage1-col", width: fill, height: fill, gap: 24, justify: "center" },
            [
              label("Stage 1", C.teal),
              text("Maximize pairs", {
                name: "s8-stage1-title",
                width: fill,
                height: hug,
                style: { fontFamily: DISPLAY, fontSize: 60, bold: true, color: C.tealDark },
              }),
              text("Find P_star: the highest feasible production quantity under current RM availability and plan caps.", {
                name: "s8-stage1-copy",
                width: wrap(720),
                height: hug,
                style: { fontFamily: FONT, fontSize: 26, color: C.muted },
              }),
            ],
          ),
          { height: fill, padding: { x: 42, y: 38 }, fill: C.paper },
        ),
        smallPanel(
          "s8-stage2",
          column(
            { name: "s8-stage2-col", width: fill, height: fill, gap: 24, justify: "center" },
            [
              label("Stage 2", C.leather),
              text("Maximize margin at P_star", {
                name: "s8-stage2-title",
                width: fill,
                height: hug,
                style: { fontFamily: DISPLAY, fontSize: 60, bold: true, color: C.leather },
              }),
              text("Keep total pairs equal to P_star, then reallocate scarce RM toward the highest-value feasible FG mix.", {
                name: "s8-stage2-copy",
                width: wrap(720),
                height: hug,
                style: { fontFamily: FONT, fontSize: 26, color: C.muted },
              }),
            ],
          ),
          { height: fill, padding: { x: 42, y: 38 }, fill: "#FBF7F0" },
        ),
        footer("Fallback LP plus greedy heuristics are available when exact MIP status is not feasible in time.", 2),
      ],
    ),
  ]),
);

const purchase = data.purchaseRows;
const purchaseCategories = purchase.map((r) => `${r.targetFillPct.toFixed(0)}%`);
const purchaseCosts = purchase.map((r) => Math.round(r.totalBuyCost / 1000));

addSlide(
  presentation,
  layers({ name: "s9-root", width: fill, height: fill }, [
    bg(C.paper),
    grid(
      {
        name: "s9-grid",
        width: fill,
        height: fill,
        columns: [fr(0.86), fr(1.14)],
        rows: [auto, fr(1), auto],
        padding: { x: 84, y: 62 },
        columnGap: 48,
        rowGap: 28,
      },
      [
        titleBlock(
          "Purchase planning answers the next procurement question",
          "For each minimum fill target, the model minimizes extra RM buy cost while staying inside FG caps and BOM requirements.",
          { columnSpan: 2, subtitleWidth: wrap(1370) },
        ),
        column(
          { name: "s9-left", width: fill, height: fill, gap: 30, justify: "center" },
          [
            metric("Target fill", "is a minimum service threshold, not a forced exact output", { name: "s9-m1", color: C.tealDark, size: 56 }),
            metric("Buy quantity", "is solved RM by RM against material rates", { name: "s9-m2", color: C.leather, size: 56 }),
            text("This lets procurement compare the cost of moving from today's feasible plan to 50%, 75%, or full plan fill before committing cash.", {
              name: "s9-copy",
              width: wrap(700),
              height: hug,
              style: { fontFamily: FONT, fontSize: 26, color: C.muted },
            }),
          ],
        ),
        chart({
          name: "purchase-cost-chart",
          chartType: "bar",
          width: fill,
          height: fill,
          config: {
            title: "Extra RM buy cost by target (000s)",
            categories: purchaseCategories,
            series: [{ name: "Buy cost", values: purchaseCosts }],
          },
        }),
        footer("Chart uses recomputed Purchase_Summary values; costs shown in thousands of sample currency units.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s10-root", width: fill, height: fill }, [
    bg(),
    grid(
      {
        name: "s10-grid",
        width: fill,
        height: fill,
        columns: [fr(1)],
        rows: [auto, fr(1), auto],
        padding: { x: 86, y: 62 },
        rowGap: 26,
      },
      [
        titleBlock(
          "The output workbook is built for decision review",
          "It does not just return a solver number. It gives planners and procurement a traceable set of worksheets for production, bottlenecks, and buy scenarios.",
          { subtitleWidth: wrap(1380) },
        ),
        grid(
          {
            name: "s10-body",
            width: fill,
            height: fill,
            columns: [fr(0.98), fr(1.02)],
            columnGap: 42,
          },
          [
            smallPanel(
              "s10-table-panel",
              simpleTable(
                "s10-output-table",
                ["Sheet", "Decision it supports"],
                [
                  ["FG_Result", "What to build, how much is short, likely limiting RM"],
                  ["RM_Diagnostic", "Which raw materials are used up or still available"],
                  ["Purchase_Summary", "Target fill, achieved pairs, margin, and buy cost"],
                  ["Purchase_Detail", "RM-level buy quantities and cost lines"],
                  ["Run_Metadata", "Solver status, method, timings, and audit warnings"],
                ],
                { widths: [0.62, 1.38], fontSize: 22 },
              ),
              { height: fill, padding: { x: 30, y: 28 }, fill: "#FBF7F0" },
            ),
            column(
              { name: "s10-interpret", width: fill, height: fill, gap: 24, justify: "center" },
              [
                label("Interpretation flow", C.leather),
                text("1. Find under-filled FG in FG_Result.\n2. Confirm binding RM in RM_Diagnostic.\n3. Compare incremental buy cost in Purchase_Summary.\n4. Use Purchase_Detail for procurement handoff.", {
                  name: "s10-flow-copy",
                  width: wrap(780),
                  height: hug,
                  style: { fontFamily: FONT, fontSize: 34, bold: true, color: C.ink },
                }),
              ],
            ),
          ],
        ),
        footer("Workbook sheets are generated by app/excel_io.py and app/orchestrator.py."),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s11-root", width: fill, height: fill }, [
    bg(C.paper),
    grid(
      {
        name: "s11-grid",
        width: fill,
        height: fill,
        columns: [fr(0.8), fr(1.2)],
        rows: [auto, fr(1), auto],
        padding: { x: 84, y: 62 },
        columnGap: 50,
        rowGap: 28,
      },
      [
        titleBlock(
          "Sample run: stock limits cap production at 29.6% fill",
          "The bundled footwear-shaped demo shows how a small number of shared RM constraints can block many FG styles at once.",
          { columnSpan: 2, subtitleWidth: wrap(1340) },
        ),
        column(
          { name: "s11-left", width: fill, height: fill, gap: 26, justify: "center" },
          [
            metric(fmt.format(data.kpis.achievedPairs), `optimized pairs out of ${fmt.format(data.kpis.totalCapPairs)} plan cap`, { name: "s11-pairs", color: C.tealDark, size: 66 }),
            metric(pct(data.kpis.overallFillPairs), "current-stock plan fill", { name: "s11-fill", color: C.leather, size: 66 }),
            metric(money(data.kpis.achievedMargin), "achieved margin in sample data", { name: "s11-margin", color: C.ink, size: 52 }),
          ],
        ),
        column(
          { name: "s11-right", width: fill, height: fill, gap: 24 },
          [
            chart({
              name: "pairs-chart",
              chartType: "bar",
              width: fill,
              height: grow(1),
              config: {
                title: "Pairs: plan cap vs optimized current-stock output",
                categories: ["Plan cap", "Optimized"],
                series: [{ name: "Pairs", values: [Math.round(data.kpis.totalCapPairs), Math.round(data.kpis.achievedPairs)] }],
              },
            }),
            smallPanel(
              "s11-rm-panel",
              simpleTable(
                "s11-rm-table",
                ["Likely bottleneck", "Utilization"],
                data.topRm.map((r) => [r["RM Code"], `${(Number(r.availability_utilization_pct) * 100).toFixed(1)}% used`]),
                { widths: [1, 1], fontSize: 22 },
              ),
              { padding: { x: 28, y: 22 }, fill: C.bg },
            ),
          ],
        ),
        footer("Run status: Optimal via lex_mip; bottlenecks are from recomputed RM_Diagnostic utilization.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s12-root", width: fill, height: fill }, [
    bg(),
    grid(
      {
        name: "s12-grid",
        width: fill,
        height: fill,
        columns: [fr(1.05), fr(0.95)],
        rows: [auto, fr(1), auto],
        padding: { x: 88, y: 66 },
        columnGap: 60,
        rowGap: 36,
      },
      [
        titleBlock(
          "The savings come from better tradeoffs, faster",
          "The optimizer does not create material. It makes the scarcity visible, prioritizes feasible output, and prices the cost of moving to the next service level.",
          { columnSpan: 2, subtitleWidth: wrap(1360) },
        ),
        column(
          { name: "s12-left", width: fill, height: fill, gap: 28, justify: "center" },
          [
            text("Time saved", { name: "s12-time-title", width: fill, height: hug, style: { fontFamily: DISPLAY, fontSize: 50, bold: true, color: C.tealDark } }),
            text("Manual reconciliation becomes scenario review. Planners spend less time stitching worksheets and more time testing targets.", {
              name: "s12-time-copy",
              width: wrap(780),
              height: hug,
              style: { fontFamily: FONT, fontSize: 28, color: C.muted },
            }),
            rule({ name: "s12-rule", width: fixed(280), stroke: C.amber, weight: 5 }),
            text("Lost opportunity avoided", { name: "s12-opp-title", width: fill, height: hug, style: { fontFamily: DISPLAY, fontSize: 50, bold: true, color: C.leather } }),
            text("The FG_Result sheet shows which plan commitments are blocked before the month is gone.", {
              name: "s12-opp-copy",
              width: wrap(760),
              height: hug,
              style: { fontFamily: FONT, fontSize: 28, color: C.muted },
            }),
          ],
        ),
        smallPanel(
          "s12-money",
          column(
            { name: "s12-money-col", width: fill, height: fill, gap: 24, justify: "center" },
            [
              label("Money saved through controlled buys", C.leather),
              metric("818,600", "sample buy cost to reach 50% fill", { name: "s12-buy50", color: C.tealDark, size: 54 }),
              metric("3,291,426", "sample buy cost to reach 100% fill", { name: "s12-buy100", color: C.leather, size: 54 }),
              text("The business can choose the service-level step it can afford instead of blindly buying toward the full shortage.", {
                name: "s12-money-copy",
                width: wrap(700),
                height: hug,
                style: { fontFamily: FONT, fontSize: 24, color: C.muted },
              }),
            ],
          ),
          { height: fill, padding: { x: 38, y: 34 }, fill: C.paper },
        ),
        footer("Cost examples are sample currency units from the bundled Purchase_Summary.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s13-root", width: fill, height: fill }, [
    bg(C.paper),
    grid(
      {
        name: "s13-grid",
        width: fill,
        height: fill,
        columns: [fr(1), fr(1)],
        rows: [auto, fr(1), auto],
        padding: { x: 86, y: 64 },
        columnGap: 48,
        rowGap: 32,
      },
      [
        titleBlock(
          "Implementation is intentionally adoption-friendly",
          "The first version fits the client where planning already happens: Excel workbooks, a simple Streamlit upload flow, and auditable outputs.",
          { columnSpan: 2, subtitleWidth: wrap(1360) },
        ),
        smallPanel(
          "s13-fit",
          column(
            { name: "s13-fit-col", width: fill, height: fill, gap: 24 },
            [
              label("What is already in place"),
              openBullets([
                "Workbook template and sample input/output downloads.",
                "Schema validation before the solver runs.",
                "Progress heartbeat and solver settings for long runs.",
                "Run metadata for status, fallback flags, and audit warnings.",
              ]),
            ],
          ),
          { height: fill, padding: { x: 34, y: 30 }, fill: C.bg },
        ),
        smallPanel(
          "s13-limits",
          column(
            { name: "s13-limits-col", width: fill, height: fill, gap: 24 },
            [
              label("Not yet modeled", C.leather),
              openBullets([
                "Lead time and multi-period planning.",
                "Manpower, routing, and work-center sequencing.",
                "MOQ, supplier allocation, and cash timing.",
                "Vendor-specific constraints or substitution logic.",
              ]),
            ],
          ),
          { height: fill, padding: { x: 34, y: 30 }, fill: "#FBF7F0" },
        ),
        footer("Scope boundaries from README.md.", 2),
      ],
    ),
  ]),
);

addSlide(
  presentation,
  layers({ name: "s14-root", width: fill, height: fill }, [
    image({
      name: "pilot-photo",
      dataUrl: pilotImage,
      contentType: "image/png",
      width: fill,
      height: fill,
      fit: "cover",
      alt: "Footwear manufacturing planning team reviewing a pilot workflow",
    }),
    grid(
      {
        name: "s14-grid",
        width: fill,
        height: fill,
        columns: [fr(0.9), fr(1.1)],
        padding: { x: 92, y: 76 },
      },
      [
        column(
          { name: "s14-copy", width: fill, height: fill, gap: 26, justify: "center" },
          [
            label("Recommended next step", C.tealDark),
            text("Run one planning-cycle pilot with sanitized client data.", {
              name: "s14-title",
              width: wrap(750),
              height: hug,
              style: { fontFamily: DISPLAY, fontSize: 66, bold: true, color: C.ink },
            }),
            rule({ name: "s14-rule", width: fixed(260), stroke: C.amber, weight: 5 }),
            text("Start with current workbook structure, confirm output interpretation with planning and procurement, then add client-specific constraints only where they change decisions.", {
              name: "s14-copy-text",
              width: wrap(720),
              height: hug,
              style: { fontFamily: FONT, fontSize: 28, color: C.muted },
            }),
          ],
        ),
        text("", { name: "s14-spacer", width: fill, height: hug }),
      ],
    ),
  ]),
);

const pptx = await PresentationFile.exportPptx(presentation);
const workspacePptx = path.join(OUT, "output.pptx");
await pptx.save(workspacePptx);
await fs.copyFile(workspacePptx, path.join(DELIVERABLES, "lp-optimizer-client-presentation.pptx"));

for (let i = 0; i < presentation.slides.items.length; i += 1) {
  const slide = presentation.slides.items[i];
  const slideNo = String(i + 1).padStart(2, "0");
  const png = await slide.export({ format: "png" });
  await fs.writeFile(path.join(PREVIEWS, `slide-${slideNo}.png`), Buffer.from(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(LAYOUTS, `slide-${slideNo}.layout.json`), Buffer.from(await layout.arrayBuffer()));
}

await fs.writeFile(
  path.join(SCRATCH, "build-summary.json"),
  JSON.stringify(
    {
      deck: path.join(DELIVERABLES, "lp-optimizer-client-presentation.pptx"),
      workspacePptx,
      previews: PREVIEWS,
      layouts: LAYOUTS,
      slideCount: presentation.slides.items.length,
      kpis: data.kpis,
    },
    null,
    2,
  ),
);

console.log(JSON.stringify({ ok: true, pptx: path.join(DELIVERABLES, "lp-optimizer-client-presentation.pptx"), previews: PREVIEWS, slides: presentation.slides.items.length }, null, 2));
