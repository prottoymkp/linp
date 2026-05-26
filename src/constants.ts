export const VIDEO = {
  width: 1920,
  height: 1080,
  fps: 30,
};

// Easy timing edits: all durations are in seconds, converted below.
export const SCENE_SECONDS = {
  hook: 8,
  hiddenCombination: 14,
  portfolioDilemma: 16,
  quantityVsProfit: 17,
  workflow: 17,
  closing: 13,
};

export const sceneFrames = Object.fromEntries(
  Object.entries(SCENE_SECONDS).map(([key, seconds]) => [key, seconds * VIDEO.fps]),
) as Record<keyof typeof SCENE_SECONDS, number>;

export const TOTAL_DURATION_FRAMES = Object.values(sceneFrames).reduce(
  (sum, frames) => sum + frames,
  0,
);

export const sceneStarts = {
  hook: 0,
  hiddenCombination: sceneFrames.hook,
  portfolioDilemma: sceneFrames.hook + sceneFrames.hiddenCombination,
  quantityVsProfit:
    sceneFrames.hook + sceneFrames.hiddenCombination + sceneFrames.portfolioDilemma,
  workflow:
    sceneFrames.hook +
    sceneFrames.hiddenCombination +
    sceneFrames.portfolioDilemma +
    sceneFrames.quantityVsProfit,
  closing:
    sceneFrames.hook +
    sceneFrames.hiddenCombination +
    sceneFrames.portfolioDilemma +
    sceneFrames.quantityVsProfit +
    sceneFrames.workflow,
};

// Asset filenames: replace the files in public/assets without changing code.
export const ASSETS = {
  infographic: 'assets/a_4_panel_infographic_slide_set_2x2_grid_about_i.png',
  tradeoffs: 'assets/inventory_trade_offs_and_material_management.png',
  quantityProfit: 'assets/maximizing_quantity_vs_maximizing_profit.png',
};

export const COLORS = {
  background: '#F7F4EE',
  paper: '#FFFDF8',
  ink: '#15201D',
  muted: '#52615B',
  quiet: '#81908A',
  navy: '#0E2E4F',
  navy2: '#123E68',
  green: '#1F7A4D',
  greenSoft: '#DDEFE5',
  red: '#C84C42',
  redSoft: '#F5DFDC',
  amber: '#D9912E',
  amberSoft: '#F6E3C5',
  border: '#D8CABB',
  white: '#FFFFFF',
};

// Scene copy: edit these short lines first when tailoring the LinkedIn story.
// They are also the optional narration beats if you later add voiceover audio.
export const COPY = {
  hook: {
    kicker: 'The hidden production blocker',
    headline: 'Inventory is available.\nProduction is still stuck.',
    secondLine: 'The problem is not stock quantity.\nIt is the wrong combination.',
    caption: 'A factory can have material on hand and still miss the finished-good mix it planned.',
  },
  hiddenCombination: {
    title: 'The hidden combination problem',
    subtitle: 'The shortage is often in the match between products and materials.',
    callouts: [
      'More SKUs = more combinations',
      'Shared materials compete across products',
      'Unique materials quietly block specific products',
    ],
    final: 'The bottleneck is not always visible from stock quantity.',
  },
  portfolio: {
    title: 'The portfolio dilemma',
    red: 'Pushing one SKU may consume scarce materials.',
    green: 'Balanced production may unlock more total output.',
    key: 'The best plan is not always the most obvious plan.',
  },
  quantityProfit: {
    title: 'Quantity vs profit dilemma',
    left: 'Sometimes the highest quantity plan gives lower profit.',
    right: 'Sometimes fewer pairs create higher profit.',
    key: "There is no single 'best' plan.\nThe best answer depends on the business goal.",
  },
  workflow: {
    title: 'What LP Optimizer does',
    subtitle: 'It turns the planning workbook into a practical production decision.',
    input: ['Product list', 'BOM', 'Current RM stock', 'Production plan'],
    optimizer: ['Checks combinations', 'Respects limitations', 'Finds best feasible mix'],
    output: [
      'What can be produced now',
      'Which materials are blocking the plan',
      'What to buy to unlock more production',
      'Quantity-focused or profit-focused planning',
    ],
  },
  closing: {
    title: 'LP Optimizer',
    tagline: 'Turning inventory data into better production decisions.',
    footer: 'Built for RM-constrained FG planning',
    challenge: 'Test the tool. Challenge the logic.\nTell me where it breaks in real factory conditions.',
  },
};
