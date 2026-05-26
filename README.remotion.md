# LP Optimizer Remotion Demo

LinkedIn-ready Remotion animation for the LP Optimizer manufacturing planning case study.

## Run

```bash
npm install
npm start
```

## Render

```bash
npm run render
```

The main composition is `LPOptimizerDemo` at `1920x1080`, `30fps`, and `85s`.

## Edit

- Scene timing, colors, copy, and asset filenames live in `src/constants.ts`.
- Reusable components live in `src/components/`.
- Case study images are loaded with `staticFile()` from `public/assets/`.
- Replace the fallback PNGs in `public/assets/` with your final case study images using the same filenames.
