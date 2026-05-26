import React from 'react';
import {AbsoluteFill, Sequence, useCurrentFrame} from 'remotion';
import {ASSETS, COLORS, COPY, sceneFrames, sceneStarts} from './constants';
import {AssetZoomScene} from './components/AssetZoomScene';
import {CalloutBox} from './components/CalloutBox';
import {HighlightRing} from './components/HighlightRing';
import {SceneTitle} from './components/SceneTitle';
import {WorkflowStep} from './components/WorkflowStep';
import {fade, slideY} from './utils/animation';

const SoftBackground: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill
    style={{
      background: COLORS.background,
      color: COLORS.ink,
      fontFamily:
        'Aptos, Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      overflow: 'hidden',
    }}
  >
    {children}
  </AbsoluteFill>
);

const LowerCaption: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div
    style={{
      position: 'absolute',
      left: 86,
      right: 86,
      bottom: 44,
      color: COLORS.quiet,
      fontSize: 24,
      fontWeight: 650,
    }}
  >
    {children}
  </div>
);

const Arrow: React.FC<{start: number; left: number}> = ({start, left}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        position: 'absolute',
        left,
        top: 542,
        width: 108,
        opacity: fade(frame, start, 15),
        transform: `translateX(${slideY(frame, start, -30)}px)`,
      }}
    >
      <div
        style={{
          height: 5,
          background: COLORS.amber,
          borderRadius: 999,
        }}
      />
      <div
        style={{
          position: 'absolute',
          right: -2,
          top: -13,
          width: 0,
          height: 0,
          borderTop: '15px solid transparent',
          borderBottom: '15px solid transparent',
          borderLeft: `24px solid ${COLORS.amber}`,
        }}
      />
    </div>
  );
};

const HookScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AssetZoomScene
      asset={ASSETS.infographic}
      duration={sceneFrames.hook}
      dim={0.42}
      keyframes={[
        {frame: 0, scale: 1.08, x: 0, y: 0},
        {frame: sceneFrames.hook, scale: 1.55, x: 360, y: 170},
      ]}
    >
      <div
        style={{
          position: 'absolute',
          left: 68,
          top: 78,
          width: 930,
          height: 360,
          background: COLORS.paper,
          borderLeft: `8px solid ${COLORS.amber}`,
          borderRadius: 30,
          boxShadow: '0 26px 70px rgba(21, 32, 29, 0.16)',
          opacity: 1,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 90,
          top: 104,
          width: 880,
          opacity: fade(frame, -8, 18),
          transform: `translateY(${slideY(frame, -8, 36)}px)`,
        }}
      >
        <SceneTitle
          kicker={COPY.hook.kicker}
          title={COPY.hook.headline}
          subtitle={COPY.hook.caption}
          color={COLORS.ink}
          maxWidth={840}
        />
      </div>
      <div style={{position: 'absolute', left: 90, bottom: 118}}>
        <CalloutBox start={106} tone="amber" width={760}>
          <span style={{whiteSpace: 'pre-line'}}>{COPY.hook.secondLine}</span>
        </CalloutBox>
      </div>
    </AssetZoomScene>
  );
};

const HiddenCombinationScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AssetZoomScene
      asset={ASSETS.tradeoffs}
      duration={sceneFrames.hiddenCombination}
      keyframes={[
        {frame: 0, scale: 1.0, x: 0, y: 0},
        {frame: sceneFrames.hiddenCombination, scale: 1.55, x: -210, y: 20},
      ]}
    >
      <div
        style={{
          position: 'absolute',
          left: 56,
          top: 38,
          width: 830,
          height: 275,
          background: COLORS.paper,
          borderLeft: `8px solid ${COLORS.navy}`,
          borderRadius: 28,
          boxShadow: '0 24px 60px rgba(21, 32, 29, 0.14)',
        }}
      />
      <div style={{position: 'absolute', left: 76, top: 58, width: 760}}>
        <SceneTitle
          title={COPY.hiddenCombination.title}
          subtitle={COPY.hiddenCombination.subtitle}
          maxWidth={760}
        />
      </div>
      <div style={{position: 'absolute', right: 80, top: 150, display: 'flex', flexDirection: 'column', gap: 20}}>
        <CalloutBox start={60} tone="navy" width={585} compact>
          {COPY.hiddenCombination.callouts[0]}
        </CalloutBox>
        <CalloutBox start={155} tone="amber" width={585} compact>
          {COPY.hiddenCombination.callouts[1]}
        </CalloutBox>
        <CalloutBox start={250} tone="red" width={585} compact>
          {COPY.hiddenCombination.callouts[2]}
        </CalloutBox>
      </div>
      <HighlightRing x={690} y={260} width={690} height={360} start={120} color={COLORS.amber} label="Material combinations" />
      <div
        style={{
          position: 'absolute',
          left: 350,
          right: 350,
          bottom: 72,
          opacity: fade(frame, 332, 20),
          transform: `translateY(${slideY(frame, 332, 26)}px)`,
          background: COLORS.navy,
          color: COLORS.white,
          borderRadius: 999,
          padding: '22px 34px',
          textAlign: 'center',
          fontSize: 34,
          fontWeight: 900,
          boxShadow: '0 18px 44px rgba(14, 46, 79, 0.25)',
        }}
      >
        {COPY.hiddenCombination.final}
      </div>
    </AssetZoomScene>
  );
};

const PortfolioDilemmaScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AssetZoomScene
      asset={ASSETS.tradeoffs}
      duration={sceneFrames.portfolioDilemma}
      keyframes={[
        {frame: 0, scale: 1.08, x: 0, y: -112},
        {frame: sceneFrames.portfolioDilemma, scale: 1.18, x: 0, y: -170},
      ]}
    >
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 0,
          height: 180,
          background: 'rgba(247, 244, 238, 0.94)',
          borderBottom: `1px solid ${COLORS.border}`,
        }}
      />
      <div style={{position: 'absolute', left: 86, top: 52}}>
        <SceneTitle title={COPY.portfolio.title} maxWidth={760} />
      </div>
      <HighlightRing x={205} y={545} width={675} height={335} start={58} color={COLORS.red} label="Choice 1" />
      <HighlightRing x={995} y={545} width={700} height={335} start={148} color={COLORS.green} label="Choice 2" />
      <div
        style={{
          position: 'absolute',
          left: 410,
          right: 410,
          bottom: 38,
          opacity: fade(frame, 330, 22),
          transform: `translateY(${slideY(frame, 330, 24)}px)`,
          background: COLORS.paper,
          border: `3px solid ${COLORS.navy}`,
          borderRadius: 28,
          padding: '20px 30px',
          textAlign: 'center',
          fontSize: 32,
          lineHeight: 1.12,
          fontWeight: 950,
          color: COLORS.navy,
          boxShadow: '0 24px 54px rgba(21, 32, 29, 0.15)',
        }}
      >
        {COPY.portfolio.key}
      </div>
    </AssetZoomScene>
  );
};

const QuantityProfitScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AssetZoomScene
      asset={ASSETS.quantityProfit}
      duration={sceneFrames.quantityVsProfit}
      keyframes={[
        {frame: 0, scale: 1.0, x: 0, y: 0},
        {frame: sceneFrames.quantityVsProfit, scale: 1.04, x: 0, y: -10},
      ]}
    >
      <HighlightRing x={270} y={410} width={645} height={430} start={82} color={COLORS.navy} />
      <HighlightRing x={995} y={410} width={650} height={430} start={268} color={COLORS.green} />
    </AssetZoomScene>
  );
};

const WorkflowScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SoftBackground>
      <div style={{position: 'absolute', left: 82, top: 62}}>
        <SceneTitle
          title={COPY.workflow.title}
          subtitle={COPY.workflow.subtitle}
          maxWidth={1180}
        />
      </div>
      <div
        style={{
          position: 'absolute',
          left: 84,
          right: 84,
          top: 300,
          display: 'flex',
          gap: 86,
          alignItems: 'stretch',
        }}
      >
        <WorkflowStep
          title="Input"
          icon="IN"
          items={COPY.workflow.input}
          start={28}
          accent={COLORS.navy}
        />
        <WorkflowStep
          title="LP Optimizer"
          icon="MIX"
          items={COPY.workflow.optimizer}
          start={94}
          accent={COLORS.green}
        />
        <WorkflowStep
          title="Output"
          icon="OUT"
          items={COPY.workflow.output}
          start={166}
          accent={COLORS.amber}
        />
      </div>
      <Arrow start={78} left={600} />
      <Arrow start={146} left={1218} />
      <div
        style={{
          position: 'absolute',
          left: 450,
          right: 450,
          bottom: 58,
          opacity: fade(frame, 385, 20),
          transform: `translateY(${slideY(frame, 385, 22)}px)`,
          borderRadius: 999,
          padding: '18px 28px',
          background: COLORS.navy,
          color: COLORS.white,
          fontSize: 30,
          lineHeight: 1.18,
          fontWeight: 900,
          textAlign: 'center',
        }}
      >
        One workbook in. A decision-ready plan out.
      </div>
    </SoftBackground>
  );
};

const ClosingScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SoftBackground>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'linear-gradient(135deg, rgba(31,122,77,0.08) 0%, transparent 34%, transparent 68%, rgba(217,145,46,0.1) 100%)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 130,
          top: 155,
          width: 1060,
          opacity: fade(frame, 18, 22),
          transform: `translateY(${slideY(frame, 18, 36)}px)`,
        }}
      >
        <SceneTitle title={COPY.closing.title} subtitle={COPY.closing.tagline} maxWidth={980} />
      </div>
      <div
        style={{
          position: 'absolute',
          left: 130,
          top: 480,
          width: 780,
          color: COLORS.quiet,
          fontSize: 30,
          fontWeight: 700,
          opacity: fade(frame, 72, 22),
        }}
      >
        {COPY.closing.footer}
      </div>
      <div
        style={{
          position: 'absolute',
          left: 130,
          bottom: 126,
          width: 1160,
          opacity: fade(frame, 170, 24),
          transform: `translateY(${slideY(frame, 170, 34)}px)`,
          background: COLORS.paper,
          border: `4px solid ${COLORS.green}`,
          borderRadius: 34,
          padding: '34px 42px',
          boxShadow: '0 26px 70px rgba(21, 32, 29, 0.14)',
          color: COLORS.ink,
          whiteSpace: 'pre-line',
          fontSize: 42,
          lineHeight: 1.13,
          fontWeight: 950,
        }}
      >
        {COPY.closing.challenge}
      </div>
      <LowerCaption>
        Practical manufacturing case study, built to work even without voiceover.
      </LowerCaption>
    </SoftBackground>
  );
};

export const LPOptimizerDemo: React.FC = () => {
  return (
    <AbsoluteFill>
      <Sequence from={sceneStarts.hook} durationInFrames={sceneFrames.hook}>
        <HookScene />
      </Sequence>
      <Sequence
        from={sceneStarts.hiddenCombination}
        durationInFrames={sceneFrames.hiddenCombination}
      >
        <HiddenCombinationScene />
      </Sequence>
      <Sequence
        from={sceneStarts.portfolioDilemma}
        durationInFrames={sceneFrames.portfolioDilemma}
      >
        <PortfolioDilemmaScene />
      </Sequence>
      <Sequence
        from={sceneStarts.quantityVsProfit}
        durationInFrames={sceneFrames.quantityVsProfit}
      >
        <QuantityProfitScene />
      </Sequence>
      <Sequence from={sceneStarts.workflow} durationInFrames={sceneFrames.workflow}>
        <WorkflowScene />
      </Sequence>
      <Sequence from={sceneStarts.closing} durationInFrames={sceneFrames.closing}>
        <ClosingScene />
      </Sequence>
    </AbsoluteFill>
  );
};
