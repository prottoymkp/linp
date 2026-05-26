import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS} from '../constants';
import {fade, pop, slideY} from '../utils/animation';

type CalloutBoxProps = {
  children: React.ReactNode;
  start?: number;
  tone?: 'navy' | 'green' | 'red' | 'amber' | 'neutral';
  width?: number;
  compact?: boolean;
};

const toneMap = {
  navy: {border: COLORS.navy, bg: '#EEF4FA', text: COLORS.navy},
  green: {border: COLORS.green, bg: COLORS.greenSoft, text: COLORS.green},
  red: {border: COLORS.red, bg: COLORS.redSoft, text: COLORS.red},
  amber: {border: COLORS.amber, bg: COLORS.amberSoft, text: COLORS.amber},
  neutral: {border: COLORS.border, bg: COLORS.paper, text: COLORS.ink},
};

export const CalloutBox: React.FC<CalloutBoxProps> = ({
  children,
  start = 0,
  tone = 'neutral',
  width = 520,
  compact = false,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = toneMap[tone];
  const scale = 0.96 + pop(frame, start, fps) * 0.04;

  return (
    <div
      style={{
        width,
        opacity: fade(frame, start, 16),
        transform: `translateY(${slideY(frame, start, 28)}px) scale(${scale})`,
        transformOrigin: 'left center',
        background: t.bg,
        border: `3px solid ${t.border}`,
        borderRadius: 22,
        padding: compact ? '18px 22px' : '24px 28px',
        boxShadow: '0 18px 42px rgba(18, 32, 29, 0.12)',
        color: t.text,
        fontSize: compact ? 30 : 34,
        fontWeight: 850,
        lineHeight: 1.15,
      }}
    >
      {children}
    </div>
  );
};
