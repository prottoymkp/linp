import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS} from '../constants';
import {fade, pop, slideY} from '../utils/animation';

type WorkflowStepProps = {
  title: string;
  items: string[];
  start: number;
  accent: string;
  icon: string;
};

export const WorkflowStep: React.FC<WorkflowStepProps> = ({
  title,
  items,
  start,
  accent,
  icon,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const scale = 0.97 + pop(frame, start, fps) * 0.03;

  return (
    <div
      style={{
        flex: 1,
        minHeight: 570,
        borderRadius: 28,
        border: `3px solid ${accent}`,
        background: COLORS.paper,
        padding: '38px 36px',
        opacity: fade(frame, start, 18),
        transform: `translateY(${slideY(frame, start, 38)}px) scale(${scale})`,
        boxShadow: '0 22px 54px rgba(21, 32, 29, 0.1)',
      }}
    >
      <div
        style={{
          width: 78,
          height: 78,
          borderRadius: 22,
          background: `${accent}18`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 25,
          color: accent,
          fontWeight: 950,
          letterSpacing: 0,
          marginBottom: 26,
        }}
      >
        {icon}
      </div>
      <div
        style={{
          color: accent,
          fontSize: 40,
          lineHeight: 1.04,
          fontWeight: 950,
          marginBottom: 28,
        }}
      >
        {title}
      </div>
      <div style={{display: 'flex', flexDirection: 'column', gap: 18}}>
        {items.map((item, index) => (
          <div
            key={item}
            style={{
              opacity: fade(frame, start + 10 + index * 5, 12),
              display: 'flex',
              gap: 14,
              alignItems: 'flex-start',
              color: COLORS.ink,
              fontSize: 29,
              lineHeight: 1.18,
              fontWeight: 700,
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: 999,
                background: accent,
                flex: '0 0 auto',
                marginTop: 13,
              }}
            />
            <span>{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
