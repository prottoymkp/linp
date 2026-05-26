import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS} from '../constants';
import {fade, pop} from '../utils/animation';

type HighlightRingProps = {
  x: number;
  y: number;
  width: number;
  height: number;
  start?: number;
  color?: string;
  label?: string;
};

export const HighlightRing: React.FC<HighlightRingProps> = ({
  x,
  y,
  width,
  height,
  start = 0,
  color = COLORS.green,
  label,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const scale = 0.94 + pop(frame, start, fps) * 0.06;

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width,
        height,
        border: `7px solid ${color}`,
        borderRadius: 26,
        opacity: fade(frame, start, 15),
        transform: `scale(${scale})`,
        transformOrigin: 'center',
        boxShadow: `0 0 0 8px ${color}22, 0 0 34px ${color}66`,
      }}
    >
      {label ? (
        <div
          style={{
            position: 'absolute',
            top: -52,
            left: 18,
            background: color,
            color: COLORS.white,
            borderRadius: 999,
            padding: '10px 18px',
            fontSize: 24,
            fontWeight: 900,
            whiteSpace: 'nowrap',
          }}
        >
          {label}
        </div>
      ) : null}
    </div>
  );
};
