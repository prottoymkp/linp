import React from 'react';
import {COLORS} from '../constants';

type SceneTitleProps = {
  kicker?: string;
  title: string;
  subtitle?: string;
  align?: 'left' | 'center';
  color?: string;
  maxWidth?: number;
};

export const SceneTitle: React.FC<SceneTitleProps> = ({
  kicker,
  title,
  subtitle,
  align = 'left',
  color = COLORS.ink,
  maxWidth = 1180,
}) => {
  return (
    <div
      style={{
        maxWidth,
        textAlign: align,
        color,
      }}
    >
      {kicker ? (
        <div
          style={{
            color: COLORS.amber,
            fontSize: 26,
            fontWeight: 800,
            letterSpacing: 1.2,
            textTransform: 'uppercase',
            marginBottom: 18,
          }}
        >
          {kicker}
        </div>
      ) : null}
      <div
        style={{
          whiteSpace: 'pre-line',
          fontSize: 62,
          lineHeight: 1.04,
          fontWeight: 900,
          letterSpacing: 0,
        }}
      >
        {title}
      </div>
      {subtitle ? (
        <div
          style={{
            whiteSpace: 'pre-line',
            marginTop: 20,
            fontSize: 30,
            lineHeight: 1.25,
            color: COLORS.muted,
            fontWeight: 500,
          }}
        >
          {subtitle}
        </div>
      ) : null}
    </div>
  );
};
