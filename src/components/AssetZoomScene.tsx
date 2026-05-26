import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {COLORS} from '../constants';
import {smooth} from '../utils/animation';

type ZoomKeyframe = {
  frame: number;
  scale: number;
  x: number;
  y: number;
};

type AssetZoomSceneProps = {
  asset: string;
  duration: number;
  keyframes: [ZoomKeyframe, ZoomKeyframe];
  children?: React.ReactNode;
  dim?: number;
};

export const AssetZoomScene: React.FC<AssetZoomSceneProps> = ({
  asset,
  duration,
  keyframes,
  children,
  dim = 0,
}) => {
  const frame = useCurrentFrame();
  const [from, to] = keyframes;
  const scale = smooth(frame, [from.frame, Math.min(to.frame, duration)], [from.scale, to.scale]);
  const x = smooth(frame, [from.frame, Math.min(to.frame, duration)], [from.x, to.x]);
  const y = smooth(frame, [from.frame, Math.min(to.frame, duration)], [from.y, to.y]);

  return (
    <AbsoluteFill style={{background: COLORS.background, overflow: 'hidden'}}>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transform: `translate(${x}px, ${y}px) scale(${scale})`,
          transformOrigin: 'center center',
        }}
      >
        <Img
          src={staticFile(asset)}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            filter: 'drop-shadow(0 26px 64px rgba(20, 32, 29, 0.14))',
          }}
        />
      </div>
      {dim > 0 ? (
        <AbsoluteFill style={{background: `rgba(247, 244, 238, ${dim})`}} />
      ) : null}
      {children}
    </AbsoluteFill>
  );
};
