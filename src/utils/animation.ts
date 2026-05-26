import {Easing, interpolate, spring} from 'remotion';

export const fade = (frame: number, start: number, duration = 18) =>
  interpolate(frame, [start, start + duration], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

export const slideY = (frame: number, start: number, distance = 30, duration = 22) =>
  interpolate(frame, [start, start + duration], [distance, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

export const pop = (frame: number, start: number, fps: number) =>
  spring({
    frame: Math.max(0, frame - start),
    fps,
    config: {
      damping: 18,
      stiffness: 120,
      mass: 0.7,
    },
  });

export const smooth = (
  frame: number,
  input: [number, number],
  output: [number, number],
) =>
  interpolate(frame, input, output, {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.cubic),
  });
