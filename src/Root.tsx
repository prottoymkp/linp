import {Composition} from 'remotion';
import {LPOptimizerDemo} from './LPOptimizerDemo';
import {TOTAL_DURATION_FRAMES, VIDEO} from './constants';

export const Root: React.FC = () => {
  return (
    <Composition
      id="LPOptimizerDemo"
      component={LPOptimizerDemo}
      durationInFrames={TOTAL_DURATION_FRAMES}
      fps={VIDEO.fps}
      width={VIDEO.width}
      height={VIDEO.height}
    />
  );
};
