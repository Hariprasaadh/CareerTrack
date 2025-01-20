"use client";
import { MaskContainer } from "./ui/svg-mask-effect";

export function SVGMaskEffectDemo() {
  return (
    <div className="h-[40rem] w-full flex items-center justify-center  overflow-hidden">
      <MaskContainer
        revealText={
          <p className="max-w-4xl mx-auto text-slate-800 text-center  text-4xl font-bold">
            The first rule of Career Track is you focus on your growth. The
            second rule of Career Track  is you empower others to grow.
          </p>
        }
        className="h-[40rem] border rounded-md"
      >
        The first rule of <span className="text-red-500">Career Track </span> is
        you focus on your growth. The second rule of Career Track  is you
        empower <span className="text-red-500">others to grow.</span>.
      </MaskContainer>
    </div>
  );
}
