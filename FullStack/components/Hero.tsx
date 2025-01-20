import React from "react";
import { FaComments } from "react-icons/fa6";

import { TextGenerateEffect } from "./ui/TextGenerateEffect";
import { Spotlight } from "./ui/Spotlight";
import ParticlesComponent from "./Particles";

const Hero: React.FC = () => {
  return (
    <div className="pb-20 pt-36">
      <div>
        <Spotlight
          className="-top-40 -left-10 md:-left-32 md:-top-20 h-screen"
          fill="white"
        />
        <Spotlight
          className="h-[80vh] w-[50vw] top-10 left-full"
          fill="purple"
        />
        <Spotlight className="left-80 top-28 h-[80vh] w-[50vw]" fill="blue" />
      </div>

      <div
        className="h-screen w-full dark:bg-violet-950 bg-white dark:bg-grid-white/[0.03] bg-grid-black-100/[0.2]
        absolute top-0 left-0 flex items-center justify-center"
      >
        <div
          className="absolute pointer-events-none inset-0 flex items-center justify-center dark:bg-black-100
          bg-white [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]"
        />
      </div>

      {/* Chat Icon - Fixed Position */}
      <a
        href="localhost:5000"
        className="fixed bottom-8 right-8 z-50 bg-blue-500 hover:bg-blue-600 text-white p-4 rounded-full 
        shadow-lg transition-transform hover:scale-110 flex items-center justify-center"
        aria-label="Open Chat"
      >
        <FaComments className="text-2xl" />
      </a>

      <div className="flex justify-center relative my-20 z-10">
        <ParticlesComponent />
        <div className="max-w-[89vw] md:max-w-2xl lg:max-w-[60vw] flex flex-col items-center justify-center">
          <p className="uppercase tracking-widest text-xl text-center text-blue-100 max-w-80">
            CareerTrack
          </p>

          <TextGenerateEffect
            words="AI-powered career guidance and student wellness"
            className="text-center text-[40px] md:text-5xl lg:text-6xl"
          />

          <p className="text-center md:tracking-wider mb-4 text-sm md:text-lg lg:text-2xl">
            Empowering students with ai-driven career guidance, mental wellness
            support, and tools for academic success
          </p>
        </div>
      </div>
    </div>
  );
};

export default Hero;
