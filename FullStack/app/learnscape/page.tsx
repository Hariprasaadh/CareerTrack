"use client";
import React from "react";
import { FaLocationArrow } from "react-icons/fa6";
import MagicButton from "../../components/MagicButton";
import { TextGenerateEffect } from "../../components/ui/TextGenerateEffect";
import { Spotlight } from "../../components/ui/Spotlight";

import { PinContainer } from "@/components/ui/Pin";
import { projects_1 } from "@/data";
import StarBackground from "@/components/StarBackground";

const Hero: React.FC = () => {
  return (
    <div className="pb-20 pt-36">
      {/** Spotlights */}
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

      {/** Grid */}
      <div
        className="h-screen w-full dark:bg-black-100 bg-white dark:bg-grid-white/[0.03] bg-grid-black-100/[0.2]
        absolute top-0 left-0 flex items-center justify-center"
      >
        <div
          className="absolute pointer-events-none inset-0 flex items-center justify-center dark:bg-black-100
          bg-white [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]"
        />
      </div>

      <div className="flex justify-center relative my-20 z-10">
        <StarBackground />
        <div className="max-w-[89vw] md:max-w-2xl lg:max-w-[60vw] flex flex-col items-center justify-center">
                  <p className="uppercase tracking-widest text-6xl text-center text-blue-100 max-w-80">
                    LearnScape
                  </p>
        
                  <TextGenerateEffect
                    words="Your personalized learning landscape, tailored to all your academic needs."
                    className="text-center text-[25px]"
                  />
                </div>
      </div>
    </div>
  );
};

const RecentProjects: React.FC = () => {
  return (
    <div className="py-20">
      <h1 className="heading">
        A small selection of <span className="text-purple">Our Programmes</span>
      </h1>
      <div className="flex flex-wrap items-center justify-center p-4 gap-40">
        {projects_1.map((item) => (
          <div
            className="lg:min-h-[32.5rem] h-[25rem] flex items-center justify-center sm:w-96 w-[80vw]"
            key={item.id}
          >
            <PinContainer
              title=""
              href=""
            >
              <div className="relative flex items-center justify-center sm:w-96 w-[80vw] overflow-hidden h-[20vh] lg:h-[30vh] mb-10">
                <div
                  className="relative w-full h-full overflow-hidden lg:rounded-3xl"
                  style={{ backgroundColor: "#13162D" }}
                >
                  <img src="/bg.png" alt="bgimg" />
                </div>
                <img
                  src={item.img}
                  alt="cover"
                  className="z-10 absolute bottom-0"
                />
              </div>

              <h1 className="font-bold lg:text-2xl md:text-xl text-base line-clamp-1">
                {item.title}
              </h1>

              <p
                className="lg:text-xl lg:font-normal font-light text-sm line-clamp-2"
                style={{
                  color: "#BEC1DD",
                  margin: "1vh 0",
                }}
              >
                {item.des}
              </p>

              <div className="flex items-center justify-between mt-7 mb-3">
                <a href={item.link} className="flex items-center">
                  <p className="flex lg:text-xl md:text-xs text-sm text-purple">
                    Check Live Site
                  </p>
                  <FaLocationArrow className="ms-3" color="#CBACF9" />
                </a>
              </div>
            </PinContainer>
          </div>
        ))}
      </div>
    </div>
  );
};

const Main: React.FC = () => {
  return (
    <main className="relative bg-black-100 flex justify-center items-center flex-col overflow-hidden mx-auto sm:px-10 px-5">
      <Hero />
      <RecentProjects />
    </main>
  );
};

export default Main;
