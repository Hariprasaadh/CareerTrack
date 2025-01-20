"use client";

import { navItems } from "@/data";

import Hero from "@/components/Hero";
import Grid from "@/components/Grid";
import Clients from "@/components/Clients";
import Approach from "@/components/Approach";
import { FloatingNav } from "@/components/ui/FloatingNavbar";
import { AnimatedTestimonialsDemo } from "@/components/AnimatedTestimonialsDemo";
import { SVGMaskEffectDemo } from "@/components/SVGMaskEffectDemo";
import { FooterDemo } from "@/components/FooterDemo";
import Programmes from "@/components/Programmes";




const Home = () => {
  return (
    <main className="relative bg-black-100 flex justify-center items-center flex-col overflow-hidden mx-auto sm:px-10 px-5">
      <div className="max-w-7xl w-full">
        <FloatingNav navItems={navItems} />
        <Hero />
        <Grid />
        <Programmes />

        <Clients />

        <AnimatedTestimonialsDemo />
        <SVGMaskEffectDemo />
        <Approach />
        <div className="">
          <FooterDemo />
        </div>
      </div>
    </main>
  );
};

export default Home;
