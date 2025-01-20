import React from "react";
import { Vortex } from "./ui/vortex";

export function FooterDemo() {
  return (
    <div className="w-[calc(100%-4rem)] mx-auto rounded-md h-[30rem] overflow-hidden">
      <Vortex
        backgroundColor="transparent" // Set the background color to transparent
        className="flex items-center flex-col justify-center px-2 md:px-10 py-4 w-full h-full"
      >
        <h1 className="heading lg:max-w-[45vw]">
          Ready to elevate <span className="text-purple">your</span> career
          journey with CareerPulse?
        </h1>
        <p className="text-white-200 md:mt-10 my-5 text-center">
          Reach out to me today and let&apos;s discuss how I can help you
          achieve your goals.
        </p>
        <div className="flex flex-col sm:flex-row items-center gap-4 mt-6">
          <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 transition duration-200 rounded-lg text-white shadow-[0px_2px_0px_0px_#FFFFFF40_inset]">
            Contact Us
          </button>
        
        </div>
      </Vortex>
    </div>
  );
}
