import { useState } from "react";

import { cn } from "@/lib/utils";
import GridGlobe from "./GridGlobe";
import animationData from "@/data/confetti.json";


export const BentoGrid = ({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        "grid grid-cols-1 md:grid-cols-6 lg:grid-cols-5 gap-4 lg:gap-8 mx-auto",
        className
      )}
    >
      {children}
    </div>
  );
};

export const BentoGridItem = ({
  className,
  id,
  title,
  description,
  img,
  imgClassName,
  titleClassName,
  spareImg,
}: {
  className?: string;
  id: number;
  title?: string | React.ReactNode;
  description?: string | React.ReactNode;
  img?: string;
  imgClassName?: string;
  titleClassName?: string;
  spareImg?: string;
}) => {
  const [copied, setCopied] = useState(false);

  const defaultOptions = {
    loop: copied,
    autoplay: copied,
    animationData: animationData,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice",
    },
  };

  const handleCopy = () => {
    const text = "contact@careertrack.com";
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 3000);
  };

  return (
    <div
      className={cn(
        "row-span-1 relative overflow-hidden rounded-3xl border border-white/[0.1] group/bento hover:shadow-xl transition duration-200 shadow-input dark:shadow-none justify-between flex flex-col space-y-4 text-green ",
        className
      )}
      style={{
        background: "rgb(4,7,29)",
        backgroundImage:
          "linear-gradient(90deg, rgba(4,7,29,1) 0%, rgba(12,14,35,1) 100%)",
      }}
    >
      <div className={`${id === 4 ? "flex justify-center" : ""} h-15`}>
        <div className="w-full h-[450px] absolute">
          {img && (
            <img
              src={img}
              alt="content"
              className={cn(imgClassName, "object-cover object-center")}
            />
          )}
        </div>
        <div
          className={`absolute right-0 -bottom-5 ${
            id === 5 ? "w-full opacity-80" : ""
          }`}
        >
          {spareImg && (
            <img
              src={spareImg}
              alt="additional content"
              className="object-cover object-center w-full h-full"
            />
          )}
        </div>

        {id === 2 && <GridGlobe />}

        <div
          className={cn(
            titleClassName,
            "group-hover/bento:translate-x-2 transition duration-200 relative md:h-full min-h-20 flex flex-col px-5 p-5 lg:p-10"
          )}
        >
          {description && (
            <div className="font-sans font-extralight min-h-15 md:max-w-32 md:text-xs lg:text-base text-sm text-green-600 z-10">
              {description}
              <div className="rounded-full"></div>
            </div>
          )}

          <div className="font-sans text-lg lg:text-3xl max-w-96 font-bold z-10 text-black-500">
            {title}
          </div>

          
        </div>
      </div>
    </div>
  );
};
