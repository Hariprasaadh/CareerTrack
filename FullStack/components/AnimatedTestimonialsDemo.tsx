import { AnimatedTestimonials } from "@/components/ui/animated-testimonials";

export function AnimatedTestimonialsDemo() {
  const testimonials = [
    {
      quote:
        "As the web developer, I focused on every detail to ensure that the website was both visually appealing and user-friendly ",
      name: "Prasanth",
      designation: "FullStack-Developer",
      src: "/Prasanth.jpeg",
    },
    //As a machine learning developer, I dedicated myself to designing and implementing intelligent models that address our specific challenges. The solutions we’ve developed have streamlined our workflow and delivered exactly the results we were striving for.
    {
      quote:
        "As a machine learning developer, I concentrated on integrating precise algorithma and innovative solutions that have revolutionized our workflow. This is exactly the transformation we aimed for, it is significantly improved our efficiency and decision-making",
      name: "Prannavakhanth",
      designation: "AIML Member",
      src: "/Prannavakhanth.jpg",
    },
    {
      quote:
        "As a machine learning developer, I dedicated myself to designing and implementing intelligent models that address our specific challenges. The solutions we have developed have streamlined our workflow and delivered exactly the results we were striving for.",
      name: "Hariprasaadh",
      designation: "AIML Member",
      src: "/Hariprasaadh.jpg",
    },

    //As a front-end web developer, I focused on refining the user interface and incorporating cutting-edge features to enhance the website’s usability. The improvements have not only transformed our workflow but also delivered the seamless experience we envisioned.
    {
      quote: "As a front-end web developer, I focused on refining the user interface and incorporating cutting-edge features to enhance the websites usability. Thr improvements have not only transformed our workflow but also delivered the seamless experience we envisioned",
      name: "venkatakrishnan",
      designation: "Front-end developer",
      src: "/venkatakrishnan.jpeg",
    },
  ];
  return <AnimatedTestimonials testimonials={testimonials} />;

  // so this is a website where i can go 

  
}
