import { FocusCards } from "@/components/ui/focus-cards";

export function FocusCardsDemo() {
  const cards = [
    {
      title: "Forest Adventure",
      src: "/pr1.jpg",
    },
    {
      title: "Valley of life",
      src: "/pr1.jpg",
    },
    {
      title: "Sala behta hi jayega",
      src: "/pr1.jpg",
    },
    {
      title: "Camping is for pros",
      src: "/pr1.jpg",
    },
    {
      title: "The road not taken",
      src: "/pr1.jpg",
    },
    {
      title: "The First Rule",
      src: "/pr1.jpg",
    },
  ];

  return <FocusCards cards={cards} />;
}
