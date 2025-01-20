"use client";
import React, { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  z: number;
  size: number;
  brightness: number;
}

const StarBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let speed = 0.5;
    const maxDepth = 32;

    // Create stars with 3D position
    const createStars = (width: number, height: number): Star[] => {
      return Array.from({ length: 400 }, () => ({
        x: Math.random() * width - width / 2,
        y: Math.random() * height - height / 2,
        z: Math.random() * maxDepth,
        size: Math.random() * 2,
        brightness: Math.random() * 0.8 + 0.2,
      }));
    };

    // Set canvas size and create initial stars
    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      stars = createStars(canvas.width, canvas.height);
    };

    let stars = createStars(canvas.width, canvas.height);
    setCanvasSize();

    const moveStars = () => {
      stars.forEach((star) => {
        // Move star closer (smaller z = closer)
        star.z -= speed;

        // Reset star to far distance if it gets too close
        if (star.z <= 0) {
          star.z = maxDepth;
          star.x = Math.random() * canvas.width - canvas.width / 2;
          star.y = Math.random() * canvas.height - canvas.height / 2;
        }
      });
    };

    const drawStars = () => {
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      stars.forEach((star) => {
        // Calculate star position with perspective
        const scale = maxDepth / (maxDepth - star.z);
        const x = star.x * scale + centerX;
        const y = star.y * scale + centerY;

        // Only draw if star is in bounds
        if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
          const size = star.size * scale;
          const opacity = star.brightness * (1 - star.z / maxDepth);

          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
          ctx.fill();

          // Add motion trail
          if (size > 1) {
            ctx.beginPath();
            ctx.arc(x, y, size * 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.1})`;
            ctx.fill();
          }
        }
      });
    };

    // Animation loop
    const animate = () => {
      moveStars();
      drawStars();
      requestAnimationFrame(animate);
    };

    // Handle window resize
    window.addEventListener("resize", setCanvasSize);

    // Start animation
    animate();

    // Cleanup
    return () => {
      window.removeEventListener("resize", setCanvasSize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full -z-10"
    />
  );
};

export default StarBackground;
