const backgroundImages = [
  "https://github.com/user-attachments/assets/5222b6dd-5ee1-42d4-bf89-da9e7632ac70",
  "https://github.com/user-attachments/assets/c06357a1-5e7c-4d7f-80b8-490c49019f5f",
  "https://github.com/user-attachments/assets/d40303c8-bde0-4236-97d6-ef66505c8daa",
  "https://github.com/user-attachments/assets/3db4b249-fe7b-420c-8ac1-0bff82c46e2c",
  "https://github.com/user-attachments/assets/86e5391b-fe04-4b40-8ca9-b8a826b4d5b4",
  "https://github.com/user-attachments/assets/51b1cdf8-8f5c-4573-8b07-d0cf415532d2",
  "https://github.com/user-attachments/assets/cefaad36-4162-4f81-9062-043d1fc64834",
  "https://github.com/user-attachments/assets/9f8cf7b8-239b-4068-8d06-94d02b7e1003",
  "https://github.com/user-attachments/assets/7034f156-28ff-4adb-975f-9c954d1e18fd",
  "https://github.com/user-attachments/assets/45c1f19a-1f0a-4107-a891-6ed49108f1fd",
  "https://github.com/user-attachments/assets/2a7f0373-824d-4153-94aa-03c58adcfcec",
  "https://github.com/user-attachments/assets/376b15fa-f0af-41e1-a751-35dd6e9901c3",
  "https://github.com/user-attachments/assets/96d59e1d-25ce-41fa-a890-124b099a2c8a",
];

const layerA = document.querySelector(".bg-layer-a");
const layerB = document.querySelector(".bg-layer-b");

const SLIDE_INTERVAL_MS = 3600;
const SLOW_LOAD_WARN_MS = 2500;
const panPresets = [
  { sx: "-2%", sy: "-1%", ex: "2%", ey: "1%" },
  { sx: "2%", sy: "-1%", ex: "-2%", ey: "1%" },
  { sx: "-1%", sy: "2%", ex: "1%", ey: "-2%" },
  { sx: "1%", sy: "-2%", ex: "-1%", ey: "2%" },
];

const preloadImage = (src) =>
  new Promise((resolve, reject) => {
    const img = new Image();
    const startedAt = performance.now();
    const slowTimer = setTimeout(() => {
      console.warn(
        `[slideshow] Slow image load (> ${SLOW_LOAD_WARN_MS}ms): ${src}`
      );
    }, SLOW_LOAD_WARN_MS);

    img.decoding = "async";
    img.onload = () => {
      clearTimeout(slowTimer);
      const elapsed = Math.round(performance.now() - startedAt);
      console.info(`[slideshow] Image loaded in ${elapsed}ms: ${src}`);
      resolve(src);
    };
    img.onerror = () => {
      clearTimeout(slowTimer);
      const elapsed = Math.round(performance.now() - startedAt);
      console.error(`[slideshow] Image failed after ${elapsed}ms: ${src}`);
      reject(src);
    };
    img.src = src;
  });

const applyImage = (layer, imagePath) => {
  layer.style.backgroundImage = `url("${imagePath}")`;
};

const applyPan = (layer, step) => {
  const preset = panPresets[step % panPresets.length];
  const scaleOffset = (step % 3) * 0.01;
  const startScale = (1.07 + scaleOffset).toFixed(2);
  const endScale = (1.13 + scaleOffset).toFixed(2);

  layer.style.setProperty("--pan-start-x", preset.sx);
  layer.style.setProperty("--pan-start-y", preset.sy);
  layer.style.setProperty("--pan-end-x", preset.ex);
  layer.style.setProperty("--pan-end-y", preset.ey);
  layer.style.setProperty("--pan-start-scale", startScale);
  layer.style.setProperty("--pan-end-scale", endScale);

  // Restart keyframe animation when the layer becomes visible.
  layer.style.animation = "none";
  layer.offsetHeight;
  layer.style.animation = `bg-pan ${SLIDE_INTERVAL_MS}ms linear forwards`;
};

if (layerA && layerB) {
  let activeLayer = layerA;
  let hiddenLayer = layerB;
  let currentIndex = -1;
  let motionStep = 0;
  const loadedImages = new Set();

  const warmUpImages = () => {
    backgroundImages.forEach((src, index) => {
      preloadImage(src)
        .then((loaded) => {
          loadedImages.add(loaded);

          // Start from the first successfully loaded image.
          if (currentIndex === -1) {
            currentIndex = index;
            applyImage(activeLayer, backgroundImages[currentIndex]);
            applyPan(activeLayer, motionStep++);
          }
        })
        .catch(() => {
          // Keep slideshow running even if one image fails to load.
        });
    });
  };

  const getNextLoadedImageIndex = (startIndex) => {
    for (let offset = 1; offset <= backgroundImages.length; offset += 1) {
      const candidate = (startIndex + offset) % backgroundImages.length;
      if (loadedImages.has(backgroundImages[candidate])) {
        return candidate;
      }
    }

    return startIndex;
  };

  const startSlideshow = () => {
    warmUpImages();

    if (backgroundImages.length > 1) {
      setInterval(() => {
        if (currentIndex === -1) {
          return;
        }

        const nextIndex = getNextLoadedImageIndex(currentIndex);
        if (nextIndex === currentIndex) {
          // Keep subtle motion even when only one image is available.
          applyPan(activeLayer, motionStep++);
          return;
        }

        currentIndex = nextIndex;
        applyImage(hiddenLayer, backgroundImages[currentIndex]);
        applyPan(hiddenLayer, motionStep++);

        hiddenLayer.style.opacity = "1";
        activeLayer.style.opacity = "0";

        [activeLayer, hiddenLayer] = [hiddenLayer, activeLayer];
      }, SLIDE_INTERVAL_MS);
    }
  };

  startSlideshow();
}
