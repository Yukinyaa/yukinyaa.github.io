const backgroundImages = [
  "https://github.com/user-attachments/assets/761ec3d0-5ef8-4edd-b596-22c3288be7d8", //루루도 도촬
  "https://github.com/user-attachments/assets/b55b3a15-9793-4641-b813-fc688e3d5dfb", //강남스타일
  //"https://github.com/user-attachments/assets/e4008e44-955e-48cf-a4cb-8a6f63729232", //합삐
  "https://github.com/user-attachments/assets/21e36e30-434b-4851-b2bd-bda30cb6c089", //자전거
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
