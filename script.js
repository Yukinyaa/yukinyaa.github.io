const backgroundImages = [
  "https://github.com/user-attachments/assets/761ec3d0-5ef8-4edd-b596-22c3288be7d8", //루루도 도촬
  "https://github.com/user-attachments/assets/b55b3a15-9793-4641-b813-fc688e3d5dfb", //강남스타일
  "https://github.com/user-attachments/assets/e4008e44-955e-48cf-a4cb-8a6f63729232", //합삐
];

const layerA = document.querySelector(".bg-layer-a");
const layerB = document.querySelector(".bg-layer-b");

if (layerA && layerB) {
  let activeLayer = layerA;
  let hiddenLayer = layerB;
  let currentIndex = 0;

  const applyImage = (layer, imagePath) => {
    layer.style.backgroundImage = `url("${imagePath}")`;
  };

  applyImage(activeLayer, backgroundImages[currentIndex]);

  if (backgroundImages.length > 1) {
    setInterval(() => {
      currentIndex = (currentIndex + 1) % backgroundImages.length;
      applyImage(hiddenLayer, backgroundImages[currentIndex]);

      hiddenLayer.style.opacity = "1";
      activeLayer.style.opacity = "0";

      [activeLayer, hiddenLayer] = [hiddenLayer, activeLayer];
    }, 6500);
  }
}
