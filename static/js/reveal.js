document.addEventListener("DOMContentLoaded", () => {
  const deferredImages = document.querySelectorAll("img[data-src]");
  const imageObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const image = entry.target;
        image.src = image.dataset.src;
        image.removeAttribute("data-src");
        imageObserver.unobserve(image);
      }
    },
    { rootMargin: "500px 0px", threshold: 0 }
  );
  deferredImages.forEach((image) => imageObserver.observe(image));

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const panels = document.querySelectorAll(".split-section");
  if (reduced) {
    panels.forEach((el) => el.classList.add("is-in"));
    return;
  }
  panels.forEach((el) => el.classList.add("will-reveal"));
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.08, rootMargin: "0px 0px 12% 0px" }
  );
  panels.forEach((el) => io.observe(el));
});
